"""One interface over two providers: Anthropic Claude and Azure OpenAI.

Keys are read from the encrypted settings store (Settings tab) and fall back to environment variables.
They are never sent to the browser.
"""
from __future__ import annotations

import json
import os
import re
import time

from . import storage
from .config import (DEFAULT_AZURE_API_VERSION, DEFAULT_AZURE_DEPLOYMENT, DEFAULT_AZURE_ENDPOINT,
                     DEFAULT_CLAUDE_MODEL)

PROVIDERS = ("claude", "azure_openai")


class LLMError(RuntimeError):
    pass


def provider_status() -> dict:
    s = storage.get_settings(include_secrets=True)
    claude_key = s.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
    az_key = s.get("azure_openai_api_key") or os.getenv("AZURE_OPENAI_API_KEY", "")
    az_ep = s.get("azure_openai_endpoint") or DEFAULT_AZURE_ENDPOINT
    return {"claude": bool(claude_key), "azure_openai": bool(az_key and az_ep)}


def default_provider() -> str:
    s = storage.get_settings()
    p = s.get("default_provider")
    st = provider_status()
    if p in PROVIDERS and st.get(p):
        return p
    return next((k for k in PROVIDERS if st.get(k)), "claude")


def get_provider(name: str | None = None) -> "BaseProvider":
    name = name or default_provider()
    s = storage.get_settings(include_secrets=True)
    if name == "claude":
        key = s.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise LLMError("Claude is not set up. Add the Anthropic API key in Settings.")
        return ClaudeProvider(key, s.get("anthropic_model") or DEFAULT_CLAUDE_MODEL)
    if name == "azure_openai":
        key = s.get("azure_openai_api_key") or os.getenv("AZURE_OPENAI_API_KEY", "")
        ep = s.get("azure_openai_endpoint") or DEFAULT_AZURE_ENDPOINT
        if not (key and ep):
            raise LLMError("Azure OpenAI is not set up. Add the endpoint and API key in Settings.")
        return AzureProvider(key, ep, s.get("azure_openai_deployment") or DEFAULT_AZURE_DEPLOYMENT,
                             s.get("azure_openai_api_version") or DEFAULT_AZURE_API_VERSION)
    raise LLMError(f"Unknown provider '{name}'")


def parse_json(text: str):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except ValueError:
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except ValueError:
                pass
    raise LLMError("The model did not return valid JSON.")


class BaseProvider:
    name = "base"

    def complete(self, system: str, parts: list, max_tokens: int = 4000, json_mode: bool = False) -> str:
        raise NotImplementedError

    def complete_json(self, system: str, parts: list, max_tokens: int = 8000):
        text = self.complete(system, parts, max_tokens=max_tokens, json_mode=True)
        try:
            return parse_json(text)
        except LLMError:
            retry = parts + [{"type": "text", "text": "Your previous reply was not valid JSON. "
                              "Reply again with ONLY the JSON object, no prose, no code fences."}]
            return parse_json(self.complete(system, retry, max_tokens=max_tokens, json_mode=True))

    def tool_loop(self, system, history, question, tools, executor, max_rounds=8):
        raise NotImplementedError

    def ping(self) -> str:
        return self.complete("You are a connectivity check.", [{"type": "text", "text": "Reply with the word OK."}],
                             max_tokens=10)


# ------------------------------------------------------------------ Claude
class ClaudeProvider(BaseProvider):
    name = "claude"

    def __init__(self, key, model):
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("The 'anthropic' package is not installed.") from e
        self.client = anthropic.Anthropic(api_key=key, timeout=600, max_retries=3)
        self.model = model

    @staticmethod
    def _parts(parts):
        out = []
        for p in parts:
            if p["type"] == "text":
                out.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                out.append({"type": "image", "source": {"type": "base64", "media_type": p["media_type"],
                                                         "data": p["data"]}})
        return out

    def _call(self, **kw):
        try:
            return self.client.messages.create(model=self.model, **kw)
        except Exception as e:  # surface a readable message
            raise LLMError(f"Claude request failed: {e}") from e

    def complete(self, system, parts, max_tokens=4000, json_mode=False):
        if json_mode:
            system = system + "\nRespond with a single valid JSON object only."
        resp = self._call(max_tokens=max_tokens, system=system,
                          messages=[{"role": "user", "content": self._parts(parts)}])
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def tool_loop(self, system, history, question, tools, executor, max_rounds=8):
        tdefs = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        msgs = [{"role": m["role"], "content": m["content"]} for m in history] + [{"role": "user", "content": question}]
        trace = []
        for _ in range(max_rounds):
            resp = self._call(max_tokens=4000, system=system, tools=tdefs, messages=msgs)
            blocks, calls = [], []
            for b in resp.content:
                if b.type == "text":
                    blocks.append({"type": "text", "text": b.text})
                elif b.type == "tool_use":
                    blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
                    calls.append(b)
            msgs.append({"role": "assistant", "content": blocks})
            if resp.stop_reason != "tool_use" or not calls:
                return "".join(b["text"] for b in blocks if b["type"] == "text"), trace
            results = []
            for c in calls:
                out = executor(c.name, c.input or {})
                trace.append({"tool": c.name, "args": c.input, "result": out})
                results.append({"type": "tool_result", "tool_use_id": c.id,
                                "content": json.dumps(out, default=str)[:30000]})
            msgs.append({"role": "user", "content": results})
        return "I ran out of analysis steps before finishing. Try a narrower question.", trace


# ------------------------------------------------------------------ Azure OpenAI
class AzureProvider(BaseProvider):
    name = "azure_openai"

    def __init__(self, key, endpoint, deployment, api_version):
        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise LLMError("The 'openai' package is not installed.") from e
        self.client = AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=api_version,
                                  timeout=600, max_retries=3)
        self.deployment = deployment

    @staticmethod
    def _parts(parts):
        out = []
        for p in parts:
            if p["type"] == "text":
                out.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                out.append({"type": "image_url", "image_url": {"url": f"data:{p['media_type']};base64,{p['data']}"}})
        return out

    def _call(self, max_tokens, **kw):
        try:
            return self.client.chat.completions.create(model=self.deployment, max_tokens=max_tokens, **kw)
        except Exception as e:
            if "max_completion_tokens" in str(e):
                try:
                    return self.client.chat.completions.create(model=self.deployment,
                                                               max_completion_tokens=max_tokens, **kw)
                except Exception as e2:
                    raise LLMError(f"Azure OpenAI request failed: {e2}") from e2
            raise LLMError(f"Azure OpenAI request failed: {e}") from e

    def complete(self, system, parts, max_tokens=4000, json_mode=False):
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        if json_mode:
            system = system + "\nRespond with a single valid JSON object only."
        resp = self._call(max_tokens, messages=[{"role": "system", "content": system},
                                                {"role": "user", "content": self._parts(parts)}],
                          temperature=0.2, **kw)
        return resp.choices[0].message.content or ""

    def tool_loop(self, system, history, question, tools, executor, max_rounds=8):
        tdefs = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                   "parameters": t["parameters"]}} for t in tools]
        msgs = [{"role": "system", "content": system}] + \
               [{"role": m["role"], "content": m["content"]} for m in history] + \
               [{"role": "user", "content": question}]
        trace = []
        for _ in range(max_rounds):
            resp = self._call(4000, messages=msgs, tools=tdefs, temperature=0.1)
            m = resp.choices[0].message
            if not m.tool_calls:
                return m.content or "", trace
            msgs.append({"role": "assistant", "content": m.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}} for tc in m.tool_calls]})
            for tc in m.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except ValueError:
                    args = {}
                out = executor(tc.function.name, args)
                trace.append({"tool": tc.function.name, "args": args, "result": out})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, default=str)[:30000]})
        return "I ran out of analysis steps before finishing. Try a narrower question.", trace
