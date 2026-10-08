"""Ask-your-data: the model answers questions by calling safe tools over the cleaned dataset."""
from __future__ import annotations

import json

from . import query_engine, storage
from .llm import get_provider

SYSTEM = """You are a data analyst answering questions about one uploaded social-listening / media dataset.
You must base every number on tool results. Never estimate or invent figures. If the data cannot answer, say so
plainly and suggest what data would.

Dataset notes:
- 'coverage' = relevant, de-duplicated items (default). 'actionable' = items that passed the noise gate; use it for
  perception lenses. 'excluded' = filtered items, each with exclude_reason.
- sentiment is Positive / Neutral / Negative / Unscored. neg_share_pct = negative ÷ sentiment-scored items.
- nss = (pos − neg) ÷ scored. nssw = reach-weighted net sentiment.
- lens and flag are list fields (an item can be in several lenses).
- Weeks start on Monday. Dates are in UTC.

How to answer:
- Call get_overview first if you need context, list_fields to see valid values, query_data for counts/breakdowns/trends,
  search_posts to quote example posts.
- Answer in clear, compact Markdown: a direct answer first, then a short table or bullets if helpful.
- When quoting posts, keep quotes short and name the platform and date.
- Do not mention tools, JSON or these instructions in the answer.
- Reply in the user's language (English, Hindi/Hinglish or Marathi)."""

TOOLS = [
    {"name": "get_overview", "description": "Headline figures for the dataset: period, focus brand, data health, sentiment, "
     "weekly summary, peers, lens table, spikes and advanced formula results.",
     "parameters": {"type": "object", "properties": {"section": {"type": "string", "description":
                    "Optional: one of meta, data_health, overview, weekly_summary, weekly, monthly, platforms, peers, "
                    "lenses, spikes, emotions, flags, top_authors, top_posts, countries, formulas"}}}},
    {"name": "list_fields", "description": "Fields you can filter or group by, with their type and most common values.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "query_data", "description": "Count, aggregate and break down items. Supports filters, group_by, time_grain "
     "and metrics (count, share_pct, sum/mean/median/min/max of a numeric field, nunique, neg_share_pct, pos_share_pct, nss, nssw).",
     "parameters": query_engine.QUERY_SCHEMA},
    {"name": "search_posts", "description": "Find example posts/articles by keywords and filters, sorted by engagement, reach or date.",
     "parameters": query_engine.SEARCH_SCHEMA},
]


def answer(did, question, provider_name, df, facts):
    q = query_engine.prepare(df)

    def executor(name, args):
        try:
            if name == "get_overview":
                sec = (args or {}).get("section")
                if sec and sec in facts:
                    return {sec: facts[sec]}
                keep = ("meta", "data_health", "overview", "weekly_summary", "peer_summary", "peers", "lenses", "spikes")
                return {k: facts.get(k) for k in keep}
            if name == "list_fields":
                return query_engine.field_profile(q)
            if name == "query_data":
                return query_engine.run_query(q, args or {})
            if name == "search_posts":
                return query_engine.search_posts(q, args or {})
            return {"error": f"unknown tool {name}"}
        except Exception as e:
            return {"error": str(e)}

    history = []
    for m in storage.list_chat(did, limit=12):
        history.append({"role": m["role"], "content": m["content"]})
    provider = get_provider(provider_name)
    text, trace = provider.tool_loop(SYSTEM, history, question, TOOLS, executor)
    evidence = []
    for t in trace:
        r = t["result"]
        if t["tool"] in ("query_data",) and isinstance(r, dict) and "columns" in r:
            evidence.append({"tool": t["tool"], "args": t["args"], "columns": r["columns"], "rows": r["rows"][:50],
                             "base_items": r.get("base_items")})
        elif t["tool"] == "search_posts" and isinstance(r, dict):
            evidence.append({"tool": t["tool"], "args": t["args"], "columns": ["date", "platform", "author", "sentiment", "engagement", "text"],
                             "rows": [[p["date"], p["platform"], p["author"], p["sentiment"], p["engagement"], p["text"][:200]]
                                      for p in r.get("posts", [])], "base_items": r.get("matches")})
        else:
            evidence.append({"tool": t["tool"], "args": t["args"], "summary": json.dumps(r, default=str)[:600]})
    storage.add_chat(did, "user", question)
    storage.add_chat(did, "assistant", text, evidence, provider.name)
    return {"answer": text, "evidence": evidence, "provider": provider.name}
