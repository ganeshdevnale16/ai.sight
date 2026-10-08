"""Optional AI enrichment (Steps 2 & 5 with a classifier instead of keywords).

Sends a sample of actionable items to the selected model in batches and stores, per item:
sentiment, emotion, lenses (from the taxonomy, or a new emergent topic) and flags.
Items outside the sample keep the keyword / provider values.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .llm import LLMError, get_provider
from .taxonomy import EMOTION_ORDER, lens_list

SYSTEM = """You label social-media and news items about a brand for a reputation analyst.
For each item return: sentiment toward the brand (-1 negative, 0 neutral, 1 positive), the dominant emotion,
the perception lenses it belongs to (zero or more, ONLY from the allowed list), flags, and — only if no lens fits
and the item is about a clear, specific topic — a short 2-4 word emergent topic name.
Judge sentiment toward the brand, not the general mood. News that reports a fine, lawsuit or failure is negative."""


def ai_enrich(df: pd.DataFrame, provider_name: str | None, sample_size: int, lenses_cfg, log) -> pd.DataFrame:
    provider = get_provider(provider_name)
    lenses = [L["name"] for L in lens_list(lenses_cfg)]
    cand = df[df["actionable"]]
    if not len(cand):
        return df
    n = min(sample_size, len(cand))
    top = cand.assign(_w=cand["reach"].rank(pct=True) + cand["engagement"].rank(pct=True)).sort_values("_w", ascending=False)
    k_top = int(n * 0.6)
    chosen = top.head(k_top)
    rest = top.iloc[k_top:]
    if len(rest) and n - k_top > 0:
        chosen = pd.concat([chosen, rest.sample(min(n - k_top, len(rest)), random_state=7)])
    ids = chosen.index.tolist()
    batches = [ids[i:i + 40] for i in range(0, len(ids), 40)]
    log(f"AI classification: {len(ids)} items in {len(batches)} batches")
    schema = ('{"items":[{"id":<int>,"sentiment":-1|0|1,"emotion":"' + "|".join(EMOTION_ORDER + ["neutral-info"])
              + '","lenses":[...],"flags":["complaint"|"risk"|"misinformation"],"emergent_topic":""}]}')

    def run(batch):
        lines = "\n".join(f"[{i}] {str(df.at[i, 'text'])[:500]}" for i in batch)
        prompt = (f"Allowed lenses: {lenses}\nFlags: complaint (a customer grievance), risk (legal, regulatory, "
                  f"ombudsman, police language), misinformation (rumour or false claim).\n"
                  f"Return JSON exactly in this shape: {schema}\n\nITEMS:\n{lines}")
        try:
            return provider.complete_json(SYSTEM, [{"type": "text", "text": prompt}], max_tokens=6000).get("items", [])
        except LLMError as e:
            log(f"A batch failed and kept keyword labels: {str(e)[:120]}")
            return []

    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(run, batches):
            for it in res:
                try:
                    i = int(it.get("id"))
                except (TypeError, ValueError):
                    continue
                if i not in df.index:
                    continue
                s = it.get("sentiment")
                if s in (-1, 0, 1):
                    df.at[i, "sentiment"] = float(s)
                    df.at[i, "sentiment_score"] = float(s)
                    df.at[i, "sentiment_source"] = "ai"
                emo = it.get("emotion")
                if emo in EMOTION_ORDER + ["neutral-info"]:
                    df.at[i, "emotion"] = emo
                ls = [x for x in (it.get("lenses") or []) if x in lenses]
                df.at[i, "lenses"] = ls
                df.at[i, "flags"] = [f for f in (it.get("flags") or []) if f in ("complaint", "risk", "misinformation")]
                if not ls and it.get("emergent_topic"):
                    df.at[i, "emergent_topic"] = str(it["emergent_topic"])[:40]
            done += 1
            log(f"AI classification batch {done}/{len(batches)} done")
    return df
