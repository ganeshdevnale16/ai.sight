"""Report generation.

A report is a *blueprint* (ordered sections with KPIs, visuals and AI writing slots).
1. Python resolves every KPI, chart and table from the cleaned data (numbers are never invented).
2. The model writes the narrative for each section from those resolved figures.
3. Every number in the narrative is checked against the source figures.
4. The result is rendered as a standalone HTML report.

The built-in blueprint mirrors a reputation-risk scan. A blueprint can also be learned from a
user's sample report (see build_template_blueprint).
"""
from __future__ import annotations

import html
import json
import re
from datetime import datetime

import pandas as pd

from . import query_engine
from .analytics import metric_catalog

# ======================================================================== built-in visuals
BUILTINS = {
    "weekly_sentiment": "Stacked weekly sentiment mix (neg/neu/pos items) with negative share line",
    "weekly_volume": "Weekly item volume line",
    "monthly_sentiment": "Monthly negative share and net sentiment",
    "platform_mix": "Share of items vs share of engagement by platform",
    "channel_mix": "Social vs news & media items",
    "sov_chart": "Share of voice by brand / entity",
    "peers_table": "Peer benchmark table: items, share of voice, negative share, social engagement",
    "lens_table": "Perception lens table: comments, share, pos/neu/neg, reach-weighted net sentiment, emotion, momentum, verdict",
    "lens_priority": "Lens priority score ranking",
    "spikes_table": "Significant spike days: items, peak-to-mean, negative share, top terms",
    "top_authors_negative": "Most active authors on negative items",
    "top_authors": "Most active authors overall",
    "top_posts": "Most engaged posts",
    "countries": "Items by country",
    "emotions": "Emotion mix of actionable items",
    "formulas_table": "Advanced signals: coordination C, newsroom escalation E, journalist trend, vacuum V, pull-through PT, resilience R",
    "data_health": "Noise gate: items excluded by reason",
    "author_mix": "Media vs individual vs brand-owned authors",
}

C_NEG, C_NEU, C_POS, C_INK, C_ACC = "#B42318", "#A9B4C2", "#1E7B4E", "#16202B", "#0F5C63"


def _fmt(v, f="num"):
    if v is None:
        return "—"
    try:
        if f == "pct":
            return f"{float(v):.1f}%"
        if f == "x":
            return f"{float(v):.1f}×"
        if f == "dec":
            return f"{float(v):+.2f}"
        if f == "text":
            return str(v)
        fv = float(v)
        return f"{int(round(fv)):,}" if abs(fv) >= 100 or fv == int(fv) else f"{fv:,.1f}"
    except (TypeError, ValueError):
        return str(v)


def builtin(name, facts):
    W = [w for w in facts.get("weekly", []) if not w.get("partial")]
    if name == "weekly_sentiment" and W:
        return {"kind": "chart", "type": "stacked", "title": "Weekly sentiment mix",
                "labels": [w["week"] for w in W],
                "datasets": [{"label": "Negative", "data": [w["neg"] for w in W], "color": C_NEG},
                             {"label": "Neutral", "data": [w["neu"] for w in W], "color": C_NEU},
                             {"label": "Positive", "data": [w["pos"] for w in W], "color": C_POS},
                             {"label": "Negative share %", "data": [w["neg_share_pct"] for w in W], "color": C_INK,
                              "line": True, "axis": "y2"}],
                "note": "Partial final week excluded." if facts.get("weekly_summary", {}).get("partial_week_excluded") else ""}
    if name == "weekly_volume" and W:
        return {"kind": "chart", "type": "line", "title": "Weekly items", "labels": [w["week"] for w in W],
                "datasets": [{"label": "Items", "data": [w["items"] for w in W], "color": C_ACC}]}
    if name == "monthly_sentiment" and facts.get("monthly"):
        M = facts["monthly"]
        return {"kind": "chart", "type": "bar", "title": "Negative share by month", "labels": [m["month"] for m in M],
                "datasets": [{"label": "Negative share %", "data": [m["neg_share_pct"] for m in M], "color": C_NEG}]}
    if name == "platform_mix" and facts.get("platforms"):
        P = facts["platforms"][:8]
        return {"kind": "chart", "type": "hbar", "title": "Items vs engagement by platform (share %)",
                "labels": [p["platform"] for p in P],
                "datasets": [{"label": "Share of items %", "data": [p["items_share_pct"] for p in P], "color": C_ACC},
                             {"label": "Share of engagement %", "data": [p["engagement_share_pct"] for p in P], "color": "#C7A13A"}]}
    if name == "channel_mix" and facts.get("channels"):
        C = facts["channels"]
        return {"kind": "chart", "type": "hbar", "title": "Items by channel", "labels": [c["channel"] for c in C],
                "datasets": [{"label": "Items", "data": [c["items"] for c in C], "color": C_ACC}]}
    if name == "sov_chart" and facts.get("peers"):
        P = facts["peers"]
        return {"kind": "chart", "type": "hbar", "title": "Share of voice and negative share",
                "labels": [p["entity"] for p in P],
                "datasets": [{"label": "Share of voice %", "data": [p["share_of_voice_pct"] for p in P], "color": C_ACC},
                             {"label": "Negative share %", "data": [p["neg_share_pct"] for p in P], "color": C_NEG}]}
    if name == "peers_table" and facts.get("peers"):
        return {"kind": "table", "title": "Peer benchmark",
                "columns": ["Brand", "Items", "Share of voice", "Negative items", "Negative share", "Social posts",
                            "Social engagement", "Eng. / post"],
                "rows": [[p["entity"], _fmt(p["items"]), _fmt(p["share_of_voice_pct"], "pct"), _fmt(p["negative_items"]),
                          _fmt(p["neg_share_pct"], "pct"), _fmt(p["social_posts"]), _fmt(p["social_engagement"]),
                          _fmt(p["eng_per_post"])] for p in facts["peers"]],
                "highlight_first_col": [p["entity"] for p in facts["peers"] if p["is_primary"]]}
    if name == "lens_table" and facts.get("lenses"):
        L = [r for r in facts["lenses"] if r.get("volume")]
        return {"kind": "table", "title": "Perception lenses (actionable items, ranked by priority)",
                "columns": ["Lens", "Comments", "Share", "Pos / Neu / Neg %", "Net sent. (reach-wtd)", "Emotion",
                            "Momentum", "Denotes"],
                "rows": [[r["lens"], _fmt(r["volume"]), _fmt(r["share_pct"], "pct"),
                          f"{_fmt(r['pos_pct'])} / {_fmt(r['neu_pct'])} / {_fmt(r['neg_pct'])}",
                          _fmt(r["nssw"], "dec"), r["dominant_emotion"], r["momentum"],
                          r["denotes"].split(" — ", 1)[-1]] for r in L],
                "tone_col": 4}
    if name == "lens_priority" and facts.get("lenses"):
        L = [r for r in facts["lenses"] if r.get("volume")][:10]
        return {"kind": "chart", "type": "hbar", "title": "Lens priority score", "labels": [r["lens"] for r in L],
                "datasets": [{"label": "Priority", "data": [r["priority"] for r in L], "color": C_NEG}]}
    if name == "spikes_table" and facts.get("spikes"):
        return {"kind": "table", "title": "Spike days (peak-to-mean ≥ 1.5)",
                "columns": ["Date", "Items", "Peak / mean", "Sources", "Negative share", "Top terms"],
                "rows": [[s["date"], _fmt(s["items"]), _fmt(s["peak_to_mean"], "x"), _fmt(s["sources"]),
                          _fmt(s["neg_share_pct"], "pct"), ", ".join(s["top_terms"][:5])] for s in facts["spikes"]]}
    if name in ("top_authors_negative", "top_authors"):
        A = facts.get("top_authors", {}).get("negative" if name == "top_authors_negative" else "all", [])
        if A:
            return {"kind": "table", "title": "Most active authors on negative items" if name == "top_authors_negative"
                    else "Most active authors", "columns": ["Author", "Type", "Platform", "Items", "Engagement"],
                    "rows": [[a["author"], a["author_type"], a["platform"], _fmt(a["items"]), _fmt(a["engagement"])] for a in A]}
    if name == "top_posts" and facts.get("top_posts"):
        return {"kind": "table", "title": "Most engaged items",
                "columns": ["Date", "Platform", "Author", "Sentiment", "Engagement", "Text"],
                "rows": [[p["date"], p["platform"], p["author"], p["sentiment"], _fmt(p["engagement"]), p["text"][:160]]
                         for p in facts["top_posts"][:8]]}
    if name == "countries" and facts.get("countries"):
        C = facts["countries"]
        return {"kind": "chart", "type": "hbar", "title": "Items by country", "labels": [c["country"] for c in C],
                "datasets": [{"label": "Items", "data": [c["items"] for c in C], "color": C_ACC}]}
    if name == "emotions" and facts.get("emotions"):
        E = facts["emotions"]
        return {"kind": "chart", "type": "hbar", "title": "Emotion mix", "labels": [e["emotion"] for e in E],
                "datasets": [{"label": "Share %", "data": [e["share_pct"] for e in E], "color": C_ACC}]}
    if name == "author_mix" and facts.get("author_mix"):
        A = facts["author_mix"]
        return {"kind": "chart", "type": "hbar", "title": "Who is talking", "labels": [a["author_type"] for a in A],
                "datasets": [{"label": "Share %", "data": [a["share_pct"] for a in A], "color": C_ACC}]}
    if name == "data_health" and facts.get("data_health"):
        dh = facts["data_health"]
        rows = [["Ingested", _fmt(dh["total_ingested"])], ["Coverage set (relevant, de-duplicated)", _fmt(dh["coverage_items"])],
                ["Actionable (used for lenses)", _fmt(dh["actionable_items"])],
                ["Signal rate", _fmt(dh["signal_rate_pct"], "pct")]]
        rows += [[f"Excluded — {k}", _fmt(v)] for k, v in dh["excluded_by_reason"].items()]
        return {"kind": "table", "title": "Relevance & noise gate", "columns": ["Measure", "Items"], "rows": rows}
    if name == "formulas_table" and facts.get("formulas"):
        F = facts["formulas"]
        rows = []
        oc = F.get("organic_vs_coordinated") or {}
        rows.append(["Organic vs coordinated (C)", _fmt(oc.get("C"), "dec").lstrip("+"), oc.get("reading", "—"),
                     f"dup {oc.get('duplication')}, burst {oc.get('burstiness')}, anomaly {oc.get('account_anomaly')}, Gini {oc.get('reach_gini')}"])
        ne = F.get("newsroom_escalation")
        if ne:
            rows.append(["Newsroom escalation (E)", _fmt(ne.get("E"), "dec").lstrip("+"), ne.get("reading", "—"),
                         f"week of {ne.get('week')}; journalist engagement {ne.get('journo_engage')}, velocity z {ne.get('velocity_z')}"])
        jt = (F.get("journalist_trend") or {}).get("windows", {})
        if jt:
            rows.append(["Journalist warming / cooling", " · ".join(f"{k}: {v['trend']}" for k, v in jt.items()),
                         next(iter(jt.values()))["trend"], "slope of trailing mean media sentiment"])
        vr = (F.get("vacuum_risk") or {}).get("by_lens", [])
        if vr:
            rows.append(["Silence / vacuum risk (V), top lens", _fmt(vr[0]["V"], "dec").lstrip("+"), vr[0]["lens"],
                         f"owned voice {vr[0]['owned_voice']}, competitor presence {vr[0]['competitor_presence']}"])
        pt = F.get("narrative_pull_through") or {}
        rows.append(["Narrative pull-through (PT)", _fmt(pt.get("PT"), "dec").lstrip("+") if pt.get("PT") is not None else "—",
                     "Measured" if pt.get("PT") is not None else "Not configured", pt.get("note") or f"{pt.get('items_on_frame', 0)} items on frame"])
        rr = F.get("reputation_resilience")
        if rr:
            rows.append(["Reputation resilience (R)", _fmt(rr.get("R"), "dec").lstrip("+"), rr.get("reading", "—"),
                         f"recovery {rr.get('recovery_speed')}, defenders {rr.get('defender_ratio')}, advocacy {rr.get('advocacy_share')}, floor {rr.get('sentiment_floor')}"])
        return {"kind": "table", "title": "Advanced signals", "columns": ["Signal", "Value", "Reading", "Components"], "rows": rows}
    return None


# ======================================================================== standard blueprint
def standard_blueprint(facts) -> dict:
    multi = facts.get("meta", {}).get("multi_entity")
    kpis = [
        {"id": "k_neg", "label": "of coverage is negative", "metric": "overview.neg_share_pct", "format": "pct"},
        {"id": "k_step", "label": "rise in weekly negative volume, baseline to peak", "metric": "weekly.step_change_x", "format": "x"},
        {"id": "k_items", "label": "items analysed for the focus brand", "metric": "overview.items", "format": "num"},
        {"id": "k_signal", "label": "of ingested items passed the noise gate", "metric": "data.signal_rate_pct", "format": "pct"},
    ]
    if multi:
        kpis[2] = {"id": "k_peer", "label": "peer average negative share", "metric": "peers.peer_avg_neg_share_pct", "format": "pct"}
    sections = [
        {"id": "summary", "title": "Executive summary", "kpis": kpis, "visuals": [],
         "ai": {"narrative": True, "items": {"kind": "findings", "count": 5},
                "guidance": "Open with the single most important risk story. Then 5 findings, each a bold one-line "
                            "claim followed by evidence. Mention when the shift first became visible (first alert week)."}},
        {"id": "risks", "title": "Risk landscape", "visuals": [],
         "ai": {"narrative": False, "items": {"kind": "risk_themes", "count": 6},
                "guidance": "Six risk themes ranked by severity (reach, sentiment, proximity to the brand) and momentum "
                            "(direction of weekly volume in the final weeks). Ground each in lenses, spikes and sample posts."}},
        {"id": "trajectory", "title": "Sentiment trajectory and event timeline",
         "visuals": [{"builtin": "weekly_sentiment"}, {"builtin": "spikes_table"}],
         "ai": {"narrative": True, "timeline": True,
                "guidance": "Describe how negative share moved week to week and label each spike day with what drove it, "
                            "based only on its top terms and sample posts."}},
        {"id": "lenses", "title": "Perception lenses", "visuals": [{"builtin": "lens_table"}, {"builtin": "lens_priority"}],
         "ai": {"narrative": True, "guidance": "Which lenses to act on first and why (priority, tone, emotion, momentum)."}},
        {"id": "social", "title": "Social deep dive", "visuals": [{"builtin": "platform_mix"}, {"builtin": "top_authors_negative"}],
         "ai": {"narrative": True, "items": {"kind": "social_readings", "count": 4},
                "guidance": "Who is talking (media vs individuals), where engagement concentrates, whether the story is "
                            "media-led or customer-led."}},
        {"id": "signals", "title": "Advanced signals", "visuals": [{"builtin": "formulas_table"}],
         "ai": {"narrative": True, "guidance": "Interpret coordination, newsroom escalation, journalist trend, vacuum "
                                               "risk and resilience in plain language for a board audience."}},
        {"id": "watch", "title": "What to watch next", "visuals": [],
         "ai": {"narrative": False, "items": {"kind": "watchlist", "count": 5},
                "guidance": "Forward-looking items that follow from the data (rising lenses, unresolved spikes, alert thresholds)."}},
        {"id": "actions", "title": "Recommended actions and monitoring", "visuals": [],
         "ai": {"narrative": False, "items": {"kind": "recommendations", "count": 4},
                "guidance": "Concrete actions: alerts to set, owners, content responses, channels to prioritise."}},
        {"id": "method", "title": "Method and scope", "visuals": [{"builtin": "data_health"}], "ai": {"narrative": False},
         "method": True},
    ]
    if multi:
        sections.insert(3, {"id": "peers", "title": "Competitor context", "visuals": [{"builtin": "peers_table"}, {"builtin": "sov_chart"}],
                            "ai": {"narrative": True, "guidance": "Compare the focus brand's negative share and share of voice with peers."}})
    return {"name": "Reputation risk scan", "description": "Built-in reputation intelligence report", "sections": sections}


# ======================================================================== resolution
def resolve(blueprint, df, facts) -> list:
    cat = metric_catalog(facts)
    q = query_engine.prepare(df)
    out = []
    for sec in blueprint.get("sections", []):
        s = {k: sec.get(k) for k in ("id", "title", "lede", "method")}
        s["ai"] = sec.get("ai") or {"narrative": True}
        s["kpis"] = []
        for k in sec.get("kpis") or []:
            val, avail, note = None, False, k.get("note")
            fmt = k.get("format") or "num"
            if k.get("metric") and k["metric"] in cat:
                val, avail = cat[k["metric"]]["value"], True
                fmt = k.get("format") or cat[k["metric"]]["format"]
            elif k.get("query"):
                try:
                    r = query_engine.run_query(q, k["query"])
                    if r["rows"]:
                        val, avail = r["rows"][0][-1], True
                except Exception as e:  # noqa
                    note = f"Could not compute: {e}"
            s["kpis"].append({"id": k.get("id"), "label": k.get("label"), "value": val, "display": _fmt(val, fmt) if avail else "n/a",
                              "available": avail, "note": note or (None if avail else "Not available in this data"),
                              "source_label": k.get("source_label")})
        s["visuals"] = []
        for v in sec.get("visuals") or []:
            if v.get("builtin"):
                b = builtin(v["builtin"], facts)
                if b:
                    if v.get("title"):
                        b["title"] = v["title"]
                    s["visuals"].append(b)
            elif v.get("query"):
                try:
                    r = query_engine.run_query(q, v["query"])
                    s["visuals"].append(_query_visual(r, v))
                except Exception as e:
                    s["visuals"].append({"kind": "note", "text": f"{v.get('title') or 'Visual'}: could not compute ({e})."})
        out.append(s)
    return out


def _query_visual(r, v):
    render = v.get("render") or "table"
    cols, rows = r["columns"], r["rows"]
    if render == "table" or not rows:
        return {"kind": "table", "title": v.get("title") or "", "columns": cols,
                "rows": [[_fmt(x) if isinstance(x, (int, float)) else x for x in row] for row in rows]}
    metric_idx = [i for i, c in enumerate(cols) if any(isinstance(row[i], (int, float)) for row in rows)]
    label_idx = [i for i in range(len(cols)) if i not in metric_idx] or [0]
    labels = [" · ".join(str(row[i]) for i in label_idx) for row in rows]
    palette = [C_ACC, C_NEG, "#C7A13A", C_POS, "#5B6776"]
    ds = [{"label": cols[i], "data": [row[i] for row in rows], "color": palette[n % 5]} for n, i in enumerate(metric_idx)]
    return {"kind": "chart", "type": {"bar": "bar", "line": "line", "hbar": "hbar", "stacked": "stacked"}.get(render, "bar"),
            "title": v.get("title") or "", "labels": labels, "datasets": ds}


# ======================================================================== AI writing
WRITER_SYSTEM = """You are a senior reputation-intelligence analyst writing a report for a bank's board, communications
and risk leadership. Write in clear, concise business English (Indian/British spelling). Short sentences, no hype,
no filler, no exclamation marks.

Hard rules:
- Use ONLY figures that appear in the DATA (you may round them). Never invent numbers, names, events, dates or quotes.
- Prefer exact counts, percentages, dates and named lenses/platforms/authors from the data.
- If the data cannot support a point, leave it out rather than guessing.
- Spike days: describe what drove them only from their top terms and sample posts; hedge with "appears" when inferring.
- Do not mention AI, language models, prompts, or this instruction set.
- Output one JSON object exactly matching the requested schema."""


def _compact_facts(facts):
    f = json.loads(json.dumps(facts, default=str))
    for k in ("top_posts", "top_negative_by_reach"):
        f[k] = (f.get(k) or [])[:6]
    for s in f.get("spikes", []):
        s["sample_posts"] = s.get("sample_posts", [])[:3]
    return f


def _visual_summary(v):
    if v["kind"] == "table":
        return {"table": v.get("title"), "columns": v["columns"], "rows": v["rows"][:15]}
    if v["kind"] == "chart":
        return {"chart": v.get("title"), "labels": v["labels"][:30],
                "series": {d["label"]: d["data"][:30] for d in v["datasets"]}}
    return {"note": v.get("text")}


def write_narrative(provider, blueprint_name, sections, facts, focus_note=""):
    sec_in, schema = [], {}
    for s in sections:
        ai = s.get("ai") or {}
        if s.get("method"):
            continue
        entry = {"id": s["id"], "title": s["title"], "guidance": ai.get("guidance", ""),
                 "kpis": [{"id": k["id"], "label": k["label"], "value": k["display"]} for k in s["kpis"] if k["available"]],
                 "visual_data": [_visual_summary(v) for v in s["visuals"]]}
        sec_in.append(entry)
        shape = {}
        if ai.get("narrative", True):
            shape["narrative"] = "1-3 short paragraphs; **bold** allowed; may use '- ' bullet lines"
        if s["kpis"]:
            shape["kpi_context"] = {k["id"]: "≤ 16 words of context for this figure" for k in s["kpis"] if k["available"]}
        if ai.get("items"):
            it = ai["items"]
            shape["items"] = (f"array of {it.get('count', 4)} objects {{title, body (2-4 sentences), severity "
                              f"(High|Medium|Low|Structural), momentum (Rising|Sustained|Episodic|New|Falling|Watch), "
                              f"tag (≤3 words)}} — kind: {it.get('kind')}")
        if ai.get("timeline"):
            shape["event_labels"] = "object mapping each spike date (YYYY-MM-DD) to a ≤ 14-word label of what drove it"
        if any(v["kind"] in ("chart", "table") for v in s["visuals"]):
            shape["callout"] = "one sentence 'so what' for the visuals"
        schema[s["id"]] = shape
    prompt = {
        "report": blueprint_name,
        "focus_brand": facts.get("meta", {}).get("primary_entity"),
        "period": [facts.get("meta", {}).get("period_start"), facts.get("meta", {}).get("period_end")],
        "reader_focus": focus_note or "General board-level reputation risk review",
        "sections": sec_in,
        "DATA": _compact_facts(facts),
    }
    out_schema = {"title": "report title (≤ 8 words, include brand)",
                  "subtitle": "one line on scope and peers",
                  "headline": "the single most important takeaway as a 6-12 word headline",
                  "sections": schema}
    text = ("Write the report content.\n\nINPUT:\n" + json.dumps(prompt, default=str)
            + "\n\nReturn JSON with this shape:\n" + json.dumps(out_schema))
    return provider.complete_json(WRITER_SYSTEM, [{"type": "text", "text": text}], max_tokens=10000)


# ======================================================================== number check
NUM_RX = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?\s*(k|K|M|mn|bn)?(?![\w])")


def _source_numbers(obj, acc):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.add(float(obj))
    elif isinstance(obj, str):
        for m in NUM_RX.finditer(obj):
            try:
                acc.add(float(m.group(0).replace(",", "").rstrip("kKMmnb ").strip()))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for v in obj.values():
            _source_numbers(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _source_numbers(v, acc)


def verify_numbers(ai, facts, sections):
    src = set()
    _source_numbers(facts, src)
    for s in sections:
        _source_numbers(s.get("kpis"), src)
        _source_numbers([_visual_summary(v) for v in s.get("visuals", [])], src)
    src_list = sorted(src)
    checked, unverified = 0, []
    texts = []

    def walk(o, path=""):
        if isinstance(o, str):
            texts.append(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                if k == "event_labels" and isinstance(v, dict):
                    texts.extend(v.values())
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(ai)
    for t in texts:
        for m in NUM_RX.finditer(t):
            raw = m.group(0)
            try:
                val = float(raw.replace(",", "").rstrip("kKMmnb ").strip())
            except ValueError:
                continue
            suf = (m.group(1) or "").lower()
            mult = {"k": 1e3, "m": 1e6, "mn": 1e6, "bn": 1e9}.get(suf, 1)
            if "." not in raw and mult == 1 and (abs(val) <= 12 or 1990 <= val <= 2100):
                continue
            if re.match(r"\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", t[m.end():m.end() + 5], re.I):
                continue
            checked += 1
            cands = [val, val * mult, abs(val)]
            ok = any(abs(c - s) <= max(0.051, 0.012 * abs(s)) for c in cands for s in src_list
                     if abs(s) > 0 or c == 0)
            if not ok:
                lo = max(0, m.start() - 50)
                unverified.append({"number": raw.strip(), "context": t[lo:m.end() + 40]})
    return {"checked": checked, "verified": checked - len(unverified), "unverified": unverified[:40]}


# ======================================================================== template learning
BLUEPRINT_SYSTEM = """You are a reporting architect. You study a sample report (its text and page images), identify its
structure, KPIs, charts, tables and writing style, and design an equivalent report for a NEW dataset.
You can only use figures that the system can compute: metric catalog keys, built-in visuals, or a JSON query in the
given query language over the given fields. When a KPI in the sample cannot be computed from the new data, keep it
with "available": false and a short note — never fake it. Output a single JSON object."""


def build_template_blueprint(provider, extracted, facts, df):
    cat = metric_catalog(facts)
    catalog = {k: v["label"] for k, v in cat.items()}
    profile = query_engine.field_profile(query_engine.prepare(df))
    spec = {
        "name": "short name for this report template",
        "description": "one line on what the sample report is and who it is for",
        "style": {"tone": "e.g. board briefing, punchy", "length": "short|medium|long", "notes": "layout/style observations"},
        "sections": [{
            "id": "snake_case id", "title": "section title as in the sample (adapted to the new data)",
            "purpose": "what this section tells the reader",
            "kpis": [{"id": "k1", "source_label": "KPI as named in the sample", "label": "short label for the new report",
                      "metric": "catalog key or null", "query": "query JSON or null (must return one row; last column is the value)",
                      "format": "pct|num|x|dec|text", "available": True, "note": "only if unavailable"}],
            "visuals": [{"builtin": "builtin name"}, {"query": {}, "render": "bar|hbar|line|stacked|table", "title": "..."}],
            "ai": {"narrative": True, "items": {"kind": "e.g. findings|risk_themes|watchlist|recommendations", "count": 4},
                   "timeline": False, "guidance": "what the writer should cover, mirroring the sample's section"},
        }],
    }
    text = (
        "SAMPLE REPORT TEXT:\n" + (extracted.get("text") or "(no text layer — use the images)")[:45000]
        + "\n\nMETRIC CATALOG (key: label):\n" + json.dumps(catalog, ensure_ascii=False)
        + "\n\nBUILT-IN VISUALS (name: description):\n" + json.dumps(BUILTINS)
        + "\n\nQUERY LANGUAGE (JSON schema):\n" + json.dumps(query_engine.QUERY_SCHEMA)
        + "\n\nFIELDS IN THE NEW DATA:\n" + json.dumps(profile, default=str)[:12000]
        + "\n\nNEW DATA SUMMARY:\n" + json.dumps({k: facts.get(k) for k in ("meta", "overview", "data_health")}, default=str)
        + "\n\nDesign the blueprint. Keep the sample's section order and intent. Use 4-10 sections. "
          "Prefer built-in visuals when they match; otherwise write a query. Return JSON in this shape:\n" + json.dumps(spec)
    )
    parts = [{"type": "text", "text": text}] + (extracted.get("images") or [])[:8]
    bp = provider.complete_json(BLUEPRINT_SYSTEM, parts, max_tokens=8000)
    return sanitize_blueprint(bp, cat)


def sanitize_blueprint(bp, cat=None):
    bp = bp if isinstance(bp, dict) else {}
    secs = []
    for i, s in enumerate(bp.get("sections") or []):
        if not isinstance(s, dict):
            continue
        s.setdefault("id", f"s{i + 1}")
        s["id"] = re.sub(r"[^a-z0-9_]", "_", str(s["id"]).lower())[:40] or f"s{i + 1}"
        s.setdefault("title", f"Section {i + 1}")
        kp = []
        for j, k in enumerate(s.get("kpis") or []):
            if not isinstance(k, dict):
                continue
            k.setdefault("id", f"{s['id']}_k{j + 1}")
            if cat is not None and k.get("metric") and k["metric"] not in cat and not k.get("query"):
                k["available"] = False
                k["note"] = k.get("note") or "Metric not available in this data"
            if not isinstance(k.get("query"), dict):
                k["query"] = None
            kp.append(k)
        s["kpis"] = kp
        s["visuals"] = [v for v in (s.get("visuals") or []) if isinstance(v, dict) and
                        ((v.get("builtin") in BUILTINS) or isinstance(v.get("query"), dict))]
        s["ai"] = s.get("ai") if isinstance(s.get("ai"), dict) else {"narrative": True}
        if s["ai"].get("items") and not isinstance(s["ai"]["items"], dict):
            s["ai"]["items"] = None
        secs.append(s)
    bp["sections"] = secs
    bp.setdefault("name", "Custom report")
    return bp


# ======================================================================== HTML rendering
def _md(text):
    if not text:
        return ""
    t = html.escape(str(text))
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    blocks, out = re.split(r"\n\s*\n", t.strip()), []
    for b in blocks:
        lines = [l for l in b.split("\n") if l.strip()]
        if lines and all(re.match(r"\s*[-•]\s+", l) for l in lines):
            out.append("<ul>" + "".join(f"<li>{re.sub(r'^\s*[-•]\s+', '', l)}</li>" for l in lines) + "</ul>")
        else:
            out.append("<p>" + "<br>".join(lines) + "</p>")
    return "".join(out)


SEV_CLASS = {"high": "sev-high", "medium": "sev-med", "low": "sev-low", "structural": "sev-struct"}


def _visual_html(v, idx):
    if v["kind"] == "table":
        tone_col = v.get("tone_col")
        hl = set(v.get("highlight_first_col") or [])
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in v["columns"])
        body = []
        for r in v["rows"]:
            cells = []
            for i, c in enumerate(r):
                cls = ""
                if tone_col is not None and i == tone_col:
                    try:
                        fv = float(str(c).replace("+", ""))
                        cls = ' class="neg"' if fv <= -0.15 else (' class="pos"' if fv >= 0.15 else "")
                    except ValueError:
                        pass
                cells.append(f"<td{cls}>{html.escape(str(c))}</td>")
            rc = ' class="hl"' if r and r[0] in hl else ""
            body.append(f"<tr{rc}>{''.join(cells)}</tr>")
        title = f"<figcaption>{html.escape(v.get('title') or '')}</figcaption>" if v.get("title") else ""
        return f'<figure class="tbl">{title}<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div></figure>'
    if v["kind"] == "chart":
        cfg = html.escape(json.dumps(v, default=str), quote=True)
        h = max(220, 34 * len(v["labels"]) + 60) if v["type"] == "hbar" else 280
        note = f"<p class='fignote'>{html.escape(v['note'])}</p>" if v.get("note") else ""
        return (f'<figure class="chart"><figcaption>{html.escape(v.get("title") or "")}</figcaption>'
                f'<div class="cbox" style="height:{h}px"><canvas data-chart="{cfg}"></canvas></div>{note}</figure>')
    return f'<p class="fignote">{html.escape(v.get("text", ""))}</p>'


def method_text(facts, quality):
    m, dh = facts.get("meta", {}), facts.get("data_health", {})
    src = ", ".join(f"{k or 'unknown'}: {v:,}" for k, v in (m.get("sentiment_sources") or {}).items())
    return (
        f"All figures are computed directly from the uploaded data for {m.get('period_start')} to {m.get('period_end')} "
        f"({m.get('days')} days). Focus brand: {m.get('primary_entity')}. "
        f"A relevance and noise gate removed duplicates, spam, bot-like reposts, pure noise and contentless replies; "
        f"{dh.get('actionable_items', 0):,} of {dh.get('total_ingested', 0):,} items ({dh.get('signal_rate_pct')}%) were kept as "
        f"actionable using Actionability = 0.30·relevance + 0.25·opinion + 0.15·specificity + 0.15·(1−spam) + 0.15·(1−bot), "
        f"θ = {dh.get('theta')}. Volume, share of voice and sentiment trajectory use the de-duplicated coverage set; "
        f"perception lenses use the actionable set. Sentiment sources — {src}. "
        f"Net sentiment NSS = (Pos − Neg) ÷ (Pos + Neu + Neg); NSSw weights each item by reach. "
        f"Momentum uses velocity z = (last full week − baseline mean) ÷ baseline std. "
        f"Priority = 0.30·%Neg + 0.20·normVolume + 0.20·σ(velocity z) + 0.15·reach share + 0.15·risk-flag share. "
        f"Weeks start on Monday; a partial final week is excluded from weekly charts. "
        f"Narrative figures checked against source data: {quality.get('verified', 0)} of {quality.get('checked', 0)} traced."
    )


def render_html(spec) -> str:
    sections_html = []
    n = 0
    for s in spec["sections"]:
        ai = dict(s.get("ai_out") or {})
        spec_ai = s.get("ai") or {}
        if not spec_ai.get("items"):
            ai.pop("items", None)
        if spec_ai.get("narrative") is False:
            ai.pop("narrative", None)
        n += 1
        parts = [f'<section class="sec" id="{html.escape(s["id"])}"><header class="sec-h"><span class="sec-n">{n:02d}</span>'
                 f'<h2>{html.escape(s["title"])}</h2></header>']
        if s.get("kpis"):
            cards = []
            for k in s["kpis"]:
                ctx = (ai.get("kpi_context") or {}).get(k["id"], "") if k["available"] else (k.get("note") or "")
                cls = "kpi" + ("" if k["available"] else " na")
                cards.append(f'<div class="{cls}"><div class="kv">{html.escape(k["display"])}</div>'
                             f'<div class="kl">{html.escape(k.get("label") or "")}</div>'
                             f'<div class="kc">{html.escape(ctx)}</div></div>')
            parts.append(f'<div class="kpis">{"".join(cards)}</div>')
        if ai.get("narrative"):
            parts.append(f'<div class="prose">{_md(ai["narrative"])}</div>')
        if ai.get("items"):
            cards = []
            for it in ai["items"]:
                if not isinstance(it, dict):
                    continue
                sev = str(it.get("severity") or "").strip()
                mom = str(it.get("momentum") or "").strip()
                badge = f'<span class="badge {SEV_CLASS.get(sev.lower(), "")}">{html.escape(sev)}{" · " + html.escape(mom) if mom else ""}</span>' if sev or mom else ""
                cards.append(f'<article class="item">{badge}<h3>{html.escape(str(it.get("title", "")))}</h3>'
                             f'{_md(it.get("body", ""))}</article>')
            parts.append(f'<div class="items">{"".join(cards)}</div>')
        for i, v in enumerate(s.get("visuals", [])):
            if s.get("timeline") and v.get("title", "").startswith("Spike"):
                continue
            parts.append(_visual_html(v, i))
        if s.get("timeline"):
            labels = ai.get("event_labels") or {}
            rows = []
            avg = spec.get("overall_neg") or 0
            for e in s["timeline"]:
                risk = (e.get("neg_share_pct") or 0) > max(avg * 1.5, 15)
                rows.append(f'<li class="{"risk" if risk else ""}"><time>{html.escape(e["date"])}</time>'
                            f'<div><strong>{html.escape(labels.get(e["date"], ", ".join(e["top_terms"][:4])))}</strong>'
                            f'<span>{e["items"]:,} items · {e["peak_to_mean"]}× daily mean · negative {e.get("neg_share_pct")}%</span></div></li>')
            if rows:
                parts.append(f'<figure class="tl"><figcaption>Significant events (risk-flagged in red)</figcaption><ol>{"".join(rows)}</ol></figure>')
        if ai.get("callout"):
            parts.append(f'<p class="callout">{html.escape(ai["callout"])}</p>')
        if s.get("method"):
            parts.append(f'<div class="prose small"><p>{html.escape(spec["method_text"])}</p></div>')
        parts.append("</section>")
        sections_html.append("".join(parts))

    return TEMPLATE.replace("{{TITLE}}", html.escape(spec["title"])) \
        .replace("{{SUBTITLE}}", html.escape(spec.get("subtitle") or "")) \
        .replace("{{HEADLINE}}", html.escape(spec.get("headline") or "")) \
        .replace("{{PERIOD}}", html.escape(spec.get("period") or "")) \
        .replace("{{ORG}}", html.escape(spec.get("org") or "Reputation Intelligence")) \
        .replace("{{GENERATED}}", html.escape(spec.get("generated") or "")) \
        .replace("{{SECTIONS}}", "".join(sections_html))


def build_spec(blueprint, sections, ai, facts, org, kind):
    m = facts.get("meta", {})
    ai_secs = (ai or {}).get("sections") or {}
    for s in sections:
        s["ai_out"] = ai_secs.get(s["id"]) or {}
        if (s.get("ai") or {}).get("timeline"):
            s["timeline"] = facts.get("spikes", [])
    period = ""
    if m.get("period_start"):
        a, b = pd.Timestamp(m["period_start"]), pd.Timestamp(m["period_end"])
        period = f"{a.strftime('%d %b')} – {b.strftime('%d %b %Y')} ({m.get('days')} days)"
    return {
        "title": (ai or {}).get("title") or f"{m.get('primary_entity')} · {blueprint.get('name')}",
        "subtitle": (ai or {}).get("subtitle") or "",
        "headline": (ai or {}).get("headline") or "",
        "period": period, "org": org, "kind": kind,
        "generated": datetime.utcnow().strftime("%d %b %Y"),
        "overall_neg": facts.get("overview", {}).get("neg_share_pct"),
        "sections": sections,
    }


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--ink:#16202B;--muted:#5B6776;--rule:#D8DEE6;--bg:#EEF1F4;--paper:#fff;--acc:#0F5C63;--neg:#B42318;--amb:#B54708;--pos:#1E7B4E;
--serif:"Source Serif 4",Georgia,serif;--sans:"Public Sans",system-ui,-apple-system,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 var(--sans)}
.page{max-width:980px;margin:32px auto;background:var(--paper);padding:56px 64px;box-shadow:0 1px 0 var(--rule)}
.mast{border-bottom:3px solid var(--ink);padding-bottom:28px;margin-bottom:8px}
.org{font-weight:600;color:var(--acc);font-size:13px;letter-spacing:.02em}
h1{font-family:var(--serif);font-size:44px;line-height:1.08;margin:10px 0 8px;font-weight:700}
.sub{color:var(--muted);font-size:16px;margin:0 0 6px;max-width:70ch}
.period{font-size:13px;color:var(--muted)}
.headline{font-family:var(--serif);font-size:27px;line-height:1.25;margin:26px 0 0;max-width:38ch;font-weight:600}
.mast + .sec{border-top:0;margin-top:0}
.sec{padding:34px 0 10px;border-top:1px solid var(--rule);margin-top:26px;break-inside:auto}
.sec-h{display:flex;align-items:baseline;gap:14px;margin-bottom:14px}
.sec-n{font-family:var(--serif);color:var(--acc);font-size:22px;font-weight:600}
h2{font-family:var(--serif);font-size:28px;margin:0;line-height:1.2}
.prose{font-family:var(--serif);font-size:17px;line-height:1.65;max-width:72ch}
.prose p{margin:0 0 12px}.prose ul{padding-left:20px;margin:0 0 12px}.prose li{margin:4px 0}
.prose.small{font-family:var(--sans);font-size:13px;color:var(--muted);max-width:none}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:0;margin:6px 0 22px;border-top:1px solid var(--ink)}
.kpi{padding:16px 18px 14px 0;border-right:1px solid var(--rule);margin-right:18px}.kpi:last-child{border-right:0}
.kv{font-family:var(--serif);font-size:40px;font-weight:700;line-height:1;color:var(--neg)}
.kpi:nth-child(n+3) .kv{color:var(--ink)}
.kl{font-weight:600;font-size:13px;margin-top:8px}.kc{font-size:13px;color:var(--muted);margin-top:4px}
.kpi.na .kv{color:var(--rule)}
.items{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:14px;margin:6px 0 18px}
.item{border:1px solid var(--rule);border-left:4px solid var(--ink);padding:14px 16px;background:#FBFCFD}
.item h3{font-family:var(--serif);font-size:18px;margin:6px 0 6px;line-height:1.3}.item p{margin:0 0 8px;font-size:14px}
.badge{font-size:11.5px;font-weight:700;color:var(--muted)}
.item:has(.sev-high){border-left-color:var(--neg)}.item:has(.sev-med){border-left-color:var(--amb)}
.item:has(.sev-low){border-left-color:var(--pos)}.sev-high{color:var(--neg)}.sev-med{color:var(--amb)}.sev-low{color:var(--pos)}
figure{margin:18px 0 22px}figcaption{font-weight:600;font-size:13px;margin-bottom:8px}
.cbox{position:relative;width:100%}.fignote{font-size:12px;color:var(--muted);margin:6px 0}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;font-weight:600;border-bottom:2px solid var(--ink);padding:7px 10px 7px 0;white-space:nowrap}
td{border-bottom:1px solid var(--rule);padding:7px 10px 7px 0;vertical-align:top;font-variant-numeric:tabular-nums}
tr.hl td{background:#FDF0EE;font-weight:600}td.neg{color:var(--neg);font-weight:700}td.pos{color:var(--pos);font-weight:700}
.tl ol{list-style:none;padding:0;margin:0;border-left:2px solid var(--rule)}
.tl li{display:grid;grid-template-columns:110px 1fr;gap:12px;padding:8px 0 8px 14px;position:relative}
.tl li:before{content:"";position:absolute;left:-6px;top:15px;width:10px;height:10px;border-radius:50%;background:var(--muted)}
.tl li.risk:before{background:var(--neg)}.tl li.risk strong{color:var(--neg)}
.tl time{font-size:13px;color:var(--muted);font-variant-numeric:tabular-nums}.tl span{display:block;font-size:12.5px;color:var(--muted)}
.callout{border-left:3px solid var(--acc);padding:6px 0 6px 14px;font-weight:500;max-width:72ch}
.foot{margin-top:40px;padding-top:14px;border-top:1px solid var(--rule);font-size:12px;color:var(--muted);display:flex;justify-content:space-between}
@media (max-width:760px){.page{padding:28px 20px;margin:0}h1{font-size:32px}.items{grid-template-columns:1fr}.kpi{border-right:0}}
@media print{body{background:#fff}.page{box-shadow:none;margin:0;max-width:none;padding:0 8mm}.sec{break-inside:avoid-page}
figure,.item,.kpis{break-inside:avoid}@page{size:A4;margin:14mm}}
</style></head><body><main class="page">
<header class="mast"><div class="org">{{ORG}}</div><h1>{{TITLE}}</h1><p class="sub">{{SUBTITLE}}</p>
<div class="period">Period analysed: {{PERIOD}} · Prepared {{GENERATED}}</div><p class="headline">{{HEADLINE}}</p></header>
{{SECTIONS}}
<footer class="foot"><span>{{ORG}} · Private and confidential</span><span>{{TITLE}}</span></footer>
</main>
<script>
(function(){if(!window.Chart)return;Chart.defaults.font.family='"Public Sans",system-ui,sans-serif';Chart.defaults.color='#5B6776';
document.querySelectorAll('canvas[data-chart]').forEach(function(cv){var c=JSON.parse(cv.dataset.chart);var hb=c.type==='hbar',st=c.type==='stacked';
var hasY2=c.datasets.some(function(d){return d.axis==='y2'});
var ds=c.datasets.map(function(d){return{type:d.line||c.type==='line'?'line':'bar',label:d.label,data:d.data,backgroundColor:d.color,borderColor:d.color,
borderWidth:d.line||c.type==='line'?2:0,pointRadius:d.line?3:2,yAxisID:d.axis==='y2'?'y2':(hb?'x':'y'),tension:.25,order:d.line?0:1,maxBarThickness:28}});
var sc={x:{stacked:st,grid:{display:hb}},y:{stacked:st,beginAtZero:true,grid:{display:!hb,color:'#EEF1F4'}}};
if(hasY2)sc.y2={position:'right',beginAtZero:true,grid:{display:false},ticks:{callback:function(v){return v+'%'}}};
if(hb){sc={x:{beginAtZero:true,grid:{color:'#EEF1F4'}},y:{grid:{display:false}}};ds.forEach(function(d){delete d.yAxisID})}
new Chart(cv,{type:'bar',data:{labels:c.labels,datasets:ds},options:{indexAxis:hb?'y':'x',responsive:true,maintainAspectRatio:false,
animation:false,plugins:{legend:{position:'bottom',labels:{boxWidth:10,boxHeight:10}}},scales:sc}})});})();
</script></body></html>"""
