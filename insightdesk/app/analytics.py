"""Deterministic analytics. Every number in a report comes from here, never from the language model.

Implements the Perception Lens methodology (Steps 3–7) and the additional formulas:
Narrative Pull-Through, Newsroom Escalation, Journalist Warming/Cooling, Organic vs Coordinated,
Silence/Vacuum Risk and Reputation Resilience.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from .taxonomy import as_text, lens_list

MEDIA_RX = re.compile(
    r"(?:news|times|tv\b|tv18|mint|money|business|today|express|herald|post\b|daily|live\b|media|press|journal|"
    r"cnbc|ndtv|reuters|bloomberg|economic|hindu|standard|businessline|awaaz|zee|aaj ?tak|india ?today|"
    r"moneycontrol|livemint|financial|forbes|fortune|wire|gazette|tribune|chronicle|bbc|cnn|wsj|ft\b|"
    r"bulletin|report|insider|outlook|firstpost|scroll|quint|print\b|mirror|telegraph|dna\b|republic)", re.I)
STOP = set("""a an the and or but if of to in on at for from by with without about as is are was were be been being
this that these those it its it's i me my we our you your they their them he she his her not no yes do does did
done have has had will would can could should may might must shall just also very more most so than then there here
what which who whom whose when where why how all any both each few other some such only own same too s t can't won't
don't isn't aren't wasn't amp rt via http https www com co in into out up down over under again further once new
like get got one two us per vs said says say ki ka ke hai se ko bhi nahi kya""".split())

WEEKLY_ALERT_NEG_SHARE = 15.0


def _r(x, d=1):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return None
        return round(float(x), d)
    except (TypeError, ValueError):
        return None


def _pct(a, b, d=1):
    return _r(100.0 * a / b, d) if b else None


def sentiment_stats(s: pd.Series, reach: pd.Series | None = None) -> dict:
    scored = s.dropna()
    n = len(scored)
    pos, neg = int((scored == 1).sum()), int((scored == -1).sum())
    neu = n - pos - neg
    out = {"scored": n, "pos": pos, "neu": neu, "neg": neg,
           "pos_pct": _pct(pos, n), "neu_pct": _pct(neu, n), "neg_pct": _pct(neg, n),
           "nss": _r((pos - neg) / n, 2) if n else None}
    if reach is not None and n:
        w = reach.loc[scored.index].clip(lower=0) + 1.0  # +1 so zero-reach items still count
        out["nssw"] = _r(float((w * scored).sum() / w.sum()), 2)
    return out


def tone_of(nssw):
    if nssw is None:
        return "Unscored"
    return "Positive" if nssw >= 0.15 else ("Negative" if nssw <= -0.15 else "Mixed")


def _sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x is not None else 0.5


def _gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x >= 0]
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def author_types(df: pd.DataFrame, owned_tokens: list) -> pd.Series:
    names = as_text(df["author"]).str.lower() + " " + as_text(df["handle"]).str.lower()
    owned = pd.Series(False, index=df.index)
    for t in owned_tokens:
        owned |= names.str.contains(re.escape(t), na=False)
    media = names.str.contains(MEDIA_RX.pattern, case=False, regex=True, na=False) | df["channel"].eq("News & media")
    return pd.Series(np.where(owned, "Brand-owned", np.where(media, "Media", "Individual")), index=df.index)


def owned_tokens_for(df, cfg):
    toks = [t.strip().lower().lstrip("@") for t in (cfg.get("owned_handles") or []) if t.strip()]
    return toks


def top_terms(texts: pd.Series, exclude: set, n=15, bigrams=True):
    c1, c2 = Counter(), Counter()
    for t in texts.head(60000):
        toks = [w for w in re.findall(r"[a-z\u0900-\u097f][a-z0-9\u0900-\u097f']+", str(t).lower())
                if w not in STOP and w not in exclude and len(w) > 2]
        c1.update(toks)
        if bigrams:
            c2.update(" ".join(p) for p in zip(toks, toks[1:]))
    return {"terms": [[w, c] for w, c in c1.most_common(n)],
            "bigrams": [[w, c] for w, c in c2.most_common(n) if c > 1]}


def _week_start(d: pd.Series) -> pd.Series:
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.normalize()


# ======================================================================== main entry
def build_facts(df: pd.DataFrame, cfg: dict, gate: dict) -> dict:
    lenses_cfg = lens_list(cfg.get("lenses"))
    df = df.copy()
    has_dates = df["date"].notna().any()
    if has_dates:
        df["week"] = _week_start(df["date"])
        df["day"] = df["date"].dt.normalize()

    cov_all = df[df["in_coverage"]]
    entities = cov_all["entity"].value_counts()
    primary = cfg.get("primary_entity") or (entities.index[0] if len(entities) else "Unspecified")
    multi = len([e for e in entities.index if e != "Unspecified"]) >= 2
    focus = cov_all[cov_all["entity"] == primary] if multi else cov_all
    act = focus[focus["actionable"]]

    owned = owned_tokens_for(df, cfg)
    if not owned and primary and primary != "Unspecified":
        owned = [primary.lower().replace(" ", "")]
    df["author_type"] = author_types(df, owned)
    focus = focus.assign(author_type=df.loc[focus.index, "author_type"])
    act = act.assign(author_type=df.loc[act.index, "author_type"])

    brand_words = set(re.findall(r"[a-z]+", " ".join([str(primary)] + list(cfg.get("brand_terms") or [])).lower()))
    p = re.sub(r"[^a-z0-9]", "", str(primary).lower())
    brand_words |= {"bank", "banks", "banking", "ltd", "limited", p, p + "bank", p + "_bank", p + "bank_cares"}

    facts = {
        "meta": {
            "primary_entity": primary,
            "entities": [{"entity": e, "items": int(v)} for e, v in entities.head(12).items()],
            "multi_entity": multi,
            "period_start": str(df["date"].min().date()) if has_dates else None,
            "period_end": str(df["date"].max().date()) if has_dates else None,
            "days": int((df["date"].max() - df["date"].min()).days) + 1 if has_dates else None,
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "sentiment_sources": df.loc[df["in_coverage"], "sentiment_source"].value_counts().to_dict(),
            "ai_classified_items": int((df.get("sentiment_source", pd.Series(dtype=object)) == "ai").sum()),
        },
        "data_health": gate,
    }

    # ---------------------------------------------------------------- overview
    st = sentiment_stats(focus["sentiment"], focus["reach"])
    social = focus[focus["channel"] == "Social"]
    facts["overview"] = {
        "items": int(len(focus)), "actionable_items": int(len(act)),
        "negative_items": st["neg"], "positive_items": st["pos"], "neutral_items": st["neu"],
        "neg_share_pct": st["neg_pct"], "pos_share_pct": st["pos_pct"], "neu_share_pct": st["neu_pct"],
        "nss": st["nss"], "nssw": st.get("nssw"), "tone": tone_of(st.get("nssw")),
        "social_posts": int(len(social)), "social_engagement": int(social["engagement"].sum()),
        "eng_per_post": _r(social["engagement"].sum() / len(social), 1) if len(social) else None,
        "total_reach": int(focus["reach"].sum()), "unique_authors": int(focus["author"].nunique()),
        "media_share_pct": _pct((focus["author_type"] == "Media").sum(), len(focus)),
        "news_media_items": int((focus["channel"] == "News & media").sum()),
    }

    facts["channels"] = [{"channel": k, "items": int(v), "share_pct": _pct(v, len(focus))}
                         for k, v in focus["channel"].value_counts().items()]
    facts["author_mix"] = [{"author_type": k, "items": int(v), "share_pct": _pct(v, len(focus))}
                           for k, v in focus["author_type"].value_counts().items()]

    # ---------------------------------------------------------------- time series
    weekly, wsum, monthly = [], {}, []
    if has_dates and len(focus):
        last_day = focus["day"].max()
        g = focus.groupby("week")
        for wk, sub in g:
            s = sentiment_stats(sub["sentiment"])
            partial = (wk + pd.Timedelta(days=6)) > last_day
            weekly.append({"week": str(wk.date()), "items": int(len(sub)), "neg": s["neg"], "pos": s["pos"],
                           "neu": s["neu"], "neg_share_pct": s["neg_pct"], "nss": s["nss"],
                           "engagement": int(sub["engagement"].sum()), "reach": int(sub["reach"].sum()),
                           "social_posts": int((sub["channel"] == "Social").sum()), "partial": bool(partial)})
        full = [w for w in weekly if not w["partial"]] or weekly
        if full:
            base_n = max(1, min(4, len(full) // 3 or 1))
            base_neg = float(np.mean([w["neg"] for w in full[:base_n]]))
            peak = max(full, key=lambda w: w["neg"])
            peak_share = max(full, key=lambda w: (w["neg_share_pct"] or 0))
            alerts = [w["week"] for w in full if (w["neg_share_pct"] or 0) > WEEKLY_ALERT_NEG_SHARE]
            wsum = {
                "baseline_weeks": base_n, "baseline_neg_per_week": _r(base_neg, 0),
                "peak_neg_week": peak["week"], "peak_neg_volume": peak["neg"],
                "step_change_x": _r(peak["neg"] / base_neg, 1) if base_neg else None,
                "peak_neg_share_week": peak_share["week"], "peak_neg_share_pct": peak_share["neg_share_pct"],
                "first_alert_week": alerts[0] if alerts else None, "alert_weeks": alerts,
                "alert_threshold_pct": WEEKLY_ALERT_NEG_SHARE,
                "partial_week_excluded": any(w["partial"] for w in weekly),
                "last4_neg_share_pct": _r(np.mean([w["neg_share_pct"] or 0 for w in full[-4:]]), 1),
                "first4_neg_share_pct": _r(np.mean([w["neg_share_pct"] or 0 for w in full[:4]]), 1),
            }
        mg = focus.groupby(focus["date"].dt.to_period("M"))
        for m, sub in mg:
            s = sentiment_stats(sub["sentiment"])
            monthly.append({"month": str(m), "items": int(len(sub)), "neg_share_pct": s["neg_pct"], "nss": s["nss"]})
    facts["weekly"], facts["weekly_summary"], facts["monthly"] = weekly, wsum, monthly

    # ---------------------------------------------------------------- platforms
    plat = []
    tot_posts, tot_eng = max(len(focus), 1), max(focus["engagement"].sum(), 1)
    for p, sub in focus.groupby("platform"):
        s = sentiment_stats(sub["sentiment"])
        plat.append({"platform": p, "items": int(len(sub)), "items_share_pct": _pct(len(sub), tot_posts),
                     "engagement": int(sub["engagement"].sum()),
                     "engagement_share_pct": _pct(sub["engagement"].sum(), tot_eng),
                     "eng_per_item": _r(sub["engagement"].sum() / len(sub), 1), "neg_share_pct": s["neg_pct"]})
    facts["platforms"] = sorted(plat, key=lambda x: -x["items"])

    # ---------------------------------------------------------------- peers / share of voice
    peers = []
    if multi:
        total = len(cov_all)
        for e, sub in cov_all.groupby("entity"):
            if e == "Unspecified":
                continue
            s = sentiment_stats(sub["sentiment"])
            soc = sub[sub["channel"] == "Social"]
            peers.append({"entity": e, "items": int(len(sub)), "share_of_voice_pct": _pct(len(sub), total),
                          "negative_items": s["neg"], "neg_share_pct": s["neg_pct"], "nss": s["nss"],
                          "social_posts": int(len(soc)), "social_engagement": int(soc["engagement"].sum()),
                          "eng_per_post": _r(soc["engagement"].sum() / len(soc), 0) if len(soc) else None,
                          "is_primary": e == primary})
        peers.sort(key=lambda x: -x["items"])
        others = [p for p in peers if not p["is_primary"]]
        if others:
            facts["peer_summary"] = {
                "peer_avg_neg_share_pct": _r(np.mean([p["neg_share_pct"] or 0 for p in others]), 1),
                "peers_negative_items_combined": int(sum(p["negative_items"] for p in others)),
                "primary_negative_items": next((p["negative_items"] for p in peers if p["is_primary"]), None),
            }
    facts["peers"] = peers

    # ---------------------------------------------------------------- lenses (Steps 3–7)
    facts["lenses"], facts["emergent_candidates"] = lens_table(act, focus, lenses_cfg, has_dates, brand_words, cfg)

    # ---------------------------------------------------------------- emotions & flags
    facts["emotions"] = [{"emotion": k, "items": int(v), "share_pct": _pct(v, len(act))}
                         for k, v in act["emotion"].value_counts().items()]
    fl = Counter(f for L in act["flags"] for f in L)
    facts["flags"] = {k: {"items": int(v), "share_pct": _pct(v, len(act))} for k, v in fl.items()}

    # ---------------------------------------------------------------- spikes / events
    facts["spikes"] = spikes(focus, brand_words) if has_dates else []

    # ---------------------------------------------------------------- terms, authors, posts, geo
    neg_set = act[act["sentiment"] == -1]
    facts["top_terms"] = {"all": top_terms(act["text"], brand_words), "negative": top_terms(neg_set["text"], brand_words)}
    facts["top_authors"] = {"all": _authors(focus), "negative": _authors(focus[focus["sentiment"] == -1])}
    facts["top_posts"] = _posts(focus.sort_values("engagement", ascending=False).head(10))
    facts["top_negative_by_reach"] = _posts(focus[focus["sentiment"] == -1].sort_values("reach", ascending=False).head(8))
    facts["countries"] = [{"country": k, "items": int(v), "share_pct": _pct(v, len(focus))}
                          for k, v in focus["country"].value_counts().head(10).items()]

    # ---------------------------------------------------------------- advanced formulas
    facts["formulas"] = advanced_formulas(focus, act, cov_all, primary, multi, cfg, has_dates, facts)
    return facts


def _authors(sub: pd.DataFrame, n=10):
    if not len(sub):
        return []
    g = sub.groupby("author").agg(items=("row_id", "size"), engagement=("engagement", "sum"),
                                  reach=("reach", "max"), author_type=("author_type", "first"),
                                  platform=("platform", lambda s: s.mode().iat[0] if len(s) else ""))
    g = g.sort_values(["items", "engagement"], ascending=False).head(n)
    return [{"author": a, "items": int(r["items"]), "engagement": int(r["engagement"]),
             "author_type": r["author_type"], "platform": r["platform"]} for a, r in g.iterrows()]


def _posts(sub: pd.DataFrame):
    out = []
    for _, r in sub.iterrows():
        out.append({"date": str(r["date"].date()) if pd.notna(r["date"]) else None, "platform": r["platform"],
                    "author": str(r["author"]), "engagement": int(r["engagement"]), "reach": int(r["reach"]),
                    "sentiment": {1: "Positive", 0: "Neutral", -1: "Negative"}.get(r["sentiment"], "Unscored"),
                    "text": str(r["text"])[:280], "url": None if pd.isna(r["url"]) else str(r["url"])})
    return out


def lens_table(act, focus, lenses_cfg, has_dates, brand_words, cfg):
    total = max(len(act), 1)
    total_reach = max(float(act["reach"].sum()), 1.0)
    ex = act[["row_id", "lenses", "sentiment", "reach", "engagement", "emotion", "flags", "author_type"]
             + (["week"] if has_dates else [])].explode("lenses")
    ex = ex[ex["lenses"].notna()]
    w_eng = float(cfg.get("wvolume_engagement_weight", 0.01))
    rows = []
    weeks = sorted(act["week"].dropna().unique()) if has_dates else []
    last_full = weeks[-2] if len(weeks) >= 3 else (weeks[-1] if weeks else None)
    action_map = {L["name"]: L.get("action", "monitor") for L in lenses_cfg}
    for L in lenses_cfg:
        name = L["name"]
        sub = ex[ex["lenses"] == name]
        vol = len(sub)
        if vol == 0:
            rows.append({"lens": name, "volume": 0, "share_pct": 0.0})
            continue
        st = sentiment_stats(sub["sentiment"].reset_index(drop=True), sub["reach"].reset_index(drop=True))
        wvol = float((1 + np.log1p(sub["reach"].clip(lower=0)) + w_eng * sub["engagement"]).sum())
        vz, ratio = None, None
        if has_dates and last_full is not None:
            wc = sub.groupby("week").size().reindex(weeks, fill_value=0)
            # baseline = weeks before the last full week (partial final week ignored)
            idx = list(weeks).index(last_full)
            base = wc.iloc[:idx]
            now = float(wc.loc[last_full])
            if len(base) >= 2:
                mu, sd = float(base.mean()), float(base.std(ddof=0)) or 1.0
                vz = _r((now - mu) / sd, 2)
                ratio = _r(now / mu, 1) if mu else None
        emo = sub["emotion"].value_counts()
        emo = emo.drop("neutral-info", errors="ignore") if len(emo.drop("neutral-info", errors="ignore")) else emo
        dom = emo.index[0] if len(emo) else "neutral-info"
        risk_share = sum("risk" in f for f in sub["flags"]) / vol
        momentum = "Rising" if (vz or 0) >= 1 else ("Falling" if (vz or 0) <= -1 else "Steady")
        rows.append({
            "lens": name, "volume": vol, "share_pct": _pct(vol, total), "weighted_volume": _r(wvol, 0),
            "pos": st["pos"], "neu": st["neu"], "neg": st["neg"],
            "pos_pct": st["pos_pct"], "neu_pct": st["neu_pct"], "neg_pct": st["neg_pct"],
            "nss": st["nss"], "nssw": st.get("nssw"), "tone": tone_of(st.get("nssw")),
            "dominant_emotion": dom, "velocity_z": vz, "momentum": momentum, "last_week_vs_baseline_x": ratio,
            "reach_share_pct": _pct(sub["reach"].sum(), total_reach), "risk_flag_share_pct": _r(100 * risk_share, 1),
            "media_share_pct": _pct((sub["author_type"] == "Media").sum(), vol),
        })
    vols = [r["volume"] for r in rows if r["volume"]]
    vmax = max(vols) if vols else 1
    for r in rows:
        if not r["volume"]:
            r.update({"priority": 0, "denotes": f"{r['lens']}: no mentions in this period"})
            continue
        pr = (0.30 * (r["neg_pct"] or 0) / 100 + 0.20 * r["volume"] / vmax + 0.20 * _sigmoid(r["velocity_z"] or 0)
              + 0.15 * (r["reach_share_pct"] or 0) / 100 + 0.15 * (r["risk_flag_share_pct"] or 0) / 100)
        r["priority"] = _r(pr, 3)
        action = action_map.get(r["lens"], "monitor")
        if r["tone"] == "Negative" and r["momentum"] == "Rising":
            action = f"act this week — {action}"
        if (r["risk_flag_share_pct"] or 0) >= 20:
            action = "early compliance-risk signal"
        mom = r["momentum"].lower()
        if r["last_week_vs_baseline_x"] and r["momentum"] == "Rising":
            mom = f"rising {r['last_week_vs_baseline_x']}×"
        r["denotes"] = f"{r['lens']} — {r['tone']}, {r['dominant_emotion']}-led, {mom} — {action}"
    rows.sort(key=lambda r: -(r.get("priority") or 0))
    for i, r in enumerate(rows, 1):
        r["priority_rank"] = i if r["volume"] else None

    # emergent candidates: frequent bigrams among actionable items that match no lens
    un = act[act["lenses"].map(len) == 0]
    v_min = int(cfg.get("emergent_min_volume", max(15, int(0.01 * len(act)))))
    cands = []
    if len(un):
        tt = top_terms(un["text"], brand_words, n=12)
        for phrase, c in tt["bigrams"][:8]:
            if c >= v_min:
                cands.append({"phrase": phrase, "volume": int(c)})
    ai_topics = act["emergent_topic"].replace("", np.nan).dropna().value_counts()
    for t, c in ai_topics.head(6).items():
        cands.append({"phrase": t, "volume": int(c), "source": "ai"})
    return rows, {"unassigned_items": int(len(un)), "unassigned_share_pct": _pct(len(un), len(act)),
                  "min_volume": v_min, "candidates": cands}


def spikes(focus: pd.DataFrame, brand_words, top=10):
    daily = focus.groupby("day").size()
    if len(daily) < 5:
        return []
    mean = daily.mean()
    out = []
    for day, cnt in daily.sort_values(ascending=False).head(top).items():
        ratio = cnt / mean if mean else 0
        if ratio < 1.5:
            continue
        sub = focus[focus["day"] == day]
        s = sentiment_stats(sub["sentiment"])
        tt = top_terms(sub["text"], brand_words, n=6)
        samples = sub.sort_values(["reach", "engagement"], ascending=False).drop_duplicates("norm_text").head(4)
        out.append({"date": str(day.date()), "items": int(cnt), "peak_to_mean": _r(ratio, 1),
                    "sources": int(sub["source_name"].nunique()), "neg_share_pct": s["neg_pct"],
                    "top_terms": [t for t, _ in tt["terms"][:6]], "top_phrases": [t for t, _ in tt["bigrams"][:4]],
                    "sample_posts": [str(t)[:220] for t in samples["text"]]})
    return sorted(out, key=lambda x: x["date"])


def advanced_formulas(focus, act, cov_all, primary, multi, cfg, has_dates, facts):
    F = {}
    # ---- Organic vs Coordinated: C = mean{duplication, burstiness, account_anomaly, reach_Gini}
    dup = float(focus["norm_text"].duplicated(keep=False).mean()) if len(focus) else 0
    burst = 0.0
    if has_dates and len(focus) > 10:
        hourly = focus.groupby(focus["date"].dt.floor("h")).size()
        mu, sd = hourly.mean(), hourly.std(ddof=0)
        burst = float(((sd - mu) / (sd + mu) + 1) / 2) if (sd + mu) else 0
    if has_dates and len(focus):
        per_author_day = focus.groupby([focus["author"], focus["day"]]).size()
        heavy = per_author_day[per_author_day >= 5].index.get_level_values(0).unique()
    else:
        heavy = []
    anomaly = float(focus["author"].isin(heavy).mean()) if len(focus) else 0
    gini = _gini(focus["reach"].to_numpy())
    C = np.mean([dup, burst, anomaly, gini])
    F["organic_vs_coordinated"] = {
        "C": _r(C, 2), "duplication": _r(dup, 2), "burstiness": _r(burst, 2), "account_anomaly": _r(anomaly, 2),
        "reach_gini": _r(gini, 2),
        "reading": "Organic" if C < 0.35 else ("Mixed — check top authors" if C < 0.55 else "Possible coordination"),
        "formula": "C = mean{duplication, burstiness, account_anomaly, reach_Gini}",
    }

    # ---- Newsroom escalation (latest full week)
    wk = [w for w in facts["weekly"] if not w["partial"]]
    if len(wk) >= 3:
        last, prev = wk[-1], wk[-2]
        base = [w["items"] for w in wk[:-1]]
        vz = (last["items"] - np.mean(base)) / (np.std(base) or 1)
        lw = focus[focus["week"] == pd.Timestamp(last["week"])]
        journo = float(lw.loc[lw["author_type"] == "Media", "engagement"].sum() / max(lw["engagement"].sum(), 1))
        accel = (last["reach"] - prev["reach"]) / max(prev["reach"], 1)
        intensity = abs(last["nss"] or 0)
        newsworthy = float((lw["channel"] == "News & media").mean()) if len(lw) else 0
        comps = {"journo_engage": journo, "velocity_z_norm": _sigmoid(vz), "reach_accel_norm": _sigmoid(accel),
                 "sent_intensity": intensity, "newsworthiness": newsworthy}
        E = float(np.mean(list(comps.values())))
        F["newsroom_escalation"] = {"E": _r(E, 2), "week": last["week"], **{k: _r(v, 2) for k, v in comps.items()},
                                    "velocity_z": _r(vz, 2),
                                    "reading": "High" if E >= 0.6 else ("Elevated" if E >= 0.45 else "Low"),
                                    "formula": "E = w1·journo_engage + w2·velocity_z + w3·reach_accel + w4·sent_intensity + w5·newsworthiness (equal weights)"}

    # ---- Journalist warming / cooling: slope of trailing mean sentiment of media-authored items
    if has_dates:
        media = focus[(focus["author_type"] == "Media") & focus["sentiment"].notna()]
        end = focus["date"].max()
        jt = {}
        for win in (30, 60, 90):
            sub = media[media["date"] >= end - pd.Timedelta(days=win)]
            daily = sub.groupby("day")["sentiment"].mean()
            if len(daily) >= 5:
                x = (daily.index - daily.index.min()).days.to_numpy(dtype=float)
                slope = float(np.polyfit(x, daily.to_numpy(dtype=float), 1)[0]) * 30  # per 30 days
                jt[f"{win}d"] = {"slope_per_30d": _r(slope, 3), "mean_sentiment": _r(daily.mean(), 2),
                                 "trend": "Warming" if slope > 0.02 else ("Cooling" if slope < -0.02 else "Flat")}
        F["journalist_trend"] = {"windows": jt, "media_items": int(len(media)),
                                 "formula": "trend = slope of trailing mean sentiment of media-authored items (30/60/90-day)"}

    # ---- Silence / vacuum risk per lens: V = topic_relevance × competitor_presence × (1 − owned_voice)
    vac = []
    comp_measurable = multi
    for r in facts["lenses"]:
        if not r.get("volume"):
            continue
        sub = act[act["lenses"].map(lambda L, n=r["lens"]: n in L)]
        owned = float((sub["author_type"] == "Brand-owned").mean()) if len(sub) else 0
        if comp_measurable:
            peers_all = cov_all[cov_all["entity"] != primary]
            peer_lens = peers_all[peers_all["lenses"].map(lambda L, n=r["lens"]: n in L)]
            comp = len(peer_lens) / max(len(peer_lens) + len(sub), 1)
        else:
            comp = 1.0
        V = (r["share_pct"] or 0) / 100 * comp * (1 - owned)
        vac.append({"lens": r["lens"], "V": _r(V, 3), "topic_relevance": _r((r["share_pct"] or 0) / 100, 3),
                    "competitor_presence": _r(comp, 2), "owned_voice": _r(owned, 3)})
    vac.sort(key=lambda x: -(x["V"] or 0))
    F["vacuum_risk"] = {"by_lens": vac[:8], "competitor_presence_measured": comp_measurable,
                        "formula": "V = topic_relevance × competitor_presence × (1 − owned_voice)",
                        "note": None if comp_measurable else "Single-brand dataset: competitor presence set to 1."}

    # ---- Narrative pull-through: needs 'our frame' terms
    frame = [t.strip().lower() for t in (cfg.get("frame_terms") or []) if t.strip()]
    if frame:
        rx = "|".join(re.escape(t) for t in frame)
        on = as_text(focus["text"]).str.lower().str.contains(rx, regex=True, na=False)
        tot = max(float(focus["reach"].sum()), 1)
        F["narrative_pull_through"] = {"PT": _r(float(focus.loc[on, "reach"].sum()) / tot, 3),
                                       "items_on_frame": int(on.sum()), "frame_terms": frame,
                                       "formula": "PT = Σ reach(mentions on OUR frame) ÷ Σ reach(all mentions on topic)"}
    else:
        F["narrative_pull_through"] = {"PT": None, "note": "Add 'our frame' key messages in analysis settings to compute.",
                                       "formula": "PT = Σ reach(mentions on OUR frame) ÷ Σ reach(all mentions on topic)"}

    # ---- Reputation resilience: R = w1·recovery_speed + w2·defender_ratio + w3·advocacy_vol + w4·sentiment_floor
    if wk:
        shares = [w["neg_share_pct"] or 0 for w in wk]
        base = np.mean(shares[: max(1, min(4, len(shares) // 3 or 1))])
        pk = int(np.argmax(shares))
        rec_weeks = next((i - pk for i in range(pk + 1, len(shares)) if shares[i] <= base * 1.25), None)
        recovery = 1 / (1 + rec_weeks) if rec_weeks is not None else 0.0
        st = sentiment_stats(act["sentiment"])
        defender = st["pos"] / max(st["pos"] + st["neg"], 1)
        advocacy = float(act["emotion"].isin(["praise", "advocacy"]).mean()) if len(act) else 0
        floor = (min(w["nss"] or 0 for w in wk) + 1) / 2
        R = 0.25 * recovery + 0.25 * defender + 0.25 * advocacy + 0.25 * floor
        F["reputation_resilience"] = {
            "R": _r(R, 2), "recovery_speed": _r(recovery, 2),
            "weeks_to_recover": rec_weeks, "recovered": rec_weeks is not None, "defender_ratio": _r(defender, 2),
            "advocacy_share": _r(advocacy, 3), "sentiment_floor": _r(floor, 2),
            "reading": "Resilient" if R >= 0.55 else ("Exposed" if R >= 0.35 else "Fragile"),
            "formula": "R = w1·recovery_speed + w2·defender_ratio + w3·advocacy_vol + w4·sentiment_floor (equal weights)"}
    return F


# ======================================================================== metric catalog
def metric_catalog(facts: dict) -> dict:
    """Flat key -> {value, label} used by template KPIs and number checks."""
    cat = {}

    def add(key, value, label, fmt="num"):
        if value is not None:
            cat[key] = {"value": value, "label": label, "format": fmt}

    dh, ov, ws = facts.get("data_health", {}), facts.get("overview", {}), facts.get("weekly_summary", {})
    add("data.total_ingested", dh.get("total_ingested"), "Items ingested")
    add("data.actionable_items", dh.get("actionable_items"), "Actionable items after noise gate")
    add("data.signal_rate_pct", dh.get("signal_rate_pct"), "Signal rate (actionable ÷ ingested)", "pct")
    add("data.noise_filtered_pct", dh.get("noise_filtered_pct"), "Share filtered as noise", "pct")
    for k, lab, f in [("items", "Items (coverage, focus brand)", "num"), ("negative_items", "Negative items", "num"),
                      ("positive_items", "Positive items", "num"), ("neg_share_pct", "Negative share", "pct"),
                      ("pos_share_pct", "Positive share", "pct"), ("nss", "Net sentiment score", "dec"),
                      ("nssw", "Reach-weighted net sentiment", "dec"), ("social_posts", "Social posts", "num"),
                      ("social_engagement", "Social engagement", "num"), ("eng_per_post", "Engagement per social post", "num"),
                      ("total_reach", "Total reach", "num"), ("unique_authors", "Unique authors", "num"),
                      ("media_share_pct", "Share of items from media authors", "pct"),
                      ("news_media_items", "News & media items", "num")]:
        add(f"overview.{k}", ov.get(k), lab, f)
    for k, lab, f in [("step_change_x", "Peak weekly negative volume vs baseline", "x"),
                      ("baseline_neg_per_week", "Baseline negative items per week", "num"),
                      ("peak_neg_volume", "Peak weekly negative items", "num"),
                      ("peak_neg_share_pct", "Peak weekly negative share", "pct"),
                      ("first4_neg_share_pct", "Negative share, first 4 weeks", "pct"),
                      ("last4_neg_share_pct", "Negative share, last 4 weeks", "pct")]:
        add(f"weekly.{k}", ws.get(k), lab, f)
    add("weekly.first_alert_week", ws.get("first_alert_week"), "First week negative share > 15%", "text")
    ps = facts.get("peer_summary") or {}
    add("peers.peer_avg_neg_share_pct", ps.get("peer_avg_neg_share_pct"), "Peer average negative share", "pct")
    add("peers.peers_negative_items_combined", ps.get("peers_negative_items_combined"), "Peers' negative items combined")
    for p in facts.get("peers", []):
        e = p["entity"]
        add(f"peer.{e}.share_of_voice_pct", p["share_of_voice_pct"], f"{e} share of voice", "pct")
        add(f"peer.{e}.neg_share_pct", p["neg_share_pct"], f"{e} negative share", "pct")
        add(f"peer.{e}.items", p["items"], f"{e} items")
    for r in facts.get("lenses", []):
        if r.get("volume"):
            n = r["lens"]
            add(f"lens.{n}.volume", r["volume"], f"{n}: comments")
            add(f"lens.{n}.share_pct", r["share_pct"], f"{n}: share", "pct")
            add(f"lens.{n}.nssw", r["nssw"], f"{n}: reach-weighted net sentiment", "dec")
            add(f"lens.{n}.neg_pct", r["neg_pct"], f"{n}: negative %", "pct")
    F = facts.get("formulas", {})
    add("formula.C", (F.get("organic_vs_coordinated") or {}).get("C"), "Organic vs coordinated index (C)", "dec")
    add("formula.E", (F.get("newsroom_escalation") or {}).get("E"), "Newsroom escalation (E)", "dec")
    add("formula.R", (F.get("reputation_resilience") or {}).get("R"), "Reputation resilience (R)", "dec")
    add("formula.PT", (F.get("narrative_pull_through") or {}).get("PT"), "Narrative pull-through (PT)", "dec")
    if facts.get("platforms"):
        top = facts["platforms"][0]
        add("platform.top.name", top["platform"], "Platform with most items", "text")
    for p in facts.get("platforms", []):
        add(f"platform.{p['platform']}.items", p["items"], f"{p['platform']} items")
        add(f"platform.{p['platform']}.engagement_share_pct", p["engagement_share_pct"], f"{p['platform']} share of engagement", "pct")
    return cat
