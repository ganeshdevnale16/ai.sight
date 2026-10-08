"""STEP 0 — relevance & noise gate.

Two sets come out of this:
  * coverage set   = de-duplicated, not spam / bot / pure noise / brand-mismatch.
                     Used for volume, share of voice, sentiment trajectory (all media).
  * actionable set = coverage items whose Actionability score >= theta.
                     Used for perception lenses (Steps 1–7).
Nothing is deleted: every removed row keeps an exclude_reason so the excluded bin stays queryable.

Actionability(m) = w1·relevance + w2·has_opinion + w3·specificity + w4·(1 − spam_prob) + w5·(1 − bot_prob)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .taxonomy import as_text

WEIGHTS = {"relevance": 0.30, "has_opinion": 0.25, "specificity": 0.15, "not_spam": 0.15, "not_bot": 0.15}

SPAM_RE = re.compile(
    r"(referral code|use my code|promo code|sign ?up bonus|earn (rs\.?|₹|\$|inr)\s?\d|work from home|"
    r"whatsapp\s*(me|us|now|\+?\d)|dm (me|for)|click (the |on )?link|instant loan|loan (approved|available) "
    r"in|crypto|bitcoin|forex|giveaway|airdrop|t\.me/|bit\.ly|tinyurl|join (my|our) (channel|group)|"
    r"guaranteed (return|profit)|double your money)", re.I)
JOB_RE = re.compile(r"\b(hiring|job opening|vacanc(y|ies)|walk[- ]?in|apply now|recruitment|we are hiring|"
                    r"job alert|naukri)\b", re.I)
TICKER_RE = re.compile(r"(^\s*\$[A-Z]{2,6}\b|\b(nse|bse|nyse):\s?[A-Z]+|target price|stock (alert|tip)s?|"
                       r"buy above|sell below|\bsl\s?\d)", re.I)
QUERY_RE = re.compile(r"(ifsc code|branch timing|branch hours|working hours|nearest branch|swift code)", re.I)
CONTENTLESS = {"+1", "same", "same here", "me too", "ok", "okay", "nice", "good", "thanks", "thank you",
               "true", "agreed", "yes", "no", "wow", "lol", "noted", "👍", "🙏", "👏"}
OPINION_RE = re.compile(
    r"(\bi\b|\bmy\b|\bme\b|\bwe\b|\bour\b|worst|bad|poor|pathetic|fraud|froad|scam|cheat|refund|charged|"
    r"deducted|not working|failed|issue|problem|complain|delay|useless|disappoint|frustrat|angry|terrible|"
    r"horrible|never|why|please|help|great|excellent|good|love|thank|best|helpful|recommend|should|shame|"
    r"crisis|concern|lapse|questions?|criticis|slam|warn|alleg|!|\?)", re.I)
SPECIFIC_RE = re.compile(
    r"(account|card|loan|emi|upi|net ?banking|\bapp\b|branch|atm|\bfd\b|deposit|charge|fee|refund|"
    r"transaction|kyc|locker|cheque|otp|customer care|ceo|cfo|board|chairman|rbi|sebi|fine|penalt|lawsuit|"
    r"class action|outage|glitch|fraud|scam|phishing|interest|mutual fund|nav\b|insurance|policy|reward|"
    r"cashback|results|q[1-4]|governance|msrdc|\bed\b|\d)", re.I)
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"[@#][\w_]+")


def _normalise(text: pd.Series) -> pd.Series:
    t = as_text(text).str.lower()
    t = t.str.replace(URL_RE, " ", regex=True).str.replace(MENTION_RE, " ", regex=True)
    t = t.str.replace(r"[^a-z\u0900-\u097f ]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return t


def brand_terms_for(df: pd.DataFrame, cfg: dict) -> list:
    terms = [t.strip().lower() for t in (cfg.get("brand_terms") or []) if t.strip()]
    if not terms:
        ents = [e for e in df["entity"].dropna().unique().tolist() if e and e != "Unspecified"]
        for e in ents[:10]:
            terms.append(str(e).lower())
            terms.append(str(e).lower().replace(" ", ""))
    return sorted(set(t for t in terms if len(t) >= 2))


def apply_noise_gate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    theta = float(cfg.get("theta", 0.5))
    exclude_terms = [t.strip().lower() for t in (cfg.get("exclude_terms") or []) if t.strip()]
    text = as_text(df["text"])
    low = text.str.lower()
    norm = _normalise(text)
    words = norm.str.split().str.len().fillna(0)

    reason = pd.Series("", index=df.index, dtype=object)

    # duplicates (same URL, or same author + identical text)
    dup_url = df["url"].notna() & df["url"].duplicated(keep="first")
    dup_text = (norm.str.len() > 0) & pd.Series(list(zip(df["author"], norm)), index=df.index).duplicated(keep="first")
    reason[dup_url | dup_text] = "Duplicate"

    # bot / automated: same normalised text posted by >=3 distinct authors, or template replies
    # Only social posts: syndicated news (same wire story across outlets) is legitimate coverage.
    social = (df["channel"] == "Social").to_numpy()
    day = df["date"].dt.floor("D") if df["date"].notna().any() else pd.Series(0, index=df.index)
    grp = pd.DataFrame({"n": norm, "a": df["author"], "d": day})[social & (norm.str.len() > 15).to_numpy()]
    stats = grp.groupby(["n", "d"])["a"].nunique()
    botty = set(stats[stats >= int(cfg.get("bot_min_authors", 5))].index.get_level_values(0))
    bot_prob = (norm.isin(botty) & pd.Series(social, index=df.index)).astype(float) * 0.9
    reason[(reason == "") & (bot_prob > 0.5)] = "Bot & automated"

    spam = low.str.contains(SPAM_RE, na=False) | (text.str.count(r"#\w+") > 6)
    spam_prob = spam.astype(float) * 0.9
    reason[(reason == "") & spam] = "Spam & promotional"

    letters = text.str.count(r"[A-Za-z\u0900-\u097f]")
    pure_noise = (letters < 2) | ((letters / text.str.len().clip(lower=1)) < 0.25) & (text.str.len() < 40)
    reason[(reason == "") & pure_noise] = "Pure noise"

    contentless = norm.isin(CONTENTLESS) | (words <= 2) & ~low.str.contains(SPECIFIC_RE, na=False)
    reason[(reason == "") & contentless] = "Contentless engagement"

    # relevance / brand-mismatch
    bterms = brand_terms_for(df, cfg)
    if bterms:
        brx = re.compile("|".join(re.escape(t) for t in bterms))
        mentions_brand = low.str.contains(brx, na=False) | as_text(df["url"]).str.lower().str.contains(brx, na=False)
    else:
        mentions_brand = pd.Series(True, index=df.index)
    tagged = as_text(df["entity"]).ne("Unspecified")
    relevance = np.where(mentions_brand, 1.0, np.where(tagged, 0.6, 0.3))
    if exclude_terms:
        erx = re.compile("|".join(re.escape(t) for t in exclude_terms))
        mismatch = low.str.contains(erx, na=False) & ~low.str.contains(brx, na=False) if bterms else low.str.contains(erx, na=False)
        relevance = np.where(mismatch, 0.0, relevance)
        reason[(reason == "") & mismatch] = "Off-topic / brand-mismatch"

    # non-opinion factual noise
    factual = low.str.contains(JOB_RE, na=False) | low.str.contains(TICKER_RE, na=False) | low.str.contains(QUERY_RE, na=False)

    has_opinion = np.where(low.str.contains(OPINION_RE, na=False), 1.0, np.where(words > 12, 0.3, 0.0))
    has_opinion = np.where(factual, 0.0, has_opinion)
    specificity = np.where(low.str.contains(SPECIFIC_RE, na=False), 1.0, 0.0)

    score = (WEIGHTS["relevance"] * relevance + WEIGHTS["has_opinion"] * has_opinion
             + WEIGHTS["specificity"] * specificity + WEIGHTS["not_spam"] * (1 - spam_prob)
             + WEIGHTS["not_bot"] * (1 - bot_prob))

    df["relevance"] = relevance
    df["has_opinion"] = has_opinion
    df["specificity"] = specificity
    df["spam_prob"] = spam_prob
    df["bot_prob"] = bot_prob
    df["actionability"] = np.round(score, 3)
    df["norm_text"] = norm

    df["in_coverage"] = reason.eq("").to_numpy()
    # job posts, stock-ticker chatter and branch/IFSC queries never count as actionable opinion
    low_score = df["in_coverage"] & ((df["actionability"] < theta) | factual.to_numpy())
    sub_reason = np.where(factual, "Non-opinion / factual noise", "Below actionability threshold")
    reason = reason.where(~low_score, pd.Series(sub_reason, index=df.index))
    df["exclude_reason"] = reason
    df["actionable"] = reason.eq("").to_numpy()
    return df


def gate_summary(df: pd.DataFrame, theta: float) -> dict:
    total = len(df)
    kept = int(df["actionable"].sum())
    cov = int(df["in_coverage"].sum())
    by_reason = df.loc[df["exclude_reason"] != "", "exclude_reason"].value_counts().to_dict()
    return {
        "total_ingested": total,
        "coverage_items": cov,
        "actionable_items": kept,
        "excluded_items": total - kept,
        "signal_rate_pct": round(100 * kept / total, 1) if total else 0,
        "noise_filtered_pct": round(100 * (total - kept) / total, 1) if total else 0,
        "excluded_by_reason": {k: int(v) for k, v in by_reason.items()},
        "theta": theta,
        "weights": WEIGHTS,
    }
