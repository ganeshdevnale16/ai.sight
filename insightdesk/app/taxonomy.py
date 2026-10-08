"""Seed perception lenses (Step 1), keyword assignment (Step 2 proxy), emotions & flags (Step 5).

Lenses are editable from the Settings tab. Each lens: name, terms (lower-case prefixes / phrases),
and an action hint used in the verdict line.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

DEFAULT_LENSES = [
    {"name": "Digital & service reliability", "action": "route to digital ops",
     "terms": ["app ", "mobile app", "netbanking", "net banking", "upi", "outage", "server down", "not working",
               "glitch", "otp", "login", "website", "transaction failed", "payment failed", "imps", "neft",
               "rtgs", "technical issue", "down since", "error"]},
    {"name": "Fees & charges", "action": "review pricing communication",
     "terms": ["charge", "fee", "fees", "hidden", "deducted", "annual fee", "gst", "hold charges",
               "minimum balance", "penal interest", "processing fee", "late payment", "debited without"]},
    {"name": "Customer support experience", "action": "escalate to service leadership",
     "terms": ["customer care", "customer service", "support", "complaint", "no response", "ticket",
               "call center", "call centre", "waiting", "relationship manager", "branch staff", "escalat",
               "unresolved", "_cares", "nobody", "no one is", "not responding", "follow up"]},
    {"name": "Fraud / scam mentions", "action": "route to fraud team",
     "terms": ["fraud", "froad", "scam", "phishing", "cyber", "hacked", "unauthori", "stolen", "fake call",
               "money lost", "siphon", "cheat", "duped", "vishing", "kyc update link"]},
    {"name": "Regulatory & legal chatter", "action": "early compliance-risk signal",
     "terms": ["rbi", "sebi", "ombudsman", "penalty", "fined", "enforcement directorate", " ed ", "compliance",
               "aml", "money laundering", "investigation", "regulator", "court", "nclat", "lawsuit",
               "class action", "securities fraud", "sdny", "legal notice", "consumer forum"]},
    {"name": "Governance & leadership", "action": "board & IR attention",
     "terms": ["ceo", "cfo", "board", "chairman", "governance", "resign", "retire", "succession", "leadership",
               "msrdc", "overreach", "management", "jagdishan", "md & ceo", "whistleblower"]},
    {"name": "Rewards / cards / product", "action": "amplify",
     "terms": ["reward", "points", "cashback", "credit card", "debit card", "offer", "lounge", "millennia",
               "regalia", "infinia", "launch", "new product", "fd rate", "savings account"]},
    {"name": "Loans & credit", "action": "review lending journey",
     "terms": ["loan", "home loan", "personal loan", "emi", "foreclos", "cibil", "credit score",
               "disburs", "sanction", "interest rate", "moratorium"]},
    {"name": "Investor & market", "action": "IR briefing",
     "terms": ["share price", "stock", "adr", "nyse", "hdb", "results", "earnings", "q1 ", "q2 ", "q3 ",
               "q4 ", "analyst", "target price", "market cap", "nifty", "sensex", "brokerage", "dividend"]},
    {"name": "Brand / CSR / sponsorship", "action": "goodwill — amplify",
     "terms": ["csr", "sponsor", "campaign", "initiative", "award", "foundation", "parivartan", "partnership",
               "community", "felicitat", "celebrat"]},
    {"name": "Rumour / misinformation", "action": "monitor and correct quickly",
     "terms": ["rumour", "rumor", "fake news", "hoax", "whatsapp forward", "misleading", "clarification",
               "is it true", "viral message", "false claim", "deepfake"]},
]

EMOTIONS = {
    "anger": ["worst", "pathetic", "disgusting", "shame", "shameless", "useless", "nonsense", "rubbish",
              "fed up", "hate", "!!!", "idiot", "harass"],
    "distrust": ["fraud", "froad", "scam", "cheat", "trust", "lie", "lying", "misleading", "hidden",
                 "loot", "looting", "daylight robbery", "fishy", "cover up", "cover-up"],
    "frustration": ["still", "again", "waiting", "since", "days", "weeks", "months", "no response",
                    "not resolved", "unresolved", "multiple times", "how many times", "every time"],
    "anxiety": ["worried", "scared", "afraid", "lost money", "is my money", "safe?", "urgent", "panic",
                "help me", "concerned"],
    "confusion": ["how do i", "how to", "why is", "what is", "confused", "unclear", "not sure", "anyone know",
                  "does anyone", "?"],
    "praise": ["thank", "great", "excellent", "helpful", "love", "best", "appreciate", "smooth", "kudos",
               "impressed", "good job", "well done"],
    "advocacy": ["recommend", "must try", "switch to", "proud", "loyal", "go for", "highly suggest"],
}
EMOTION_ORDER = ["anger", "distrust", "anxiety", "frustration", "praise", "advocacy", "confusion"]

FLAG_TERMS = {
    "risk": ["rbi", "ombudsman", "consumer court", "consumer forum", "legal notice", "lawyer", "sue ",
             "complaint to", "sebi", "police", " fir ", "cyber cell", "class action", "lawsuit", "court"],
    "complaint": ["complaint", "not resolved", "refund", "issue", "problem", "not working", "deducted",
                  "help", "please resolve", "escalate", "worst"],
    "misinformation": ["rumour", "rumor", "fake news", "hoax", "forward", "misleading", "false claim"],
}

POS_LEX = ["thank", "great", "excellent", "good", "helpful", "love", "best", "appreciate", "smooth",
           "happy", "kudos", "impressed", "well done", "recommend", "fast", "easy", "proud", "congrat",
           "award", "launch", "growth", "record", "strong"]
NEG_LEX = ["worst", "bad", "poor", "pathetic", "fraud", "froad", "scam", "cheat", "refund", "charged",
           "deducted", "not working", "failed", "issue", "problem", "complaint", "delay", "useless",
           "disappoint", "frustrat", "angry", "terrible", "horrible", "never", "hidden", "fine", "penalty",
           "lawsuit", "class action", "crisis", "resign", "probe", "investigation", "loss", "decline",
           "glitch", "outage", "harass", "lost", "stolen", "hacked", "unresolved", "no response"]


def _rx(terms):
    parts = []
    for t in terms:
        t = t.lower()
        esc = re.escape(t.strip())
        if t.startswith(" ") or t.endswith(" "):
            parts.append(r"(?<![a-z0-9])" + esc + r"(?![a-z0-9])")
        elif re.match(r"^[a-z]", t):
            parts.append(r"(?<![a-z0-9])" + esc)
        else:
            parts.append(esc)
    return re.compile("|".join(parts)) if parts else None


def as_text(s):
    """Object-dtype text so regex features (look-behind etc.) use Python's re engine everywhere."""
    import pandas as _pd
    vals = ["" if (v is None or (isinstance(v, float) and v != v) or v is _pd.NA) else str(v) for v in s.tolist()]
    return _pd.Series(vals, index=s.index, dtype=object)


def lens_list(custom=None):
    return custom if custom else DEFAULT_LENSES


def tag(df: pd.DataFrame, lenses=None) -> pd.DataFrame:
    """Adds lenses (list), emotion, flags (list) and a lexicon sentiment where provider sentiment is missing."""
    lenses = lens_list(lenses)
    low = " " + as_text(df["text"]).str.lower() + " "

    hits = {}
    for L in lenses:
        rx = _rx(L.get("terms", []))
        hits[L["name"]] = low.str.contains(rx.pattern, regex=True, na=False) if rx is not None else pd.Series(False, index=df.index)
    lens_mat = pd.DataFrame(hits)
    names = np.array(list(hits.keys()), dtype=object)
    arr = lens_mat.to_numpy()
    df["lenses"] = [list(names[row]) for row in arr]

    emo_hits = {e: low.str.count(_rx(t).pattern) for e, t in EMOTIONS.items()}
    emo_df = pd.DataFrame(emo_hits)[EMOTION_ORDER]
    best = emo_df.idxmax(axis=1)
    df["emotion"] = np.where(emo_df.max(axis=1) > 0, best, "neutral-info")

    flags = {f: low.str.contains(_rx(t).pattern, regex=True, na=False) for f, t in FLAG_TERMS.items()}
    fdf = pd.DataFrame(flags)
    fnames = np.array(list(flags.keys()), dtype=object)
    df["flags"] = [list(fnames[row]) for row in fdf.to_numpy()]

    # Lexicon sentiment only where the provider gave none.
    pos = low.str.count(_rx(POS_LEX).pattern)
    neg = low.str.count(_rx(NEG_LEX).pattern)
    lex = np.where(neg > pos, -1.0, np.where(pos > neg, 1.0, 0.0))
    missing = df["sentiment"].isna()
    df.loc[missing, "sentiment"] = lex[missing.to_numpy()]
    df.loc[missing, "sentiment_score"] = np.clip((pos - neg) / 3.0, -1, 1)[missing]
    df.loc[missing, "sentiment_source"] = "lexicon"
    # complaint flag only counts when not positive
    df["flags"] = [f if s != 1 else [x for x in f if x != "complaint"] for f, s in zip(df["flags"], df["sentiment"])]
    df["emergent_topic"] = ""
    return df
