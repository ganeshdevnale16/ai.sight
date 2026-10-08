"""Read an uploaded data file and map its columns onto one canonical schema.

Works with social-listening / media-intelligence exports (Signal AI, Meltwater, Brandwatch,
Talkwalker style) but falls back to content-based detection for any column layout.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

URL_LIKE = re.compile(r"^\s*https?:\\?//\S*\s*$", re.I)
HASHES = re.compile(r"^#+$")

# Canonical field -> candidate source column names, in order of preference.
CANDIDATES = {
    "url": ["url", "permalink", "link", "post_url", "article_url", "display_url"],
    "date": ["published", "published_at", "publish_date", "date", "created_at", "created", "post_date",
             "timestamp", "indexed", "search_indexed", "time"],
    "text": ["content", "text", "message", "body", "post_text", "full_text", "content_snippet", "snippet",
             "description", "summary", "mention"],
    "title": ["title", "headline", "title_snippet", "subject"],
    "author": ["extra_author_attributes.name", "author", "author_name", "user_name", "account_name", "user", "name"],
    "handle": ["extra_author_attributes.short_name", "handle", "screen_name", "username", "author_handle"],
    "source_type": ["source_type", "media_type", "channel", "platform", "network", "source"],
    "post_type": ["post_type", "content_type", "type"],
    "sentiment": ["sentiment", "sentiment_label", "tone", "polarity", "sentiment_score"],
    "entity": ["matched_profile", "entity", "brand", "profile", "search_name", "query", "topic", "keyword"],
    "country": ["extra_article_attributes.world_data.country", "extra_author_attributes.world_data.country",
                "extra_source_attributes.world_data.country", "country", "location"],
    "lang": ["lang", "language"],
    "engagement": ["engagement", "engagements", "total_engagement", "interactions"],
    "reach": ["reach", "impressions", "potential_reach", "audience", "estimated_reach"],
    "followers": ["followers", "author_followers", "follower_count"],
    "source_name": ["extra_source_attributes.name", "source_name", "publication", "publisher", "site_name",
                    "outlet", "domain_url", "host_url", "root_url"],
}
FIELD_HELP = {
    "url": "Link to the post or article", "date": "Publication date/time", "text": "Main post / article text",
    "title": "Headline (news)", "author": "Author display name", "handle": "Author handle / username",
    "source_type": "Platform or media type", "post_type": "Post type (comment, reply, post)",
    "sentiment": "Sentiment from the provider", "entity": "Brand / profile the item matched",
    "country": "Country", "lang": "Language", "engagement": "Total engagement",
    "reach": "Reach / impressions", "followers": "Author followers", "source_name": "Publication / site",
}

POS_WORDS = {"positive", "pos", "p", "1", "+1", "favourable", "favorable", "good"}
NEG_WORDS = {"negative", "neg", "n", "-1", "unfavourable", "unfavorable", "bad"}
NEU_WORDS = {"neutral", "neu", "0", "mixed", "none"}

SOCIAL_PLATFORMS = {"X / Twitter", "Instagram", "Facebook", "LinkedIn", "YouTube", "Reddit", "Threads",
                    "TikTok", "Telegram", "Quora", "Forums", "Reviews", "Social (other)"}
PLATFORM_TOKENS = [
    ("TWITTER", "X / Twitter"), ("INSTAGRAM", "Instagram"), ("FACEBOOK", "Facebook"),
    ("LINKEDIN", "LinkedIn"), ("YOUTUBE", "YouTube"), ("REDDIT", "Reddit"), ("THREADS", "Threads"),
    ("TIKTOK", "TikTok"), ("TELEGRAM", "Telegram"), ("QUORA", "Quora"), ("PODCAST", "Podcast"),
    ("BROADCAST", "Broadcast"), ("TV", "Broadcast"), ("RADIO", "Broadcast"), ("PRINT", "Print"),
    ("FORUM", "Forums"), ("FORUMS", "Forums"), ("REVIEW", "Reviews"), ("REVIEWS", "Reviews"),
    ("BLOG", "Blogs"), ("BLOGS", "Blogs"), ("NEWS", "Online news"), ("ONLINE", "Online news"),
    ("SOCIALMEDIA", "Social (other)"), ("SOCIAL", "Social (other)"),
]
DOMAIN_PLATFORMS = [
    ("twitter.com", "X / Twitter"), ("x.com", "X / Twitter"), ("instagram.com", "Instagram"),
    ("facebook.com", "Facebook"), ("fb.com", "Facebook"), ("linkedin.com", "LinkedIn"),
    ("youtube.com", "YouTube"), ("youtu.be", "YouTube"), ("reddit.com", "Reddit"),
    ("threads.net", "Threads"), ("tiktok.com", "TikTok"), ("t.me", "Telegram"), ("quora.com", "Quora"),
]


# ------------------------------------------------------------------ reading
def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        df = max(sheets.values(), key=len)
    elif ext == ".json":
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(raw, dict):
            raw = next((v for v in raw.values() if isinstance(v, list)), [raw])
        df = pd.json_normalize(raw)
    elif ext in (".md", ".txt") and _looks_like_markdown(path):
        df = _read_markdown_table(path.read_text(encoding="utf-8", errors="replace"))
    else:
        df = _read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    # Excel shows "########" when a column is too narrow; that is display noise, not data.
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            s = df[c].astype("string")
            df[c] = s.where(~s.str.fullmatch(r"#+", na=False)).str.replace(r"\\:", ":", regex=True)
    return df.dropna(how="all")


def _read_csv(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            if df.shape[1] == 1:
                df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            return df
        except UnicodeDecodeError as e:
            last_err = e
    raise ValueError(f"Could not read the file as CSV: {last_err}")


def _looks_like_markdown(path: Path) -> bool:
    head = path.read_text(encoding="utf-8", errors="replace")[:5000]
    return sum(1 for line in head.splitlines() if line.strip().startswith("|")) >= 2


def _read_markdown_table(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        rows.append(cells)
    if len(rows) < 2:
        raise ValueError("No table found in the text file.")
    header = rows[0]
    data = [(r + [""] * len(header))[: len(header)] for r in rows[1:]]
    return pd.DataFrame(data, columns=header).replace({"": np.nan})


# ------------------------------------------------------------------ validators
def _nonnull_str(s: pd.Series, n=500) -> pd.Series:
    s = s.dropna()
    if len(s) > n:
        s = s.sample(n, random_state=1)
    return s.astype(str)


def _is_url(s: pd.Series) -> bool:
    v = _nonnull_str(s)
    return len(v) > 0 and v.str.match(r"\s*https?://", case=False).mean() > 0.5


def _is_text(s: pd.Series) -> bool:
    v = _nonnull_str(s)
    if len(v) == 0:
        return False
    url_share = v.str.match(URL_LIKE).mean()
    return url_share < 0.5 and v.str.len().mean() >= 12 and v.str.count(r"\s").mean() >= 1.5


def _is_numeric(s: pd.Series) -> bool:
    v = s.dropna()
    if len(v) == 0:
        return False
    return pd.to_numeric(v, errors="coerce").notna().mean() > 0.6


def _is_date(s: pd.Series) -> bool:
    v = s.dropna()
    if len(v) == 0:
        return False
    if len(v) > 400:
        v = v.sample(400, random_state=1)
    return parse_dates(v).notna().mean() > 0.6


def _sentiment_values(s: pd.Series):
    num = pd.to_numeric(s, errors="coerce")
    nonnull = s.notna()
    if nonnull.sum() and num[nonnull].notna().mean() > 0.8:
        v = num.astype(float)
        if v.max() > 1.5 and v.min() >= 0:  # rating 1..5 style
            v = (v - 3) / 2
        score = v.clip(-1, 1)
        label = np.where(score > 0.05, 1, np.where(score < -0.05, -1, 0)).astype(float)
        label[score.isna().to_numpy()] = np.nan
        return pd.Series(label, index=s.index), score
    low = s.astype("string").str.strip().str.lower()
    label = pd.Series(np.nan, index=s.index, dtype=float)
    label[low.isin(POS_WORDS).fillna(False).to_numpy()] = 1
    label[low.isin(NEG_WORDS).fillna(False).to_numpy()] = -1
    label[low.isin(NEU_WORDS).fillna(False).to_numpy()] = 0
    return label, label.copy()


def _is_sentiment(s: pd.Series) -> bool:
    v = s.dropna()
    if len(v) == 0:
        return False
    label, _ = _sentiment_values(v)
    return label.notna().mean() > 0.5


VALIDATORS = {
    "url": _is_url, "date": _is_date, "text": _is_text, "title": _is_text, "engagement": _is_numeric,
    "reach": _is_numeric, "followers": _is_numeric, "sentiment": _is_sentiment,
}


def parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s, errors="coerce")
        return out.dt.tz_localize(None) if out.dt.tz is not None else out
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.8:
        if num.between(20000, 80000).mean() > 0.5:  # Excel serial dates
            return pd.to_datetime("1899-12-30") + pd.to_timedelta(num, unit="D")
        if num.between(1e9, 2.5e9).mean() > 0.5:
            return pd.to_datetime(num, unit="s", errors="coerce")
        if num.between(1e12, 2.5e12).mean() > 0.5:
            return pd.to_datetime(num, unit="ms", errors="coerce")
        return pd.Series(pd.NaT, index=s.index)
    txt = s.astype("string")
    dayfirst = bool(txt.str.match(r"^\s*(1[3-9]|2\d|3[01])[/.-]\d{1,2}[/.-]\d{2,4}", na=False).any())
    out = pd.to_datetime(txt, errors="coerce", utc=True, format="mixed", dayfirst=dayfirst)
    return out.dt.tz_convert(None)


# ------------------------------------------------------------------ mapping
def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")


def auto_map(df: pd.DataFrame) -> dict:
    """Return {field: {"column": name|None, "method": "..."}}."""
    normcols = {}
    for c in df.columns:
        normcols.setdefault(_norm(c), c)
    used, mapping = set(), {}
    order = ["url", "date", "text", "title", "author", "handle", "source_type", "post_type", "sentiment",
             "entity", "country", "lang", "engagement", "reach", "followers", "source_name"]
    for field in order:
        validator = VALIDATORS.get(field)
        chosen = None
        for cand in CANDIDATES[field]:
            col = normcols.get(_norm(cand))
            if col is None or col in used or df[col].notna().sum() == 0:
                continue
            if validator is None or validator(df[col]):
                chosen = col
                break
        method = "name"
        if chosen is None and field in ("date", "text"):
            chosen = _content_fallback(df, field, used)
            method = "content"
        if chosen is not None and field not in ("title",):
            used.add(chosen)
        mapping[field] = {"column": chosen, "method": method if chosen else None}
    if mapping["title"]["column"] and mapping["title"]["column"] == mapping["text"]["column"]:
        mapping["title"] = {"column": None, "method": None}
    return mapping


def _content_fallback(df, field, used):
    best, best_score = None, 0
    for c in df.columns:
        if c in used or df[c].notna().sum() == 0:
            continue
        s = df[c]
        if field == "date" and _is_date(s):
            score = parse_dates(s.dropna().head(400)).notna().mean()
        elif field == "text" and _is_text(s):
            score = _nonnull_str(s).str.count(r"\s").mean()
        else:
            continue
        if score > best_score:
            best, best_score = c, score
    return best


def _col(df, mapping, field):
    m = mapping.get(field) or {}
    c = m.get("column")
    return df[c] if c and c in df.columns else None


def _num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(float) if s is not None else None


def _platform_from_type(value: str) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    tokens = set(re.split(r"[^A-Z0-9]+", value.upper()))
    joined = value.upper()
    for tok, name in PLATFORM_TOKENS:
        if tok in tokens or (len(tok) > 4 and tok in joined):
            if name in ("Social (other)",) and any(t in joined for t, _ in PLATFORM_TOKENS[:12]):
                continue
            return name
    return None


def _platform_from_url(url: str) -> str | None:
    if not isinstance(url, str):
        return None
    m = re.match(r"\s*https?:\\?//(?:www\.|m\.)?([^/\s]+)", url, re.I)
    if not m:
        return None
    host = m.group(1).lower()
    for dom, name in DOMAIN_PLATFORMS:
        if host == dom or host.endswith("." + dom):
            return name
    return "Online news"


def build_canonical(raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    n = len(raw)
    out = pd.DataFrame(index=raw.index)
    out["row_id"] = np.arange(n)

    url = _col(raw, mapping, "url")
    out["url"] = url.astype("string") if url is not None else pd.Series(pd.NA, index=raw.index, dtype="string")

    d = _col(raw, mapping, "date")
    out["date"] = parse_dates(d) if d is not None else pd.NaT

    text = _col(raw, mapping, "text")
    title = _col(raw, mapping, "title")
    t = text.astype("string").fillna("") if text is not None else pd.Series("", index=raw.index, dtype="string")
    if title is not None:
        ti = title.astype("string").fillna("")
        ti = ti.where(~ti.str.match(URL_LIKE, na=False), "")
        add = (ti.str.len() > 0) & ~pd.Series([a[:60] in b for a, b in zip(ti, t)], index=raw.index)
        t = t.where(~add, ti + ". " + t)
        out["title"] = ti
    else:
        out["title"] = ""
    out["text"] = t.str.replace(r"\s+", " ", regex=True).str.strip()

    author = _col(raw, mapping, "author")
    handle = _col(raw, mapping, "handle")
    out["handle"] = handle.astype("string") if handle is not None else pd.NA
    if handle is None and url is not None:
        out["handle"] = url.astype("string").str.extract(
            r"(?:twitter|x|instagram)\.com/([A-Za-z0-9_.]+)", expand=False)
    out["author"] = author.astype("string") if author is not None else out["handle"]
    out["author"] = out["author"].fillna(out["handle"]).fillna("Unknown")

    # followers: explicit column, else max across any *followers*/*subscribers* columns
    fol = _col(raw, mapping, "followers")
    if fol is not None:
        out["followers"] = _num(fol)
    else:
        cols = [c for c in raw.columns if re.search(r"followers|subscribers|group_members", c, re.I)
                and not re.search(r"following", c, re.I)]
        out["followers"] = raw[cols].apply(pd.to_numeric, errors="coerce").max(axis=1).fillna(0) if cols else 0.0

    # engagement: explicit column, else sum of interaction columns
    eng = _col(raw, mapping, "engagement")
    eng_v = _num(eng) if eng is not None else None
    if eng_v is None or eng_v.sum() == 0:
        cols = [c for c in raw.columns
                if re.search(r"likes|shares|retweets|replies|comments|reactions|repins|_pins|quote", c, re.I)
                and not re.search(r"dislike|followers|source_extended|reactions_total", c, re.I)]
        if cols:
            eng_v = raw[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    out["engagement"] = eng_v if eng_v is not None else 0.0

    reach = _col(raw, mapping, "reach")
    reach_v = _num(reach) if reach is not None else None
    if reach_v is None or reach_v.sum() == 0:
        cols = [c for c in raw.columns if re.search(r"impressions|url_views|video_views|_views$", c, re.I)]
        if cols:
            reach_v = raw[cols].apply(pd.to_numeric, errors="coerce").fillna(0).max(axis=1)
        if reach_v is None or reach_v.sum() == 0:
            reach_v = out["followers"].astype(float)
    out["reach"] = reach_v

    st = _col(raw, mapping, "source_type")
    plat = st.astype("string").map(_platform_from_type) if st is not None else pd.Series(None, index=raw.index)
    if url is not None:
        from_url = url.astype("string").map(_platform_from_url)
        plat = plat.where(plat.notna() & (plat != "Social (other)"), from_url.where(from_url.notna(), plat))
    out["platform"] = plat.fillna("Other").astype("string")
    out["channel"] = np.where(out["platform"].isin(SOCIAL_PLATFORMS), "Social", "News & media")

    pt = _col(raw, mapping, "post_type")
    out["post_type"] = pt.astype("string").fillna("") if pt is not None else ""
    out["is_reply"] = out["post_type"].astype("string").str.contains("comment|reply", case=False, na=False)

    sent = _col(raw, mapping, "sentiment")
    if sent is not None:
        lab, score = _sentiment_values(sent)
        out["sentiment"], out["sentiment_score"] = lab, score
        out["sentiment_source"] = np.where(lab.notna(), "provider", "")
    else:
        out["sentiment"], out["sentiment_score"], out["sentiment_source"] = np.nan, np.nan, ""

    ent = _col(raw, mapping, "entity")
    if ent is not None:
        e = ent.astype("string").str.split(r"[,;|]", regex=True).str[0]
        out["entity"] = e.str.split("/").str[-1].str.strip().fillna("Unspecified")
    else:
        out["entity"] = "Unspecified"

    for field in ("country", "lang", "source_name"):
        c = _col(raw, mapping, field)
        out[field] = c.astype("string").fillna("Unknown") if c is not None else "Unknown"
    if (out["source_name"] == "Unknown").all() and url is not None:
        out["source_name"] = url.astype("string").str.extract(r"https?://(?:www\.)?([^/]+)", expand=False).fillna("Unknown")

    out = out.reset_index(drop=True)
    return out


def column_profile(raw: pd.DataFrame, limit=400) -> list:
    prof = []
    for c in raw.columns[:limit]:
        s = raw[c]
        sample = s.dropna().astype(str).head(2).tolist()
        prof.append({"column": c, "non_null": int(s.notna().sum()),
                     "sample": [x[:80] for x in sample]})
    return prof
