"""A small, safe query language over the cleaned dataset.

The language model never writes or runs code. It sends a JSON spec, this module validates it
against a whitelist of fields/operators and computes the answer with pandas.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .taxonomy import as_text

FIELDS = {
    "date": "datetime", "platform": "category", "channel": "category", "post_type": "category",
    "sentiment": "category", "sentiment_score": "number", "entity": "category", "country": "category",
    "lang": "category", "author": "category", "handle": "category", "author_type": "category",
    "source_name": "category", "engagement": "number", "reach": "number", "followers": "number",
    "lens": "list", "flag": "list", "emotion": "category", "exclude_reason": "category",
    "emergent_topic": "category", "text": "text", "actionability": "number", "is_reply": "bool",
}
COLUMN_OF = {"lens": "lenses", "flag": "flags", "sentiment": "sentiment_label"}
OPS = {"eq", "ne", "in", "not_in", "contains", "gt", "gte", "lt", "lte", "between", "is_null", "not_null"}
METRICS = {"count", "share_pct", "sum", "mean", "median", "min", "max", "nunique", "neg_share_pct",
           "pos_share_pct", "nss", "nssw"}
SENT_LABEL = {1.0: "Positive", 0.0: "Neutral", -1.0: "Negative"}

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "dataset": {"type": "string", "enum": ["coverage", "actionable", "excluded", "all"],
                    "description": "coverage = de-duplicated relevant items (default, use for volume/sentiment); "
                                   "actionable = items that passed the noise gate (use for perception lenses); "
                                   "excluded = the filtered-out bin; all = everything ingested."},
        "filters": {"type": "array", "items": {"type": "object", "properties": {
            "field": {"type": "string"}, "op": {"type": "string", "enum": sorted(OPS)},
            "value": {"description": "string, number, list (for in/not_in/between) or ISO date"}},
            "required": ["field", "op"]}},
        "group_by": {"type": "array", "items": {"type": "string"},
                     "description": "Fields to group by. Use 'lens' or 'flag' to break down by perception lens/flag."},
        "time_grain": {"type": "string", "enum": ["day", "week", "month"],
                       "description": "Adds a 'period' grouping column (week starts Monday)."},
        "metrics": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string", "enum": sorted(METRICS)},
            "field": {"type": "string", "description": "Numeric field for sum/mean/median/min/max, any field for nunique"}},
            "required": ["name"]}},
        "sort_by": {"type": "string", "description": "A metric name (e.g. 'count', 'sum_engagement') or a group field"},
        "descending": {"type": "boolean"},
        "limit": {"type": "integer", "description": "Max rows returned (default 25, max 200)"},
    },
}


class QueryError(ValueError):
    pass


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment_label"] = df["sentiment"].map(SENT_LABEL).fillna("Unscored")
    for c in ("author_type",):
        if c not in df:
            df[c] = "Unknown"
    return df


def _select(df, name):
    name = name or "coverage"
    if name == "coverage":
        return df[df["in_coverage"]]
    if name == "actionable":
        return df[df["actionable"]]
    if name == "excluded":
        return df[~df["actionable"]]
    if name == "all":
        return df
    raise QueryError(f"Unknown dataset '{name}'")


def _col(field):
    if field not in FIELDS:
        raise QueryError(f"Unknown field '{field}'. Allowed: {', '.join(sorted(FIELDS))}")
    return COLUMN_OF.get(field, field)


def _coerce(field, v):
    kind = FIELDS[field]
    if kind == "datetime":
        return pd.to_datetime(v)
    if kind == "number":
        return float(v)
    if field == "sentiment" and isinstance(v, (int, float)):
        return SENT_LABEL.get(float(v), str(v))
    return v


def _mask(d: pd.DataFrame, f: dict) -> pd.Series:
    field, op, val = f.get("field"), f.get("op"), f.get("value")
    if op not in OPS:
        raise QueryError(f"Unknown operator '{op}'")
    col = _col(field)
    s = d[col]
    kind = FIELDS[field]
    if kind == "list":
        vals = val if isinstance(val, list) else [val]
        vals_l = [str(v).lower() for v in vals]
        has = s.map(lambda L: any(str(x).lower() in vals_l for x in (L or [])))
        if op in ("eq", "in", "contains"):
            if op == "contains":
                return s.map(lambda L: any(any(v in str(x).lower() for v in vals_l) for x in (L or [])))
            return has
        if op in ("ne", "not_in"):
            return ~has
        if op == "is_null":
            return s.map(lambda L: len(L or []) == 0)
        if op == "not_null":
            return s.map(lambda L: len(L or []) > 0)
        raise QueryError(f"Operator '{op}' not supported for list field '{field}'")
    if op == "is_null":
        return s.isna() | (as_text(s) == "")
    if op == "not_null":
        return s.notna() & (as_text(s) != "")
    if op == "contains":
        return as_text(s).str.contains(re.escape(str(val)), case=False, regex=True, na=False)
    if op in ("in", "not_in"):
        vals = val if isinstance(val, list) else [val]
        if kind in ("category", "text"):
            low = as_text(s).str.lower()
            m = low.isin([str(_coerce(field, v)).lower() for v in vals])
        else:
            m = s.isin([_coerce(field, v) for v in vals])
        return m if op == "in" else ~m
    if op == "between":
        if not isinstance(val, list) or len(val) != 2:
            raise QueryError("'between' needs a [low, high] list")
        lo, hi = _coerce(field, val[0]), _coerce(field, val[1])
        if kind == "datetime" and isinstance(val[1], str) and len(val[1]) <= 10:
            hi = hi + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        return (s >= lo) & (s <= hi)
    v = _coerce(field, val)
    if op in ("eq", "ne") and kind in ("category", "text"):
        m = as_text(s).str.lower() == str(v).lower()
        return m if op == "eq" else ~m
    if op in ("eq", "ne") and kind == "bool":
        v = str(val).lower() in ("true", "1", "yes")
        return (s == v) if op == "eq" else (s != v)
    return {"eq": s == v, "ne": s != v, "gt": s > v, "gte": s >= v, "lt": s < v, "lte": s <= v}[op]


def _metric(sub: pd.DataFrame, m: dict, base_total: int):
    name = m.get("name", "count")
    field = m.get("field")
    if name not in METRICS:
        raise QueryError(f"Unknown metric '{name}'")
    if name == "count":
        return len(sub)
    if name == "share_pct":
        return round(100 * len(sub) / base_total, 1) if base_total else None
    sent = sub["sentiment"].dropna()
    if name in ("neg_share_pct", "pos_share_pct"):
        if not len(sent):
            return None
        return round(100 * ((sent == (-1 if name == "neg_share_pct" else 1)).sum()) / len(sent), 1)
    if name == "nss":
        return round(float(((sent == 1).sum() - (sent == -1).sum()) / len(sent)), 2) if len(sent) else None
    if name == "nssw":
        if not len(sent):
            return None
        w = sub.loc[sent.index, "reach"].clip(lower=0) + 1
        return round(float((w * sent).sum() / w.sum()), 2)
    if not field:
        raise QueryError(f"Metric '{name}' needs a field")
    col = _col(field)
    if name == "nunique":
        return int(sub[col].nunique())
    if FIELDS[field] != "number":
        raise QueryError(f"Metric '{name}' needs a numeric field, got '{field}'")
    v = getattr(sub[col], name)()
    return None if pd.isna(v) else round(float(v), 2)


def _mname(m):
    return m["name"] if not m.get("field") else f"{m['name']}_{m['field']}"


def run_query(df: pd.DataFrame, spec: dict) -> dict:
    d = _select(df, spec.get("dataset"))
    for f in spec.get("filters") or []:
        d = d[_mask(d, f)]
    base_total = len(d)
    groups = list(spec.get("group_by") or [])
    metrics = spec.get("metrics") or [{"name": "count"}]
    for g in groups:
        _col(g)
    d = d.copy()
    gcols = []
    tg = spec.get("time_grain")
    if tg:
        dt = d["date"]
        if tg == "day":
            d["period"] = dt.dt.strftime("%Y-%m-%d")
        elif tg == "week":
            d["period"] = (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
        elif tg == "month":
            d["period"] = dt.dt.strftime("%Y-%m")
        else:
            raise QueryError("time_grain must be day, week or month")
        gcols.append("period")
    for g in groups:
        c = _col(g)
        if FIELDS[g] == "list":
            d = d.explode(c)
            d[c] = d[c].fillna("(none)")
        if FIELDS[g] == "datetime":
            d[c + "_day"] = d[c].dt.strftime("%Y-%m-%d")
            c = c + "_day"
        gcols.append(c)
    names = [_mname(m) for m in metrics]
    rows = []
    if gcols:
        for key, sub in d.groupby(gcols, dropna=False, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            rows.append([("" if (isinstance(k, float) and np.isnan(k)) else k) for k in key]
                        + [_metric(sub, m, base_total) for m in metrics])
    else:
        rows.append([_metric(d, m, base_total) for m in metrics])
    out_cols = [("lens" if c == "lenses" else "flag" if c == "flags" else "sentiment" if c == "sentiment_label"
                 else c.replace("_day", "")) for c in gcols] + names
    res = pd.DataFrame(rows, columns=out_cols)
    sort_by = spec.get("sort_by")
    if sort_by and sort_by in res.columns:
        res = res.sort_values(sort_by, ascending=not spec.get("descending", True), na_position="last")
    elif "period" in res.columns:
        res = res.sort_values("period")
    elif names[0] in res.columns and gcols:
        res = res.sort_values(names[0], ascending=False, na_position="last")
    total_rows = len(res)
    limit = max(1, min(int(spec.get("limit") or 25), 200))
    res = res.head(limit)
    return {"columns": list(res.columns), "rows": _jsonable(res.values.tolist()),
            "row_count": total_rows, "truncated": total_rows > limit, "base_items": base_total}


def _jsonable(rows):
    out = []
    for r in rows:
        rr = []
        for v in r:
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = None if np.isnan(v) else round(float(v), 2)
            elif isinstance(v, pd.Timestamp):
                v = v.strftime("%Y-%m-%d")
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            rr.append(v)
        out.append(rr)
    return out


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {"type": "array", "items": {"type": "string"}, "description": "Words or phrases to look for in the text"},
        "match": {"type": "string", "enum": ["any", "all"], "description": "Default any"},
        "dataset": {"type": "string", "enum": ["coverage", "actionable", "excluded", "all"]},
        "filters": QUERY_SCHEMA["properties"]["filters"],
        "sort_by": {"type": "string", "enum": ["engagement", "reach", "date"]},
        "limit": {"type": "integer", "description": "Max posts (default 10, max 25)"},
    },
}


def search_posts(df: pd.DataFrame, spec: dict) -> dict:
    d = _select(df, spec.get("dataset"))
    for f in spec.get("filters") or []:
        d = d[_mask(d, f)]
    kws = [k for k in (spec.get("keywords") or []) if str(k).strip()]
    if kws:
        low = as_text(d["text"]).str.lower()
        ms = [low.str.contains(re.escape(str(k).lower()), regex=True, na=False) for k in kws]
        m = ms[0]
        for x in ms[1:]:
            m = (m & x) if spec.get("match") == "all" else (m | x)
        d = d[m]
    total = len(d)
    sort = spec.get("sort_by") or "engagement"
    d = d.sort_values(sort if sort in ("engagement", "reach", "date") else "engagement", ascending=False)
    limit = max(1, min(int(spec.get("limit") or 10), 25))
    posts = []
    for _, r in d.head(limit).iterrows():
        posts.append({"date": r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else None,
                      "platform": r["platform"], "author": str(r["author"]), "author_type": r.get("author_type", ""),
                      "sentiment": r["sentiment_label"], "engagement": int(r["engagement"]), "reach": int(r["reach"]),
                      "lenses": list(r["lenses"] or []), "text": str(r["text"])[:400],
                      "url": None if pd.isna(r["url"]) else str(r["url"])})
    return {"matches": total, "posts": posts}


def field_profile(df: pd.DataFrame) -> dict:
    prof = {}
    for f, kind in FIELDS.items():
        c = COLUMN_OF.get(f, f)
        if c not in df:
            continue
        if kind == "category":
            vc = df[c].astype(str).value_counts().head(15)
            prof[f] = {"type": kind, "top_values": vc.index.tolist(), "distinct": int(df[c].nunique())}
        elif kind == "list":
            vc = df[c].explode().dropna().value_counts().head(20)
            prof[f] = {"type": kind, "values": vc.index.tolist()}
        elif kind == "number":
            prof[f] = {"type": kind, "min": float(df[c].min()), "max": float(df[c].max())}
        elif kind == "datetime":
            prof[f] = {"type": kind, "min": str(df[c].min()), "max": str(df[c].max())}
        else:
            prof[f] = {"type": kind}
    return prof
