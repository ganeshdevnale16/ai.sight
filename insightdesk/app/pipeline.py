"""Orchestration: analyse a dataset, generate reports, learn templates."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from . import analytics, cleaning, enrich, ingest, reporting, storage, taxonomy, template_extract
from .config import DATASET_DIR
from .llm import get_provider


def ds_dir(did) -> Path:
    p = DATASET_DIR / did
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_path(did) -> Path:
    d = ds_dir(did)
    files = [f for f in d.iterdir() if f.name.startswith("raw")]
    if not files:
        raise FileNotFoundError("The original upload is missing. Upload the file again.")
    return files[0]


_cache: dict = {}


def load_frame(did) -> pd.DataFrame:
    p = ds_dir(did) / "clean.pkl"
    mt = p.stat().st_mtime
    hit = _cache.get(did)
    if hit and hit[0] == mt:
        return hit[1]
    with open(p, "rb") as f:
        df = pickle.load(f)
    _cache[did] = (mt, df)
    return df


def load_facts(did) -> dict:
    return json.loads((ds_dir(did) / "facts.json").read_text())


def lens_config():
    s = storage.get_settings()
    lenses = s.get("lens_taxonomy")
    return lenses if isinstance(lenses, list) and lenses else None


def analyse(did, log):
    ds = storage.get_dataset(did)
    cfg = ds["config"]
    storage.update_dataset(did, status="analysing")
    try:
        log("Reading file")
        raw = ingest.read_table(raw_path(did))
        if raw.empty:
            raise ValueError("The file has no rows.")
        log(f"Read {len(raw):,} rows × {raw.shape[1]} columns")
        auto = ingest.auto_map(raw)
        mapping = {**auto}
        for k, col in (cfg.get("mapping_overrides") or {}).items():
            if k in mapping:
                mapping[k] = {"column": col or None, "method": "manual" if col else None}
        df = ingest.build_canonical(raw, mapping)
        if df["text"].str.len().fillna(0).sum() == 0:
            raise ValueError("No text column was found. Pick the text column under Column mapping and re-run.")
        log("Mapped columns; tagging lenses, emotions and flags")
        lenses = lens_config()
        df = taxonomy.tag(df, lenses)
        log("Running the relevance & noise gate")
        df = cleaning.apply_noise_gate(df, cfg)
        if cfg.get("ai_enrich"):
            df = enrich.ai_enrich(df, cfg.get("provider"), int(cfg.get("ai_sample", 400)), lenses, log)
        gate = cleaning.gate_summary(df, float(cfg.get("theta", 0.5)))
        log(f"Kept {gate['actionable_items']:,} actionable of {gate['total_ingested']:,} ({gate['signal_rate_pct']}%)")
        cfg_eff = {**cfg, "lenses": lenses}
        facts = analytics.build_facts(df, cfg_eff, gate)
        df["author_type"] = analytics.author_types(df, analytics.owned_tokens_for(df, cfg) or
                                                   [str(facts["meta"]["primary_entity"]).lower().replace(" ", "")])
        drop = [c for c in ("norm_text",) if c in df]
        with open(ds_dir(did) / "clean.pkl", "wb") as f:
            pickle.dump(df.drop(columns=drop), f)
        (ds_dir(did) / "facts.json").write_text(json.dumps(facts, default=str))
        meta = {"rows": len(raw), "columns": list(map(str, raw.columns)), "mapping": mapping,
                "column_profile": ingest.column_profile(raw), "field_help": ingest.FIELD_HELP,
                "data_health": gate, "overview": facts["overview"], "primary_entity": facts["meta"]["primary_entity"],
                "entities": facts["meta"]["entities"], "period": [facts["meta"]["period_start"], facts["meta"]["period_end"]]}
        storage.update_dataset(did, status="ready", meta=meta)
        log("Analysis complete")
        return {"dataset_id": did}
    except Exception:
        storage.update_dataset(did, status="error")
        raise


def generate_report(did, provider_name, template_id, focus, log):
    df = load_frame(did)
    facts = load_facts(did)
    provider = get_provider(provider_name)
    if template_id:
        t = storage.get_template(template_id)
        if not t:
            raise ValueError("Template not found")
        bp, kind = t["blueprint"], "template"
    else:
        bp, kind = reporting.standard_blueprint(facts), "standard"
    log(f"Computing {len(bp['sections'])} sections from the data")
    sections = reporting.resolve(bp, df, facts)
    log("Writing the narrative")
    ai = reporting.write_narrative(provider, bp.get("name", "Report"), sections, facts, focus)
    quality = reporting.verify_numbers(ai, facts, sections)
    log(f"Numbers check: {quality['verified']} of {quality['checked']} figures traced to the data")
    org = storage.get_settings().get("org_name") or "Reputation Intelligence"
    spec = reporting.build_spec(bp, sections, ai, facts, org, kind)
    spec["method_text"] = reporting.method_text(facts, quality)
    html = reporting.render_html(spec)
    unavailable = [k["source_label"] or k["label"] for s in sections for k in s["kpis"] if not k["available"]]
    quality["unavailable_kpis"] = unavailable
    rid = storage.save_report(did, spec["title"], kind, template_id, provider.name,
                              {"blueprint": bp, "ai": ai}, html, quality)
    log("Report ready")
    return {"report_id": rid}


def learn_template(path: Path, name, did, provider_name, log):
    provider = get_provider(provider_name)
    log("Reading the sample report")
    extracted = template_extract.extract(path, log)
    df, facts = load_frame(did), load_facts(did)
    log("Studying its structure and KPIs, and mapping them to your data")
    bp = reporting.build_template_blueprint(provider, extracted, facts, df)
    tid = storage.save_template(name or bp.get("name") or path.stem, path.name, bp)
    n_un = sum(1 for s in bp["sections"] for k in s.get("kpis", []) if k.get("available") is False)
    log(f"Template saved: {len(bp['sections'])} sections, {n_un} KPI(s) not available in this data")
    return {"template_id": tid}
