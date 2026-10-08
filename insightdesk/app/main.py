"""HTTP API + static front end."""
from __future__ import annotations

import io
import json
import logging
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import chat, pipeline, query_engine, storage
from .config import (ADMIN_TOKEN, APP_PASSWORD, EPHEMERAL_STORAGE, SESSION_HOURS, BASE_DIR, DEFAULT_AZURE_API_VERSION, DEFAULT_AZURE_DEPLOYMENT,
                     DEFAULT_AZURE_ENDPOINT, DEFAULT_CLAUDE_MODEL, MAX_UPLOAD_MB, UPLOAD_DIR)
from .llm import LLMError, default_provider, get_provider, provider_status
from .security import check_password, make_session, valid_session
from .taxonomy import DEFAULT_LENSES

log = logging.getLogger("insightdesk")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="InsightDesk", docs_url="/api/docs", redoc_url=None)
storage.init()

DATA_EXT = {".csv", ".xlsx", ".xlsm", ".xls", ".json", ".txt", ".md", ".tsv"}
SAMPLE_EXT = {".pdf", ".pptx", ".potx", ".docx", ".dotx", ".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".txt", ".md"}


# ------------------------------------------------------------------ auth
COOKIE = "insightdesk_session"
_fails: dict = {}


def require_access(request: Request, x_access_token: str | None = Header(default=None)):
    """Signed session cookie (browser) or X-Access-Token header = password (scripts)."""
    if not APP_PASSWORD:
        return
    if x_access_token and check_password(x_access_token, APP_PASSWORD):
        return
    if valid_session(request.cookies.get(COOKIE)):
        return
    raise HTTPException(401, "Please sign in")


@app.get("/api/session")
def session_state(request: Request):
    return {"auth_required": bool(APP_PASSWORD), "ephemeral": EPHEMERAL_STORAGE, "max_upload_mb": MAX_UPLOAD_MB,
            "signed_in": (not APP_PASSWORD) or valid_session(request.cookies.get(COOKIE))}


@app.post("/api/login")
async def login(request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    now = time.time()
    recent = [t for t in _fails.get(ip, []) if now - t < 300]
    if len(recent) >= 5:
        raise HTTPException(429, "Too many wrong attempts. Wait 5 minutes and try again.")
    body = await request.json()
    if not APP_PASSWORD:
        return {"ok": True}
    if not check_password(body.get("password", ""), APP_PASSWORD):
        recent.append(now)
        _fails[ip] = recent
        time.sleep(1)
        raise HTTPException(401, "Wrong password")
    _fails.pop(ip, None)
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    resp = JSONResponse({"ok": True})
    resp.set_cookie(COOKIE, make_session(SESSION_HOURS), max_age=SESSION_HOURS * 3600, httponly=True,
                    samesite="lax", secure=secure, path="/")
    return resp


@app.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE, path="/")
    return resp


def require_admin(x_admin_token: str | None = Header(default=None), _=Depends(require_access)):
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(403, "Admin token required to change settings")


@app.exception_handler(LLMError)
async def llm_error(_: Request, exc: LLMError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


# ------------------------------------------------------------------ jobs
JOBS: dict = {}
_jlock = threading.Lock()


def start_job(kind, fn, *args, **kw):
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "kind": kind, "status": "running", "progress": [], "result": None, "error": None,
           "started": time.time()}
    with _jlock:
        JOBS[jid] = job

    def logf(msg):
        job["progress"].append({"t": round(time.time() - job["started"], 1), "msg": msg})

    def run():
        try:
            job["result"] = fn(*args, log=logf, **kw)
            job["status"] = "done"
        except Exception as e:  # noqa
            log.error("job %s failed: %s", jid, traceback.format_exc())
            job["status"], job["error"] = "error", str(e) or e.__class__.__name__
        job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return jid


@app.get("/api/jobs/{jid}", dependencies=[Depends(require_access)])
def get_job(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "Job not found (the server may have restarted)")
    return j


def _save_upload(file: UploadFile, dest: Path, allowed: set):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed))}")
    size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
    with open(dest, "wb") as out:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB")
            out.write(chunk)
    return ext


# ------------------------------------------------------------------ settings
@app.get("/api/settings", dependencies=[Depends(require_access)])
def get_settings():
    s = storage.get_settings(include_secrets=False)
    st = provider_status()
    return {
        "default_provider": default_provider(),
        "providers": {
            "claude": {"configured": st["claude"], "key_masked": s.get("anthropic_api_key", ""),
                       "key_from_env": st["claude"] and not s.get("anthropic_api_key"),
                       "model": s.get("anthropic_model") or DEFAULT_CLAUDE_MODEL},
            "azure_openai": {"configured": st["azure_openai"], "key_masked": s.get("azure_openai_api_key", ""),
                             "key_from_env": st["azure_openai"] and not s.get("azure_openai_api_key"),
                             "endpoint": s.get("azure_openai_endpoint") or DEFAULT_AZURE_ENDPOINT,
                             "deployment": s.get("azure_openai_deployment") or DEFAULT_AZURE_DEPLOYMENT,
                             "api_version": s.get("azure_openai_api_version") or DEFAULT_AZURE_API_VERSION},
        },
        "org_name": s.get("org_name") or "",
        "lens_taxonomy": s.get("lens_taxonomy") or DEFAULT_LENSES,
        "admin_required": bool(ADMIN_TOKEN),
    }


@app.post("/api/settings", dependencies=[Depends(require_admin)])
async def save_settings(request: Request):
    body = await request.json()
    plain = {"default_provider", "anthropic_model", "azure_openai_endpoint", "azure_openai_deployment",
             "azure_openai_api_version", "org_name"}
    for k in plain:
        if k in body and body[k] is not None:
            storage.set_setting(k, str(body[k]).strip())
    for k in ("anthropic_api_key", "azure_openai_api_key"):
        v = body.get(k)
        if v:  # empty means "keep existing"
            storage.set_setting(k, v.strip())
        if body.get(f"clear_{k}"):
            storage.delete_setting(k)
    if "lens_taxonomy" in body:
        lt = body["lens_taxonomy"]
        if lt in (None, "", []):
            storage.delete_setting("lens_taxonomy")
        else:
            if not isinstance(lt, list) or not all(isinstance(x, dict) and x.get("name") and isinstance(x.get("terms"), list) for x in lt):
                raise HTTPException(400, "Lens taxonomy must be a list of {name, terms[], action}")
            storage.set_setting("lens_taxonomy", lt)
    return get_settings()


@app.post("/api/settings/test", dependencies=[Depends(require_access)])
async def test_provider(request: Request):
    body = await request.json()
    p = get_provider(body.get("provider"))
    t0 = time.time()
    out = p.ping()
    return {"ok": True, "reply": out.strip()[:40], "seconds": round(time.time() - t0, 2)}


# ------------------------------------------------------------------ datasets
@app.get("/api/datasets", dependencies=[Depends(require_access)])
def list_datasets():
    return [{"id": d["id"], "name": d["name"], "filename": d["filename"], "created_at": d["created_at"],
             "status": d["status"], "rows": d["meta"].get("rows"), "period": d["meta"].get("period"),
             "primary_entity": d["meta"].get("primary_entity")} for d in storage.list_datasets()]


@app.post("/api/datasets", dependencies=[Depends(require_access)])
def upload_dataset(file: UploadFile = File(...), name: str = Form(""), config: str = Form("{}")):
    try:
        cfg = json.loads(config or "{}")
    except ValueError:
        cfg = {}
    cfg.setdefault("theta", 0.5)
    did = storage.create_dataset(name or Path(file.filename).stem, file.filename, cfg)
    ext = _save_upload(file, pipeline.ds_dir(did) / f"raw{Path(file.filename).suffix.lower()}", DATA_EXT)
    jid = start_job("analyse", pipeline.analyse, did)
    return {"dataset_id": did, "job_id": jid}


@app.get("/api/datasets/{did}", dependencies=[Depends(require_access)])
def get_dataset(did: str):
    d = storage.get_dataset(did)
    if not d:
        raise HTTPException(404, "Dataset not found")
    return d


@app.post("/api/datasets/{did}/config", dependencies=[Depends(require_access)])
async def update_config(did: str, request: Request):
    d = storage.get_dataset(did)
    if not d:
        raise HTTPException(404, "Dataset not found")
    body = await request.json()
    allowed = {"mapping_overrides", "primary_entity", "brand_terms", "exclude_terms", "owned_handles", "frame_terms",
               "theta", "ai_enrich", "ai_sample", "provider", "bot_min_authors"}
    cfg = {**d["config"], **{k: v for k, v in body.items() if k in allowed}}
    storage.update_dataset(did, config=cfg)
    return {"job_id": start_job("analyse", pipeline.analyse, did)}


@app.get("/api/datasets/{did}/facts", dependencies=[Depends(require_access)])
def get_facts(did: str):
    try:
        return pipeline.load_facts(did)
    except FileNotFoundError:
        raise HTTPException(404, "Analysis not available yet")


def _rows_frame(did, which, reason, q):
    df = pipeline.load_frame(did)
    if which == "excluded":
        df = df[~df["actionable"]]
        if reason:
            df = df[df["exclude_reason"] == reason]
    elif which == "actionable":
        df = df[df["actionable"]]
    elif which == "coverage":
        df = df[df["in_coverage"]]
    if q:
        df = df[df["text"].astype(str).str.contains(q, case=False, regex=False, na=False)]
    return df


@app.get("/api/datasets/{did}/rows", dependencies=[Depends(require_access)])
def get_rows(did: str, set: str = "actionable", reason: str = "", q: str = "", offset: int = 0, limit: int = 50):
    df = _rows_frame(did, set, reason, q)
    cols = ["date", "platform", "entity", "author", "author_type", "sentiment", "engagement", "reach", "lenses",
            "emotion", "actionability", "exclude_reason", "text", "url"]
    page = df.iloc[offset: offset + min(limit, 200)]
    rows = []
    for _, r in page.iterrows():
        rows.append({c: (r[c].strftime("%Y-%m-%d %H:%M") if c == "date" and hasattr(r[c], "strftime") else
                         ({1.0: "Positive", 0.0: "Neutral", -1.0: "Negative"}.get(r[c], "Unscored") if c == "sentiment" else
                          (list(r[c]) if c == "lenses" else (None if (not isinstance(r[c], (list, str)) and str(r[c]) in ("nan", "NaT", "<NA>")) else
                           (float(r[c]) if c in ("engagement", "reach", "actionability") else str(r[c]))))))
                     for c in cols if c in r})
    return {"total": int(len(df)), "rows": rows}


@app.get("/api/datasets/{did}/export", dependencies=[Depends(require_access)])
def export_rows(did: str, set: str = "actionable"):
    df = _rows_frame(did, set, "", "")
    out = df.drop(columns=[c for c in ("row_id",) if c in df]).copy()
    for c in ("lenses", "flags"):
        if c in out:
            out[c] = out[c].map(lambda L: "; ".join(L or []))
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    name = f"{storage.get_dataset(did)['name']}_{set}.csv".replace(" ", "_")
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.delete("/api/datasets/{did}", dependencies=[Depends(require_access)])
def delete_dataset(did: str):
    storage.delete_dataset(did)
    shutil.rmtree(pipeline.ds_dir(did), ignore_errors=True)
    return {"ok": True}


# ------------------------------------------------------------------ reports
@app.post("/api/reports", dependencies=[Depends(require_access)])
async def create_report(request: Request):
    body = await request.json()
    did = body.get("dataset_id")
    d = storage.get_dataset(did or "")
    if not d or d["status"] != "ready":
        raise HTTPException(400, "Pick a dataset that has finished analysing")
    if body.get("provider") not in ("claude", "azure_openai"):
        raise HTTPException(400, "Choose which AI writes the report: 'claude' or 'azure_openai'")
    get_provider(body.get("provider"))  # fail fast if that key is missing
    jid = start_job("report", pipeline.generate_report, did, body.get("provider"), body.get("template_id"),
                    body.get("focus", ""))
    return {"job_id": jid}


@app.get("/api/reports", dependencies=[Depends(require_access)])
def list_reports(dataset_id: str = ""):
    return storage.list_reports(dataset_id or None)


@app.get("/api/reports/{rid}", dependencies=[Depends(require_access)])
def get_report(rid: str):
    r = storage.get_report(rid)
    if not r:
        raise HTTPException(404, "Report not found")
    r.pop("html", None)
    return r


@app.get("/api/reports/{rid}/html", dependencies=[Depends(require_access)])
def report_html(rid: str, download: int = 0):
    r = storage.get_report(rid)
    if not r:
        raise HTTPException(404, "Report not found")
    headers = {}
    if download:
        fname = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in r["title"])[:80].strip().replace(" ", "_")
        headers["Content-Disposition"] = f'attachment; filename="{fname or "report"}.html"'
    return HTMLResponse(r["html"], headers=headers)


@app.delete("/api/reports/{rid}", dependencies=[Depends(require_access)])
def delete_report(rid: str):
    storage.delete_report(rid)
    return {"ok": True}


# ------------------------------------------------------------------ templates
@app.post("/api/templates", dependencies=[Depends(require_access)])
def upload_template(file: UploadFile = File(...), dataset_id: str = Form(...), name: str = Form(""),
                    provider: str = Form("")):
    d = storage.get_dataset(dataset_id)
    if not d or d["status"] != "ready":
        raise HTTPException(400, "Pick a dataset that has finished analysing — the template is mapped to its fields")
    if provider not in ("claude", "azure_openai"):
        raise HTTPException(400, "Choose which AI reads the sample: 'claude' or 'azure_openai'")
    get_provider(provider)
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:10]}{Path(file.filename).suffix.lower()}"
    _save_upload(file, dest, SAMPLE_EXT)
    named = dest.with_name(f"{dest.stem}_{Path(file.filename).name}")
    dest.rename(named)
    jid = start_job("template", pipeline.learn_template, named, name, dataset_id, provider)
    return {"job_id": jid}


@app.get("/api/templates", dependencies=[Depends(require_access)])
def list_templates():
    return storage.list_templates()


@app.get("/api/templates/{tid}", dependencies=[Depends(require_access)])
def get_template(tid: str):
    t = storage.get_template(tid)
    if not t:
        raise HTTPException(404, "Template not found")
    return t


@app.put("/api/templates/{tid}", dependencies=[Depends(require_access)])
async def update_template(tid: str, request: Request):
    from .reporting import sanitize_blueprint
    t = storage.get_template(tid)
    if not t:
        raise HTTPException(404, "Template not found")
    body = await request.json()
    bp = sanitize_blueprint(body.get("blueprint") or t["blueprint"])
    storage.save_template(body.get("name") or t["name"], t["source_filename"], bp, tid=tid)
    return storage.get_template(tid)


@app.delete("/api/templates/{tid}", dependencies=[Depends(require_access)])
def delete_template(tid: str):
    storage.delete_template(tid)
    return {"ok": True}


# ------------------------------------------------------------------ chat
@app.get("/api/chat/{did}", dependencies=[Depends(require_access)])
def chat_history(did: str):
    return storage.list_chat(did)


@app.delete("/api/chat/{did}", dependencies=[Depends(require_access)])
def chat_clear(did: str):
    storage.clear_chat(did)
    return {"ok": True}


@app.post("/api/chat/{did}", dependencies=[Depends(require_access)])
async def chat_ask(did: str, request: Request):
    body = await request.json()
    q = (body.get("question") or "").strip()
    if not q:
        raise HTTPException(400, "Type a question first")
    d = storage.get_dataset(did)
    if not d or d["status"] != "ready":
        raise HTTPException(400, "Pick a dataset that has finished analysing")
    import anyio
    df, facts = pipeline.load_frame(did), pipeline.load_facts(did)
    return await anyio.to_thread.run_sync(lambda: chat.answer(did, q, body.get("provider"), df, facts))


@app.get("/api/datasets/{did}/fields", dependencies=[Depends(require_access)])
def fields(did: str):
    return query_engine.field_profile(query_engine.prepare(pipeline.load_frame(did)))


# ------------------------------------------------------------------ static
@app.get("/api/health")
def health():
    return {"ok": True}


STATIC = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
