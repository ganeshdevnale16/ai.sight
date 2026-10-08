"""SQLite persistence: settings (encrypted secrets), datasets, reports, templates, chat history."""
import json
import sqlite3
import threading
import time
import uuid

from .config import DB_PATH
from .security import decrypt, encrypt, mask

_lock = threading.RLock()
SECRET_KEYS = {"anthropic_api_key", "azure_openai_api_key"}

SCHEMA = """
create table if not exists settings(k text primary key, v text, secret integer default 0);
create table if not exists datasets(id text primary key, name text, filename text, created_at real,
    status text, config text, meta text);
create table if not exists reports(id text primary key, dataset_id text, title text, kind text,
    template_id text, provider text, created_at real, spec text, html text, quality text);
create table if not exists templates(id text primary key, name text, source_filename text,
    created_at real, blueprint text);
create table if not exists chats(id integer primary key autoincrement, dataset_id text, role text,
    content text, evidence text, provider text, created_at real);
"""


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _lock, _conn() as c:
        c.executescript(SCHEMA)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------- settings ----------------
def set_setting(key: str, value):
    secret = key in SECRET_KEYS
    stored = encrypt(value) if (secret and value) else (value if isinstance(value, str) else json.dumps(value))
    with _lock, _conn() as c:
        c.execute("insert or replace into settings(k, v, secret) values(?,?,?)", (key, stored, int(secret)))


def delete_setting(key: str):
    with _lock, _conn() as c:
        c.execute("delete from settings where k=?", (key,))


def get_settings(include_secrets: bool = False) -> dict:
    with _lock, _conn() as c:
        rows = c.execute("select k, v, secret from settings").fetchall()
    out = {}
    for r in rows:
        if r["secret"]:
            out[r["k"]] = decrypt(r["v"]) if include_secrets else mask(decrypt(r["v"]))
        else:
            v = r["v"]
            try:
                out[r["k"]] = json.loads(v) if v and v[0] in "[{" else v
            except (ValueError, TypeError):
                out[r["k"]] = v
    return out


# ---------------- datasets ----------------
def create_dataset(name, filename, config):
    did = new_id()
    with _lock, _conn() as c:
        c.execute("insert into datasets values(?,?,?,?,?,?,?)",
                  (did, name, filename, time.time(), "queued", json.dumps(config), json.dumps({})))
    return did


def update_dataset(did, **fields):
    sets, vals = [], []
    for k, v in fields.items():
        sets.append(f"{k}=?")
        vals.append(json.dumps(v) if k in ("config", "meta") else v)
    with _lock, _conn() as c:
        c.execute(f"update datasets set {', '.join(sets)} where id=?", (*vals, did))


def _ds(row):
    if not row:
        return None
    d = dict(row)
    d["config"] = json.loads(d["config"] or "{}")
    d["meta"] = json.loads(d["meta"] or "{}")
    return d


def get_dataset(did):
    with _lock, _conn() as c:
        return _ds(c.execute("select * from datasets where id=?", (did,)).fetchone())


def list_datasets():
    with _lock, _conn() as c:
        return [_ds(r) for r in c.execute("select * from datasets order by created_at desc").fetchall()]


def delete_dataset(did):
    with _lock, _conn() as c:
        c.execute("delete from datasets where id=?", (did,))
        c.execute("delete from reports where dataset_id=?", (did,))
        c.execute("delete from chats where dataset_id=?", (did,))


# ---------------- reports ----------------
def save_report(dataset_id, title, kind, template_id, provider, spec, html, quality):
    rid = new_id()
    with _lock, _conn() as c:
        c.execute("insert into reports values(?,?,?,?,?,?,?,?,?,?)",
                  (rid, dataset_id, title, kind, template_id, provider, time.time(),
                   json.dumps(spec), html, json.dumps(quality)))
    return rid


def list_reports(dataset_id=None):
    q = "select id, dataset_id, title, kind, template_id, provider, created_at, quality from reports"
    args = ()
    if dataset_id:
        q += " where dataset_id=?"
        args = (dataset_id,)
    with _lock, _conn() as c:
        rows = c.execute(q + " order by created_at desc", args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["quality"] = json.loads(d["quality"] or "{}")
        out.append(d)
    return out


def get_report(rid):
    with _lock, _conn() as c:
        r = c.execute("select * from reports where id=?", (rid,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["spec"] = json.loads(d["spec"] or "{}")
    d["quality"] = json.loads(d["quality"] or "{}")
    return d


def delete_report(rid):
    with _lock, _conn() as c:
        c.execute("delete from reports where id=?", (rid,))


# ---------------- templates ----------------
def save_template(name, source_filename, blueprint, tid=None):
    tid = tid or new_id()
    with _lock, _conn() as c:
        c.execute("insert or replace into templates values(?,?,?,?,?)",
                  (tid, name, source_filename, time.time(), json.dumps(blueprint)))
    return tid


def list_templates():
    with _lock, _conn() as c:
        rows = c.execute("select id, name, source_filename, created_at, blueprint from templates "
                         "order by created_at desc").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        bp = json.loads(d.pop("blueprint") or "{}")
        d["section_count"] = len(bp.get("sections", []))
        d["description"] = bp.get("description", "")
        out.append(d)
    return out


def get_template(tid):
    with _lock, _conn() as c:
        r = c.execute("select * from templates where id=?", (tid,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["blueprint"] = json.loads(d["blueprint"] or "{}")
    return d


def delete_template(tid):
    with _lock, _conn() as c:
        c.execute("delete from templates where id=?", (tid,))


# ---------------- chat ----------------
def add_chat(dataset_id, role, content, evidence=None, provider=None):
    with _lock, _conn() as c:
        c.execute("insert into chats(dataset_id, role, content, evidence, provider, created_at) values(?,?,?,?,?,?)",
                  (dataset_id, role, content, json.dumps(evidence or []), provider, time.time()))


def list_chat(dataset_id, limit=200):
    with _lock, _conn() as c:
        rows = c.execute("select * from chats where dataset_id=? order by id desc limit ?",
                         (dataset_id, limit)).fetchall()
    out = []
    for r in reversed(rows):
        d = dict(r)
        d["evidence"] = json.loads(d["evidence"] or "[]")
        out.append(d)
    return out


def clear_chat(dataset_id):
    with _lock, _conn() as c:
        c.execute("delete from chats where dataset_id=?", (dataset_id,))
