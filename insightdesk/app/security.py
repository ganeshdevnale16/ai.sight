"""Encryption for API keys at rest (Fernet / AES-128 + HMAC)."""
import base64
import hashlib
import os
import secrets

from cryptography.fernet import Fernet, InvalidToken

from .config import DATA_DIR

_fernet = None


def _derive_key() -> bytes:
    secret = os.getenv("APP_SECRET_KEY")
    if not secret:
        # Fallback for local use: a random secret kept next to the database.
        # On a server, ALWAYS set APP_SECRET_KEY so keys survive redeploys.
        p = DATA_DIR / ".secret_key"
        if p.exists():
            secret = p.read_text().strip()
        else:
            secret = secrets.token_urlsafe(48)
            p.write_text(secret)
            try:
                os.chmod(p, 0o600)
            except OSError:
                pass
    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())


def fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        _fernet = Fernet(_derive_key())
    return _fernet


def encrypt(value: str) -> str:
    return fernet().encrypt(value.encode()).decode()


def decrypt(value: str) -> str:
    try:
        return fernet().decrypt(value.encode()).decode()
    except InvalidToken:
        return ""


def mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 10:
        return "••••"
    return f"{value[:4]}••••{value[-4:]}"


# ---------------- login sessions (signed, expiring cookie) ----------------
import hmac
import time


def _session_key() -> bytes:
    return hashlib.sha256(_derive_key() + b"|session").digest()


def make_session(hours: int) -> str:
    exp = int(time.time() + hours * 3600)
    mac = hmac.new(_session_key(), str(exp).encode(), hashlib.sha256).hexdigest()
    return f"{exp}.{mac}"


def valid_session(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    exp, mac = token.split(".", 1)
    if not exp.isdigit() or int(exp) < time.time():
        return False
    good = hmac.new(_session_key(), exp.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(good, mac)


def check_password(given: str, expected: str) -> bool:
    return hmac.compare_digest((given or "").encode(), (expected or "").encode())
