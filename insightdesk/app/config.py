"""Runtime configuration. Everything can be overridden with environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
DATASET_DIR = DATA_DIR / "datasets"
UPLOAD_DIR = DATA_DIR / "uploads"
for _d in (DATA_DIR, DATASET_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "app.db"

# Login password for the whole app. If set, users must sign in. (APP_ACCESS_TOKEN kept as an older alias.)
APP_PASSWORD = os.getenv("APP_PASSWORD") or os.getenv("APP_ACCESS_TOKEN", "")
SESSION_HOURS = int(os.getenv("SESSION_HOURS", "12"))
# Optional. If set, saving settings (API keys) needs header X-Admin-Token as well.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))

# Model defaults (can be changed from the Settings tab)
DEFAULT_CLAUDE_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
DEFAULT_AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
DEFAULT_AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
DEFAULT_AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")

# Free hosting (e.g. Render free plan): local files vanish on restart. Shows a warning in the app.
EPHEMERAL_STORAGE = os.getenv("EPHEMERAL_STORAGE", "").lower() in ("1", "true", "yes")
