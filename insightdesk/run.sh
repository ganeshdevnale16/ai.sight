#!/usr/bin/env bash
# Local run: ./run.sh  then open http://localhost:8000
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
[ -f .env ] && set -a && source .env && set +a
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --reload
