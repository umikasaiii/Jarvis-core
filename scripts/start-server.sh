#!/usr/bin/env bash
# Start JARVIS Core in production-like mode (no reload) (Linux/macOS).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet -r requirements.txt

if [ ! -f .env ]; then
  echo "No .env found, copying .env.example -> .env"
  cp .env.example .env
fi

exec uvicorn app.main:app --host "${SERVER_HOST:-127.0.0.1}" --port "${SERVER_PORT:-8000}" --workers 1
