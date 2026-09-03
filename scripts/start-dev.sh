#!/usr/bin/env bash
# Start JARVIS Core with auto-reload for local development (Linux/macOS).
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

# Load .env into the process environment - without this, editing .env alone
# never changes what --host/--port below bind to (the app's own Settings
# reads .env directly and logs the right value, but that's informational
# only; it doesn't control the uvicorn socket).
set -a
# shellcheck disable=SC1091
source .env
set +a

exec uvicorn app.main:app --reload --host "${SERVER_HOST:-127.0.0.1}" --port "${SERVER_PORT:-8000}"
