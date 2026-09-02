#!/usr/bin/env bash
# Run the test suite (Linux/macOS).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet -r requirements.txt

exec pytest "$@"
