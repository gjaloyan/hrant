#!/usr/bin/env bash
# Run backend (uvicorn) and frontend (vite) together.
# Usage: ./dev.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
    PY="$ROOT/.venv/Scripts/python.exe"
fi
exec "$PY" "$ROOT/scripts/dev.py" "$@"
