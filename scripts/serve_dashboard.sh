#!/usr/bin/env bash
# Run the ETV analytics dashboard with the reproducible project environment.
# Inputs: ETV_DASHBOARD_HOST and ETV_DASHBOARD_PORT environment variables.
# Output: a local FastAPI/Uvicorn service; no public tunnel is created.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ETV_PYTHON_BIN:-$HOME/miniconda3/envs/graphrag/bin/python}"
HOST="${ETV_DASHBOARD_HOST:-127.0.0.1}"
PORT="${ETV_DASHBOARD_PORT:-8321}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p \
  "$PROJECT_ROOT/data/interim/runtime_cache/numba" \
  "$PROJECT_ROOT/data/interim/runtime_cache/matplotlib"

export NUMBA_CACHE_DIR="$PROJECT_ROOT/data/interim/runtime_cache/numba"
export MPLCONFIGDIR="$PROJECT_ROOT/data/interim/runtime_cache/matplotlib"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m uvicorn aecsp.api.main:app \
  --app-dir src \
  --host "$HOST" \
  --port "$PORT" \
  --workers 1 \
  --proxy-headers \
  --forwarded-allow-ips="127.0.0.1"
