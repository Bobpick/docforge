#!/usr/bin/env bash
# Launch DocForge Streamlit UI (Ubuntu launcher / dock icon)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/streamlit" ]]; then
  STREAMLIT="$ROOT/.venv/bin/streamlit"
  PYTHON="$ROOT/.venv/bin/python"
elif command -v streamlit >/dev/null 2>&1; then
  STREAMLIT="streamlit"
  PYTHON="python3"
else
  zenity --error --text="DocForge: streamlit not found. Run:\n  cd $ROOT && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" 2>/dev/null \
    || notify-send "DocForge" "streamlit not found — install requirements in .venv" 2>/dev/null \
    || echo "DocForge: streamlit not found" >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Prefer a free port; default 8501
PORT="${DOCFORGE_PORT:-8501}"

# Open browser after short delay if xdg-open exists
(
  sleep 2
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://127.0.0.1:${PORT}" >/dev/null 2>&1 || true
  fi
) &

exec "$STREAMLIT" run "$ROOT/app.py" \
  --server.port="$PORT" \
  --server.address=127.0.0.1 \
  --browser.gatherUsageStats=false \
  --server.headless=true
