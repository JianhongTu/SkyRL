#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENHANDS_ROOT="${OPENHANDS_ROOT:-/mnt/swe/SkyRL-OpenHands}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

export OPENHANDS_API_KEY="${OPENHANDS_API_KEY:-sandbox-remote}"
export ALLHANDS_API_KEY="${ALLHANDS_API_KEY:-$OPENHANDS_API_KEY}"
export SANDBOX_REMOTE_RUNTIME_API_URL="${SANDBOX_REMOTE_RUNTIME_API_URL:-http://127.0.0.1:$PORT}"
export PATH="$SCRIPT_DIR/bin:$PATH"

exec sudo env \
  "PATH=$PATH" \
  "OPENHANDS_API_KEY=$OPENHANDS_API_KEY" \
  "ALLHANDS_API_KEY=$ALLHANDS_API_KEY" \
  "SANDBOX_REMOTE_RUNTIME_API_URL=$SANDBOX_REMOTE_RUNTIME_API_URL" \
  "$OPENHANDS_ROOT/.venv/bin/python" \
  -m openhands.runtime.remote_runtime_server.main \
  --host "$HOST" \
  --port "$PORT"
