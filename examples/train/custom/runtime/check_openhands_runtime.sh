#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
API_KEY="${ALLHANDS_API_KEY:-${OPENHANDS_API_KEY:-sandbox-remote}}"
RUNTIME_URL="${SANDBOX_REMOTE_RUNTIME_API_URL:-http://$HOST:$PORT}"

curl -fsS \
  -H "X-API-Key: $API_KEY" \
  "$RUNTIME_URL/registry_prefix"

echo
echo "SkyRL-OpenHands runtime is reachable at $RUNTIME_URL"
