#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CUSTOM_DIR/../../.." && pwd)"
OPENHANDS_REPO="${OPENHANDS_REPO:-/mnt/swe/SkyRL-OpenHands}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
API_KEY="${ALLHANDS_API_KEY:-${OPENHANDS_API_KEY:-sandbox-remote}}"
PYTHON_BIN="${PYTHON_BIN:-$OPENHANDS_REPO/.venv/bin/python}"

export OPENHANDS_API_KEY="$API_KEY"
export ALLHANDS_API_KEY="$API_KEY"
export SANDBOX_REMOTE_RUNTIME_API_URL="http://$HOST:$PORT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/swe/.cache/uv}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-/mnt/swe/.local/share/uv/tools}"
export UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-/mnt/swe/.local/bin}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/mnt/swe/.local/share/uv/python}"

if [ ! -d "$OPENHANDS_REPO" ]; then
  echo "Missing SkyRL-OpenHands repo: $OPENHANDS_REPO" >&2
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing runtime Python: $PYTHON_BIN" >&2
  echo "Create it with: uv venv --python 3.12 .venv && uv pip install -e ." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required by SkyRL-OpenHands remote runtime server." >&2
  exit 1
fi

PYTHON_RUN=("$PYTHON_BIN")
if ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then
    PYTHON_RUN=(
      sudo
      env
      "OPENHANDS_API_KEY=$OPENHANDS_API_KEY"
      "ALLHANDS_API_KEY=$ALLHANDS_API_KEY"
      "SANDBOX_REMOTE_RUNTIME_API_URL=$SANDBOX_REMOTE_RUNTIME_API_URL"
      "$PYTHON_BIN"
    )
  else
    echo "Docker is installed but not accessible by this user." >&2
    echo "Run with Docker access, or allow sudo Docker access for this command." >&2
    exit 1
  fi
fi

cd "$OPENHANDS_REPO"

echo "Starting SkyRL-OpenHands runtime at $SANDBOX_REMOTE_RUNTIME_API_URL"
echo "Use this in the SkyRL smoke environment:"
echo "  export ALLHANDS_API_KEY=$ALLHANDS_API_KEY"
echo "  export SANDBOX_REMOTE_RUNTIME_API_URL=$SANDBOX_REMOTE_RUNTIME_API_URL"

exec "${PYTHON_RUN[@]}" -m openhands.runtime.remote_runtime_server.main \
  --host "$HOST" \
  --port "$PORT"
