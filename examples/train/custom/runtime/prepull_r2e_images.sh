#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CUSTOM_DIR/../../.." && pwd)"
SKYRL_AGENT_ROOT="$REPO_ROOT/skyrl-agent"

DATA_DIR="${DATA_DIR:-/mnt/swe/data/r2e-skyrl}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/train.parquet}"
IMAGES_FILE="${IMAGES_FILE:-/tmp/r2e-images.txt}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/swe/.cache/uv}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-/mnt/swe/.local/share/uv/tools}"
export UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-/mnt/swe/.local/bin}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/mnt/swe/.local/share/uv/python}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to prepull R2E images on the runtime host." >&2
  exit 1
fi

DOCKER_RUN=(docker)
if ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then
    DOCKER_RUN=(sudo docker)
  else
    echo "Docker is installed but not accessible by this user." >&2
    echo "Run with Docker access, or allow sudo Docker access for this command." >&2
    exit 1
  fi
fi

if [ ! -f "$TRAIN_DATA" ]; then
  echo "Missing train parquet: $TRAIN_DATA" >&2
  exit 1
fi

UV_PROJECT_ENVIRONMENT="$SKYRL_AGENT_ROOT/.venv" \
  /mnt/swe/.local/bin/uv run --project "$SKYRL_AGENT_ROOT" \
  python "$CUSTOM_DIR/data/list_r2e_images.py" --data "$TRAIN_DATA" > "$IMAGES_FILE"

echo "Prepulling images listed in $IMAGES_FILE"
while IFS= read -r image; do
  [ -n "$image" ] || continue
  if "${DOCKER_RUN[@]}" image inspect "$image" >/dev/null 2>&1; then
    echo "Already present: $image"
  else
    echo "Pulling: $image"
    "${DOCKER_RUN[@]}" pull "$image"
  fi
done < "$IMAGES_FILE"
