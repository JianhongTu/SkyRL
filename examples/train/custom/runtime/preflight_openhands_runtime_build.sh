#!/usr/bin/env bash
set -euo pipefail

API_URL="${SANDBOX_REMOTE_RUNTIME_API_URL:-http://127.0.0.1:8000}"
API_KEY="${OPENHANDS_API_KEY:-${ALLHANDS_API_KEY:-sandbox-remote}}"
IMAGE_TAG="${IMAGE_TAG:-openhands-runtime-preflight:custom}"
PYTHON_BIN="${PYTHON:-python3}"

tmp_dir="$(mktemp -d)"
context_dir="$tmp_dir/context"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

mkdir -p "$context_dir"

cat >"$context_dir/Dockerfile" <<'DOCKERFILE'
FROM scratch
LABEL openhands.preflight="true"
DOCKERFILE

tar -C "$context_dir" -czf "$tmp_dir/context.tar.gz" .
base64 -w0 "$tmp_dir/context.tar.gz" >"$tmp_dir/context.b64"

response="$(
  curl -fsS \
    -H "X-API-Key: $API_KEY" \
    -F "context=@$tmp_dir/context.b64;filename=context.tar.gz" \
    -F "target_image=$IMAGE_TAG" \
    "$API_URL/build"
)"

build_id="$(
  RESPONSE="$response" "$PYTHON_BIN" - <<'PY'
import json
import os

print(json.loads(os.environ["RESPONSE"])["build_id"])
PY
)"

for _ in $(seq 1 60); do
  status_json="$(
    curl -fsS \
      -H "X-API-Key: $API_KEY" \
      "$API_URL/build_status?build_id=$build_id"
  )"
  status="$(
    RESPONSE="$status_json" "$PYTHON_BIN" - <<'PY'
import json
import os

print(json.loads(os.environ["RESPONSE"]).get("status", "UNKNOWN"))
PY
  )"
  echo "$status_json"
  case "$status" in
    SUCCESS)
      exit 0
      ;;
    FAILURE|INTERNAL_ERROR|TIMEOUT|CANCELLED|EXPIRED)
      exit 1
      ;;
  esac
  sleep 2
done

echo "Timed out waiting for build $build_id" >&2
exit 1
