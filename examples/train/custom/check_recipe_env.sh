#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SKYRL_AGENT_ROOT="$REPO_ROOT/skyrl-agent"

cd "$REPO_ROOT"

export PYTHONPATH="$SCRIPT_DIR:$SKYRL_AGENT_ROOT:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

UV_LOCK_MODE="${UV_LOCK_MODE:-frozen}"
UV_LOCK_ARGS=()
if [ "$UV_LOCK_MODE" = "frozen" ]; then
  UV_LOCK_ARGS=(--frozen)
elif [ "$UV_LOCK_MODE" = "locked" ]; then
  UV_LOCK_ARGS=(--locked)
elif [ "$UV_LOCK_MODE" != "none" ]; then
  echo "Invalid UV_LOCK_MODE=$UV_LOCK_MODE; expected frozen, locked, or none." >&2
  exit 1
fi

SKYRL_AGENT_RUNTIME_WITH=(
  --with "openhands-ai @ git+https://github.com/NovaSky-AI/SkyRL-OpenHands.git@main"
  --with "swebench @ git+https://github.com/SWE-Gym/SWE-Bench-Fork.git"
  --with "swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git"
  --with litellm
  --with whatthepatch
  --with retry
  --with evaluate
  --with together
  --with daytona-api-client==0.20.1
)

uv run --isolated --python 3.12 --project "$REPO_ROOT" "${UV_LOCK_ARGS[@]}" \
  --extra fsdp \
  "${SKYRL_AGENT_RUNTIME_WITH[@]}" \
  python - <<'PY'
import importlib
import inspect

modules = ["skyrl", "skyrl_agent", "torch", "vllm", "ray", "openhands"]
for name in modules:
    module = importlib.import_module(name)
    print(f"{name}={getattr(module, '__version__', 'ok')} @ {getattr(module, '__file__', 'built-in')}")

from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_servers import utils
from tasks.swebench_docker_runtime import SWEBenchDockerRuntimeTask

signature = inspect.signature(InferenceEngineClient.generate)
print(f"InferenceEngineClient.generate={signature}")
print(f"resolve_policy_model_name={hasattr(utils, 'resolve_policy_model_name')}")
print(f"custom_task={SWEBenchDockerRuntimeTask.__name__}")
PY
