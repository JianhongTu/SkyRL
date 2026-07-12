#!/bin/bash
# =============================================================================
# FIXED RECIPE — full RL training of our SFT Qwen3-30B-A3B SWE agent.
#
#   bash examples/train/multi-env/run_rl_recipe.sh
#
# One command, no args, using the validated 8xH200 GSPO/loop recipe. This is a
# HOST-side wrapper: it checks the remote sandbox and local training inputs,
# then launches inside the `skyrl-mega` container with the prebuilt RL venv.
#
# Long job: run inside tmux/screen so it survives an SSH disconnect:
#   tmux new -s rl  ->  bash examples/train/multi-env/run_rl_recipe.sh
#
# Underlying trainer flags: rl/run_skyrl_swe_30b.sh
# =============================================================================
set -euo pipefail

# --- fixed settings (validated by the two-step smoke trial) ------------------
REPO=/home/tovi/SkyRL
CONTAINER=skyrl-mega
RL_DIR="$REPO/examples/train/multi-env/rl"
ENV_FILE="$RL_DIR/.env"

MODEL=/data/tovi/exports/skyrl_sft_openhands_hf_harnessprompt/global_step_1084/policy
DATA_DIR=/home/tovi/data/r2e-all
TASK_YAML="$RL_DIR/skyrl_swe_30b.yaml"

CKPT_DIR=/data/tovi/ckpts/skyrl_rl_swe_qwen3_30b_a3b
EXPORT_DIR=/data/tovi/exports/skyrl_rl_swe_qwen3_30b_a3b_hf
RUN_NAME=skyrl_rl_swe_qwen3_30b_a3b
PROJECT_NAME=multi-env-rl
LOG_DIR=/data/tovi/logs

# --- preflight ---------------------------------------------------------------
echo "[recipe] preflight..."

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE missing" >&2
  exit 1
fi
set -a
source "$ENV_FILE"
set +a
for name in WANDB_API_KEY SANDBOX_REMOTE_RUNTIME_API_URL ALLHANDS_API_KEY; do
  if [ -z "${!name:-}" ]; then
    echo "ERROR: $name is empty in $ENV_FILE" >&2
    exit 1
  fi
done
echo "[recipe]   W&B and sandbox environment loaded"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "ERROR: container '$CONTAINER' is not running" >&2
  exit 1
fi

if ! docker exec "$CONTAINER" test -f "$MODEL/config.json"; then
  echo "ERROR: SFT model not found in container: $MODEL" >&2
  exit 1
fi
for split in train validation; do
  if ! docker exec "$CONTAINER" test -f "$DATA_DIR/$split.parquet"; then
    echo "ERROR: $DATA_DIR/$split.parquet not found in container" >&2
    exit 1
  fi
done
if ! docker exec "$CONTAINER" test -f "$TASK_YAML"; then
  echo "ERROR: task config not found in container: $TASK_YAML" >&2
  exit 1
fi
if ! docker exec "$CONTAINER" test -x "$RL_DIR/.venv/bin/python"; then
  echo "ERROR: prebuilt RL venv missing. Build it once:" >&2
  echo "  docker exec $CONTAINER bash -lc 'cd $RL_DIR && uv sync'" >&2
  exit 1
fi

# A 404 at the API root is acceptable; curl succeeds as long as the runtime is
# reachable. Authentication and task startup are handled by the rollout client.
if ! docker exec -e SANDBOX_REMOTE_RUNTIME_API_URL "$CONTAINER" bash -lc \
  'curl -sS -o /dev/null --connect-timeout 10 "$SANDBOX_REMOTE_RUNTIME_API_URL"'; then
  echo "ERROR: remote sandbox API is unreachable: $SANDBOX_REMOTE_RUNTIME_API_URL" >&2
  exit 1
fi
echo "[recipe]   remote sandbox API reachable"

BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}')
if [ "$BUSY" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  echo "ERROR: $BUSY GPU(s) already in use. Wait, or re-run with FORCE=1." >&2
  exit 1
fi
echo "[recipe]   model, data, environment, and GPUs ready"

# --- launch ------------------------------------------------------------------
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/${RUN_NAME}_${STAMP}.log"
echo "[recipe] launching. logging to: $LOG"
echo "[recipe] follow with: tail -f $LOG"

docker exec "$CONTAINER" bash -lc "
  set -euo pipefail
  cd '$RL_DIR'
  source .venv/bin/activate
  unset RAY_RUNTIME_ENV_HOOK
  export USE_PREBUILT_VENV=1
  export REPO='$REPO'
  export ENV_FILE='$ENV_FILE'
  export MODEL='$MODEL'
  export DATA_DIR='$DATA_DIR'
  export TASK_YAML='$TASK_YAML'
  export CKPT_DIR='$CKPT_DIR'
  export EXPORT_DIR='$EXPORT_DIR'
  export RUN_NAME='$RUN_NAME'
  export PROJECT_NAME='$PROJECT_NAME'
  export LOGGER=wandb
  exec bash run_skyrl_swe_30b.sh
" 2>&1 | tee "$LOG"

echo "[recipe] training exited. Checkpoints: $CKPT_DIR"
echo "[recipe] HF exports: $EXPORT_DIR"
echo "[recipe] NOTE: /data is instance-store. Copy the final export to /home/tovi to persist it."
