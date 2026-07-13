#!/bin/bash
# =============================================================================
# FIXED RECIPE — two SFT epochs on intact-only OpenHands SWE trajectories.
#
#   bash examples/train/multi-env/run_sft_intact_recipe.sh
#
# This is the intact-only counterpart to run_sft_recipe.sh. It uses the 7,916
# trajectories that fit strictly below 32K without truncation and writes to
# separate checkpoint/export/run paths so the full-data run remains untouched.
#
# Long job: run inside tmux/screen so it survives an SSH disconnect:
#   tmux new -s sft-intact  ->  bash examples/train/multi-env/run_sft_intact_recipe.sh
# =============================================================================
set -euo pipefail

# --- fixed settings ----------------------------------------------------------
REPO=/home/tovi/SkyRL
CONTAINER=skyrl-mega
ENV_FILE="$REPO/.env"

DATA_DIR=/home/tovi/data/nemotron_sft_swe_v3_openhands_sft_harnessprompt_intact32k
DATASET_ROWS=7916
NUM_STEPS=496  # 2 * ceil(7916 / batch_size 32): exactly two intact-only epochs

CKPT_PATH=/data/tovi/ckpts/skyrl_sft_openhands_harnessprompt_intact
EXPORT_PATH=/data/tovi/exports/skyrl_sft_openhands_hf_harnessprompt_intact
MODEL_PATH=willhx/Qwen3-30B-A3B_base_math_search
RUN_NAME=skyrl_sft_openhands_qwen3_30b_a3b_harnessprompt_intact
LOG_DIR=/data/tovi/logs

# --- preflight ---------------------------------------------------------------
echo "[recipe] preflight..."

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE missing (need WANDB_API_KEY)"
  exit 1
fi
set -a
source "$ENV_FILE"
set +a
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY empty in $ENV_FILE. Get it from https://wandb.ai/authorize"
  exit 1
fi
echo "[recipe]   wandb key loaded (entity: tovi)"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "ERROR: container '$CONTAINER' is not running. See sft/container_env.sh for the docker run cmd."
  exit 1
fi

if ! docker exec "$CONTAINER" test -f "$DATA_DIR/train.parquet"; then
  echo "ERROR: $DATA_DIR/train.parquet not found in container"
  exit 1
fi

if ! docker exec "$CONTAINER" test -x "$REPO/.venv/bin/python"; then
  echo "ERROR: prebuilt venv missing. Build it once:"
  echo "  docker exec $CONTAINER bash -lc 'cd $REPO && uv sync --extra megatron'"
  exit 1
fi

BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}')
if [ "$BUSY" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  echo "ERROR: $BUSY GPU(s) already in use. Wait for them to free, or re-run with FORCE=1."
  exit 1
fi

# --- launch ------------------------------------------------------------------
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/${RUN_NAME}_${STAMP}.log"
echo "[recipe] launching two intact-only epochs ($DATASET_ROWS rows, $NUM_STEPS steps)."
echo "[recipe] logging to: $LOG"
echo "[recipe] follow with: tail -f $LOG"

docker exec -e WANDB_API_KEY "$CONTAINER" bash -lc "
  set -euo pipefail
  cd '$REPO'
  source .venv/bin/activate
  unset RAY_RUNTIME_ENV_HOOK
  export USE_PREBUILT_VENV=1
  export DATA_DIR='$DATA_DIR'
  export DATASET_ROWS='$DATASET_ROWS'
  export NUM_STEPS='$NUM_STEPS'
  export CKPT_PATH='$CKPT_PATH'
  export EXPORT_PATH='$EXPORT_PATH'
  export MODEL_PATH='$MODEL_PATH'
  export LOGGER=wandb
  exec bash examples/train/multi-env/sft/run_sft_megatron_openhands.sh run_name='$RUN_NAME'
" 2>&1 | tee "$LOG"

echo "[recipe] training process exited. Final HF model: $EXPORT_PATH/global_step_<N>"
echo "[recipe] NOTE: /data is instance-store. Copy the final export to persistent storage."
