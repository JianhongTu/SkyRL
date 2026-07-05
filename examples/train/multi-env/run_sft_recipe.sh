#!/bin/bash
# =============================================================================
# FIXED RECIPE — full SFT of Qwen3-30B-A3B on the OpenHands SWE split.
#
#   bash examples/train/multi-env/run_sft_recipe.sh
#
# One command, no args, everything hardcoded to the validated config. This is a
# HOST-side wrapper: it loads the wandb key from the gitignored .env, then drives
# the training inside the `skyrl-mega` container (prebuilt venv, /data/tovi output).
#
# Long job (~15h for 1 epoch). Run it inside tmux/screen so it survives an SSH
# disconnect:   tmux new -s sft  ->  bash examples/train/multi-env/run_sft_recipe.sh
#
# Underlying launcher (all the Megatron/MoE flags): sft/run_sft_megatron_openhands.sh
# =============================================================================
set -euo pipefail

# --- fixed settings (validated; see sft/run_sft_megatron_openhands.sh) --------
REPO=/home/tovi/SkyRL
CONTAINER=skyrl-mega
ENV_FILE="$REPO/.env"

# harness-prompt run: NEW dataset (system prompt aligned to the rollout harness) and FRESH
# ckpt/export/run names so we never resume from, or overwrite, the old (pre-swap) checkpoints at
# /data/tovi/exports/skyrl_sft_openhands_hf/global_step_{362,724,1084}. (Launcher also hardcodes
# resume_from="" + ckpt_interval=0, so no resumable checkpoint is ever loaded or written.)
DATA_DIR=/home/tovi/data/nemotron_sft_swe_v3_openhands_sft_harnessprompt_trunc32k  # train.parquet (EBS, read-only)
CKPT_PATH=/data/tovi/ckpts/skyrl_sft_openhands_harnessprompt          # unused (ckpt_interval=0), fresh dir
EXPORT_PATH=/data/tovi/exports/skyrl_sft_openhands_hf_harnessprompt   # HF export -> global_step_N (for RL), fresh dir
MODEL_PATH=willhx/Qwen3-30B-A3B_base_math_search
RUN_NAME=skyrl_sft_openhands_qwen3_30b_a3b_harnessprompt
LOG_DIR=/data/tovi/logs
# Full-run duration/parallelism/optimizer/MoE flags are hardcoded in the launcher
# (TP1 PP1 CP4 EP8 ETP1, cosine LR 1e-5->1e-6, bf16 optimizer, HF export ~3x).

# --- preflight (fail fast with a clear reason) -------------------------------
echo "[recipe] preflight..."

# 1. wandb key from the gitignored .env (host runs as tovi/1007 and can read 600)
if [ ! -f "$ENV_FILE" ]; then echo "ERROR: $ENV_FILE missing (need WANDB_API_KEY)"; exit 1; fi
set -a; source "$ENV_FILE"; set +a
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY empty in $ENV_FILE. Get it from https://wandb.ai/authorize"; exit 1
fi
echo "[recipe]   wandb key loaded (entity: tovi)"

# 2. container up
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "ERROR: container '$CONTAINER' not running. See sft/container_env.sh for the docker run cmd."; exit 1
fi

# 3. dataset present
if ! docker exec "$CONTAINER" test -f "$DATA_DIR/train.parquet"; then
  echo "ERROR: $DATA_DIR/train.parquet not found in container"; exit 1
fi

# 4. prebuilt venv present (avoids per-worker uv builds stalling startup)
if ! docker exec "$CONTAINER" test -x "$REPO/.venv/bin/python"; then
  echo "ERROR: prebuilt venv missing. Build it once:"
  echo "  docker exec $CONTAINER bash -lc 'cd $REPO && uv sync --extra megatron'"; exit 1
fi

# 5. GPUs idle (a stray run would OOM). Override with FORCE=1 if you know it's fine.
BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}')
if [ "$BUSY" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  echo "ERROR: $BUSY GPU(s) already in use. Wait for them to free, or re-run with FORCE=1."; exit 1
fi

# --- launch ------------------------------------------------------------------
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/${RUN_NAME}_${STAMP}.log"
echo "[recipe] launching. logging to: $LOG"
echo "[recipe] follow with:  tail -f $LOG"

# -e WANDB_API_KEY forwards the key from THIS shell's env (loaded above) without
# putting the secret on the container command line / in history / docker inspect.
docker exec -e WANDB_API_KEY "$CONTAINER" bash -lc "
  set -euo pipefail
  cd $REPO
  source .venv/bin/activate          # single prebuilt venv for driver + all workers
  unset RAY_RUNTIME_ENV_HOOK          # do NOT replay uv per worker (would thrash the cache lock)
  export USE_PREBUILT_VENV=1
  export DATA_DIR='$DATA_DIR'
  export CKPT_PATH='$CKPT_PATH'
  export EXPORT_PATH='$EXPORT_PATH'
  export MODEL_PATH='$MODEL_PATH'
  export LOGGER=wandb
  exec bash examples/train/multi-env/sft/run_sft_megatron_openhands.sh run_name=$RUN_NAME
" 2>&1 | tee "$LOG"

echo "[recipe] training process exited. Final HF model: $EXPORT_PATH/global_step_<N>"
echo "[recipe] NOTE: /data is instance-store (wiped on EC2 stop/start). Copy the final"
echo "[recipe]       export to EBS to persist:  cp -r $EXPORT_PATH/global_step_<N> /home/tovi/exports/"
