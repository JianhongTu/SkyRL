#!/usr/bin/env bash
# Disposable two-step smoke trial for GPU-memory and post-update weight-sync validation.
set -euo pipefail

RL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export BATCH_SIZE=${BATCH_SIZE:-8}
export MAX_TRAINING_STEPS=${MAX_TRAINING_STEPS:-2}
export EVAL_INTERVAL=${EVAL_INTERVAL:-9999}
export CKPT_INTERVAL=${CKPT_INTERVAL:-1}
export RUN_NAME=${RUN_NAME:-skyrl_rl_swe_qwen3_30b_a3b_smoke}
export CKPT_DIR=${CKPT_DIR:-/data/tovi/ckpts/skyrl_rl_swe_smoke}
export EXPORT_DIR=${EXPORT_DIR:-/data/tovi/exports/skyrl_rl_swe_smoke_hf}

exec "$RL_DIR/run_skyrl_swe_30b.sh" \
  trainer.max_training_steps="$MAX_TRAINING_STEPS" \
  "$@"
