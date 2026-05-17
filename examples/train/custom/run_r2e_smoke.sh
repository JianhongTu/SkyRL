#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CUSTOM_RECIPE="$SCRIPT_DIR/launch/run_skyrl_swe_custom.sh"

export DATA_DIR="${DATA_DIR:-/mnt/swe/data/r2e-skyrl}"
export TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/train.parquet}"
export VAL_DATA="${VAL_DATA:-$DATA_DIR/validation.parquet}"
export TASK_CONFIG="${TASK_CONFIG:-$SCRIPT_DIR/configs/skyrl_swe_smoke.yaml}"
export MODEL="${MODEL:-willhx/Qwen3-30B-A3B_base_math_search}"

export PROJECT_NAME="${PROJECT_NAME:-custom-r2e-skyrl}"
export RUN_NAME="${RUN_NAME:-r2e-openhands-smoke}"
export CKPT_DIR="${CKPT_DIR:-$HOME/ckpts/custom-r2e-skyrl-smoke}"
export EXPORT_DIR="${EXPORT_DIR:-$HOME/exports/custom-r2e-skyrl-smoke}"
export CKPT_INTERVAL="${CKPT_INTERVAL:-1}"
export HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:--1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
export MAX_CKPTS_TO_KEEP="${MAX_CKPTS_TO_KEEP:-2}"

export NNODES="${NNODES:-1}"
# Smoke defaults assume one local 8-GPU B200 node. Keep one vLLM engine and
# one rollout per prompt, but use enough prompts to satisfy SkyRL's FSDP batch
# divisibility checks with the recipe's sequence parallelism.
export NUM_GPUS="${NUM_GPUS:-8}"
export NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-1}"
export TP_SIZE="${TP_SIZE:-8}"
export SP_SIZE="${SP_SIZE:-4}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-1}"
export TRAINER_EPOCHS="${TRAINER_EPOCHS:-1}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-4096}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-12288}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
export LOGGER="${LOGGER:-wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"

ROLLOUT_GPUS=$((NUM_INFERENCE_ENGINES * TP_SIZE))
if [ "$ROLLOUT_GPUS" -ne "$NUM_GPUS" ]; then
  echo "Invalid colocated GPU layout: NUM_INFERENCE_ENGINES * TP_SIZE = $ROLLOUT_GPUS, but NUM_GPUS = $NUM_GPUS." >&2
  echo "For colocate_all=true, set rollout GPUs equal to policy GPUs, e.g. NUM_INFERENCE_ENGINES=1 TP_SIZE=8 or NUM_INFERENCE_ENGINES=2 TP_SIZE=4." >&2
  exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
  echo "Missing train parquet: $TRAIN_DATA" >&2
  exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
  echo "Missing validation parquet: $VAL_DATA" >&2
  exit 1
fi

if [ -z "${SANDBOX_REMOTE_RUNTIME_API_URL:-}" ]; then
  echo "SANDBOX_REMOTE_RUNTIME_API_URL must point at the local SkyRL-OpenHands runtime server." >&2
  exit 1
fi

mkdir -p "$CKPT_DIR" "$EXPORT_DIR"

cd "$REPO_ROOT"
exec bash "$CUSTOM_RECIPE" "$@"
