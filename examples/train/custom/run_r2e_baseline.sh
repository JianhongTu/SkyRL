#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around the self-contained custom copy of SkyRL-Agent's SWE/R2E
# SkyRL recipe.
#
# The copied recipe keeps SkyRL's original training command shape, while adding
# env overrides for dataset paths, observability names, and checkpoint cadence:
#   examples/train/custom/launch/run_skyrl_swe_custom.sh
#
# The upstream SkyRL-Agent launcher is intentionally left untouched.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CUSTOM_RECIPE="$SCRIPT_DIR/launch/run_skyrl_swe_custom.sh"

export DATA_DIR="${DATA_DIR:-$HOME/data/r2e-skyrl}"
export TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/train.parquet}"
export VAL_DATA="${VAL_DATA:-$DATA_DIR/validation.parquet}"
export PROJECT_NAME="${PROJECT_NAME:-custom-r2e-skyrl}"
export RUN_NAME="${RUN_NAME:-r2e-openhands-baseline}"
export CKPT_DIR="${CKPT_DIR:-$HOME/ckpts/custom-r2e-skyrl}"
export EXPORT_DIR="${EXPORT_DIR:-$HOME/exports/custom-r2e-skyrl}"
export CKPT_INTERVAL="${CKPT_INTERVAL:-10}"
export HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:--1}"
export MAX_CKPTS_TO_KEEP="${MAX_CKPTS_TO_KEEP:-20}"

if [ ! -f "$TRAIN_DATA" ]; then
  echo "Missing train parquet: $TRAIN_DATA" >&2
  echo "Create it with examples/train/custom/data/prepare_r2e_data.py first." >&2
  exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
  echo "Missing validation parquet: $VAL_DATA" >&2
  echo "Create it with examples/train/custom/data/prepare_r2e_data.py first." >&2
  exit 1
fi

mkdir -p "$CKPT_DIR" "$EXPORT_DIR"

exec bash "$CUSTOM_RECIPE" "$@"
