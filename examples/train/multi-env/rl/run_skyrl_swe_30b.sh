#!/bin/bash
# =============================================================================
# RL phase — GRPO/loop on the SWE task, initialized from OUR SFT checkpoint.
#
# Adapted from skyrl-agent/examples/run_skyrl/run_skyrl_swe.sh. The only things
# changed vs that reference are the WIRING to our model / host / data:
#   - MODEL  : base Qwen3-32B  ->  our SFT-exported Qwen3-30B-A3B checkpoint
#   - DATA   : /mnt/shared_storage r2e-all  ->  a LOCAL parquet dir
#   - PATHS  : ckpt/export under /data/tovi (this host's instance-store RAID0)
#   - SIZING : 2 nodes -> 1 node (8xH200); engines/TP re-fit to 8 GPUs
#   - TASK   : our tool-aligned yaml (rl/skyrl_swe_30b.yaml)
#
# BACKEND NOTE: the skyrl-agent `skyrl-train` extra pins skyrl[fsdp] (see
# skyrl-agent/pyproject.toml:70), so this RL path is FSDP2 — NOT the Megatron
# backend our SFT used. Switching this recipe to Megatron would require editing
# skyrl-agent/pyproject.toml (outside examples/train/multi-env), so it is left
# as FSDP2 like the reference. FSDP2 shards the 30B MoE fine for a single node.
#
# The isolated env is driven by examples/train/multi-env/rl/pyproject.toml (NOT
# skyrl-agent's — that path is currently unbuildable). This script cd's there.
#
#   bash examples/train/multi-env/rl/run_skyrl_swe_30b.sh
#
# Override any knob via env, e.g.:
#   DATA_DIR=/home/tovi/data/r2e-all MODEL=/home/tovi/exports/sft_final \
#     bash examples/train/multi-env/rl/run_skyrl_swe_30b.sh
# =============================================================================
set -euo pipefail

# The async OpenHands runner holds many sockets/files per trajectory. The host
# container defaults to a 1024 soft limit, which failed even at 32 agents during
# diagnosis; its hard limit is 524288.
NOFILE_LIMIT=${NOFILE_LIMIT:-524288}
if ! ulimit -n "$NOFILE_LIMIT"; then
  echo "ERROR: could not raise open-file limit to $NOFILE_LIMIT." >&2
  exit 1
fi

REPO=${REPO:-/home/tovi/SkyRL}
SKYRL_AGENT_DIR="$REPO/skyrl-agent"
RL_DIR="$REPO/examples/train/multi-env/rl"
# Env for the isolated run comes from OUR pyproject + this .env (secrets).
ENV_FILE=${ENV_FILE:-$RL_DIR/.env}

# --- our model: the SFT-exported HF checkpoint (hand-off from the SFT phase) ---
# Use the HARNESS-PROMPT re-SFT (skyrl_sft_openhands_hf_harnessprompt), NOT the
# original long-prompt SFT — this is the checkpoint the rollout smoke validated
# and the one aligned to the short system prompt the eval/RL harness actually
# sends. The HF weights live under the `policy/` subdir. It lives on /data
# (instance-store: wiped on EC2 stop/start) — copy to /home/tovi to persist.
MODEL=${MODEL:-/data/tovi/exports/skyrl_sft_openhands_hf_harnessprompt/global_step_1084/policy}

# --- RL dataset (SWE instances) ------------------------------------------------
# The reference used r2e-all on shared storage; this host has neither. Point
# DATA_DIR at a local parquet dir holding train.parquet / validation.parquet.
DATA_DIR=${DATA_DIR:-/home/tovi/data/r2e-all}
TRAIN_DATA="${DATA_DIR}/train.parquet"
VAL_DATA=${VAL_DATA:-"${DATA_DIR}/validation.parquet"}

# --- outputs: MUST be on /data (27TB instance-store RAID0), NOT the root/EBS -----
# EVERYTHING that writes to disk derives from these two paths, so keeping both on
# /data keeps the root drive clean:
#   ckpt_path   -> {ckpt_path}/global_step_N/policy  + /memory_snapshots
#   export_path -> HF exports + /dumped_data + /dumped_evals
# (the preflight below hard-fails if either is not under /data). Copy the final
# export to EBS (/home/tovi) to persist — /data is wiped on EC2 stop/start.
CKPT_DIR=${CKPT_DIR:-/data/tovi/ckpts/skyrl_rl_swe}
EXPORT_DIR=${EXPORT_DIR:-/data/tovi/exports/skyrl_rl_swe_hf}

# --- checkpoint / eval cadence (sized for a ~143-step, 1-epoch run) ------------
# Space these out so we don't stall training saving a 30B ckpt too often. At 143
# steps: eval every 10 -> ~14 evals; ckpt every 20 -> ~7 writes, of which only
# MAX_CKPTS=1 newest is kept on disk.
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
EVAL_BEFORE_TRAIN=${EVAL_BEFORE_TRAIN:-false}
CKPT_INTERVAL=${CKPT_INTERVAL:-20}
MAX_CKPTS=${MAX_CKPTS:-1}

# --- our tool-aligned task config ---------------------------------------------
TASK_YAML=${TASK_YAML:-$REPO/examples/train/multi-env/rl/skyrl_swe_30b.yaml}

# These must be inherited by the Ray rollout worker, where the agent and remote
# sandbox are actually constructed.
export PYTHONPATH="$RL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export SKYRL_PYTHONPATH_EXPORT=${SKYRL_PYTHONPATH_EXPORT:-1}
export SANDBOX_RUNTIME_MODE=${SANDBOX_RUNTIME_MODE:-mounted}
export SANDBOX_RUNTIME_BUNDLE_HOST_PATH=${SANDBOX_RUNTIME_BUNDLE_HOST_PATH:-/opt/openhands-runtime/current}
export SANDBOX_RUNTIME_BUNDLE_CONTAINER_PATH=${SANDBOX_RUNTIME_BUNDLE_CONTAINER_PATH:-/opt/openhands-runtime}

# --- single-node (8xH200) sizing ----------------------------------------------
NNODES=${NNODES:-1}
NUM_GPUS=${NUM_GPUS:-8}
# colocate_all: policy + inference share GPUs. num_engines * TP == NUM_GPUS.
NUM_INFERENCE_ENGINES=${NUM_INFERENCE_ENGINES:-2}
TP_SIZE=${TP_SIZE:-4}          # vLLM tensor-parallel per engine (2*4 = 8)
SP_SIZE=${SP_SIZE:-4}          # FSDP sequence-parallel for policy/ref
BATCH_SIZE=${BATCH_SIZE:-32}   # prompts/step (single node; ref used 64 on 2 nodes)
LOGGER=${LOGGER:-wandb}
INFERENCE_BACKEND=${INFERENCE_BACKEND:-vllm}
seed=${seed:-1}

RUN_NAME=${RUN_NAME:-skyrl_rl_swe_qwen3_30b_a3b}
PROJECT_NAME=${PROJECT_NAME:-multi-env-rl}

# --- policy loss: GSPO (default) ----------------------------------------------
# GSPO (arxiv 2507.18071) uses a SEQUENCE-level importance ratio instead of the
# token-level ratio of PPO/GRPO. This is the stability fix for MoE RL: token-level
# ratios are noisy for MoE because expert routing flips between the rollout policy
# and the updated policy; the sequence-level ratio averages that out (Qwen3 itself
# was trained with GSPO). These THREE settings are a matched set — do not change
# one without the others (skyrl/backends/skyrl_train/utils/ppo_utils.py:662):
#   1. policy_loss_type=gspo
#   2. loss_reduction=sequence_mean   (GSPO expects it; warns otherwise. Side effect:
#      max_seq_len is no longer needed — it was only for seq_mean_token_sum_norm.)
#   3. eps_clip_low/high ~ 3e-4 / 4e-4 — GSPO reads the SAME eps_clip_* knobs, but a
#      sequence-AVERAGED ratio needs a MUCH tighter band than token-level 0.2/0.28,
#      or it never clips (no regularization). These are the paper's values and are
#      the first thing to tune. Override via env for sweeps.
POLICY_LOSS_TYPE=${POLICY_LOSS_TYPE:-gspo}
LOSS_REDUCTION=${LOSS_REDUCTION:-sequence_mean}
EPS_CLIP_LOW=${EPS_CLIP_LOW:-3e-4}
EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-4e-4}

# --- preflight (fail fast with a clear reason) --------------------------------
if [ ! -d "$MODEL" ]; then
  echo "ERROR: SFT checkpoint not found: $MODEL" >&2
  echo "       Set MODEL=<dir with config.json + weights>. Final SFT export was" >&2
  echo "       /data/tovi/exports/skyrl_sft_openhands_hf/global_step_1084 (instance-store)." >&2
  exit 1
fi
if [ ! -f "$TRAIN_DATA" ]; then
  echo "ERROR: $TRAIN_DATA missing. RL needs a SWE-instance parquet (e.g. r2e-all)." >&2
  echo "       Set DATA_DIR=<dir with train.parquet + validation.parquet>." >&2
  exit 1
fi
if [ ! -f "$TASK_YAML" ]; then
  echo "ERROR: task yaml not found: $TASK_YAML" >&2; exit 1
fi
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: env file not found: $ENV_FILE" >&2
  echo "       Needs WANDB_API_KEY plus the SWE sandbox runtime vars (ALLHANDS_API_KEY," >&2
  echo "       SANDBOX_REMOTE_RUNTIME_API_URL) — the SWEBench task needs a remote runtime." >&2
  exit 1
fi
# Guarantee all checkpoints/exports/dumps land on /data, never the root/EBS drive.
for _d in "$CKPT_DIR" "$EXPORT_DIR"; do
  case "$_d" in
    /data/*) : ;;
    *) echo "ERROR: '$_d' is not under /data. Checkpoints/exports/dumps would land on" >&2
       echo "       the root/EBS drive. Set CKPT_DIR/EXPORT_DIR under /data/tovi." >&2
       exit 1 ;;
  esac
  if ! mkdir -p "$_d"; then
    echo "ERROR: could not create output directory: $_d" >&2
    exit 1
  fi
  if ! _write_probe=$(mktemp "$_d/.skyrl-write-test.XXXXXX"); then
    echo "ERROR: output directory is not writable: $_d" >&2
    exit 1
  fi
  rm -f "$_write_probe"
done

cd "$RL_DIR"

# Env from OUR pyproject (examples/train/multi-env/rl/pyproject.toml): local skyrl
# path, mounted-openhands fork, and the override-deps that unbreak the resolve
# (see that file + memory multi-env-rl-env-fix). Running from skyrl-agent instead
# hits the unbuildable dependency cascade. skyrl-train arrives via the pyproject's
# `skyrl-agent[skyrl-train]` dep, so no --extra flag.
#
# NO --with pins: the reference recipe's vllm==0.9.2 / torch==2.7.0 / flash-attn
# (cu12torch2.7) set is stale — current skyrl[fsdp] pins a coherent modern stack
# (vllm==0.23.0 / torch==2.11.0 / flash-attn==2.8.3 via prebuilt) and any torch-2.7
# --with conflicts with it. Let skyrl[fsdp] drive everything (as the smoke did).
#
# Launcher: default `uv run --isolated` builds a throwaway venv AND Ray's
# RAY_RUNTIME_ENV_HOOK replays it for EVERY worker — on this 192-CPU box that means
# many concurrent uv builds contending on the cache lock, stalling startup past the
# process-group barrier. Strongly prefer USE_PREBUILT_VENV=1 after building one venv:
#   cd "$RL_DIR" && uv sync && source .venv/bin/activate && unset RAY_RUNTIME_ENV_HOOK
# so every Ray worker inherits that single venv (workers use the driver's
# sys.executable). In prebuilt mode we load $ENV_FILE ourselves (no `uv --env-file`).
if [ "${USE_PREBUILT_VENV:-0}" = "1" ]; then
    set -a; . "$ENV_FILE"; set +a
    LAUNCH=(python rl_train_entry.py)
else
    LAUNCH=(uv run --isolated --env-file "$ENV_FILE" python rl_train_entry.py)
fi
"${LAUNCH[@]}" \
  data.train_data="['$TRAIN_DATA']" \
  data.val_data="['$VAL_DATA']" \
  trainer.algorithm.advantage_estimator="loop" \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  generator.task="$TASK_YAML" \
  trainer.epochs=1 \
  trainer.seed=$seed \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=$EVAL_BEFORE_TRAIN \
  trainer.eval_interval=$EVAL_INTERVAL \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$BATCH_SIZE \
  trainer.policy_mini_batch_size=$BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=$CKPT_INTERVAL \
  trainer.max_ckpts_to_keep=$MAX_CKPTS \
  trainer.max_prompt_length=${MAX_PROMPT_LEN:-30720} \
  generator.sampling_params.max_generate_length=2048 \
  generator.sampling_params.temperature=${TEMP:-0.6} \
  generator.sampling_params.top_p=${TOP_P:-0.95} \
  generator.sampling_params.top_k=-1 \
  generator.eval_sampling_params.max_generate_length=2048 \
  generator.eval_sampling_params.temperature=${EVAL_TEMP:-0.6} \
  generator.eval_sampling_params.top_p=${EVAL_TOP_P:-0.95} \
  generator.eval_sampling_params.top_k=-1 \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.enable_prefix_caching=true \
  trainer.algorithm.policy_loss_type="$POLICY_LOSS_TYPE" \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.sequence_parallel_size=$SP_SIZE \
  trainer.ref.sequence_parallel_size=$SP_SIZE \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.eps_clip_low=$EPS_CLIP_LOW \
  trainer.algorithm.eps_clip_high=$EPS_CLIP_HIGH \
  trainer.algorithm.loss_reduction="$LOSS_REDUCTION" \
  trainer.algorithm.grpo_norm_by_std=false \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=true \
  environment.env_class=null \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=1 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_path="$CKPT_DIR" \
  trainer.export_path="$EXPORT_DIR" \
  trainer.dump_data_batch=true \
  generator.inference_engine.max_num_batched_tokens=16384 \
  $@
