#!/usr/bin/env bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CUSTOM_DIR/../../.." && pwd)"
SKYRL_AGENT_ROOT="$REPO_ROOT/skyrl-agent"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-/mnt/shared_storage/datasets/r2e-all}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train.parquet}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/validation.parquet}"

CKPT_DIR="${CKPT_DIR:-$HOME/ckpts}"
EXPORT_DIR="${EXPORT_DIR:-$HOME/exports}"


MODEL="${MODEL:-willhx/Qwen3-30B-A3B_base_math_search}"
NNODES="${NNODES:-2}"
SP_SIZE="${SP_SIZE:-4}"
TP_SIZE="${TP_SIZE:-4}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LOGGER="${LOGGER:-wandb}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
seed="${SEED:-1}"
PROJECT_NAME="${PROJECT_NAME:-skyagent-32b-r2e-skyrl}"
RUN_NAME="${RUN_NAME:-skyagent-skyrl-32b-r2e-4500-loop-tool}"
CKPT_INTERVAL="${CKPT_INTERVAL:-2}"
HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:--1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
MAX_CKPTS_TO_KEEP="${MAX_CKPTS_TO_KEEP:-20}"
TASK_CONFIG="${TASK_CONFIG:-./examples/run_skyrl/skyrl_swe.yaml}"
TRAINER_EPOCHS="${TRAINER_EPOCHS:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8000}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-32768}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-40768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"

# export LD_LIBRARY_PATH="/opt/amazon/efa/lib:$LD_LIBRARY_PATH"

# Keep the custom task wrappers, SkyRL-Agent checkout, and current SkyRL
# checkout on PYTHONPATH. The uv project below is the repository root, so
# SkyRL's current fsdp extra owns the torch/vLLM stack instead of the older
# skyrl-agent recipe.
export PYTHONPATH="$CUSTOM_DIR:$SKYRL_AGENT_ROOT:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

UV_ENV_FILE_ARGS=()
if [ -f .env ]; then
  UV_ENV_FILE_ARGS=(--env-file .env)
fi

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

uv run --isolated --python 3.12 --project "$REPO_ROOT" "${UV_ENV_FILE_ARGS[@]}" "${UV_LOCK_ARGS[@]}" \
    --extra fsdp \
    "${SKYRL_AGENT_RUNTIME_WITH[@]}" \
    -m skyrl_agent.integrations.skyrl_train.skyrl_train_main  \
  data.train_data="['$TRAIN_DATA']" \
  data.val_data="['$VAL_DATA']" \
  trainer.algorithm.advantage_estimator="loop" \
  trainer.policy.model.path=$MODEL \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  generator.task="$TASK_CONFIG" \
  trainer.epochs="$TRAINER_EPOCHS" \
  trainer.seed=$seed \
  trainer.eval_batch_size="$EVAL_BATCH_SIZE" \
  trainer.eval_before_train=false \
  trainer.eval_interval="$EVAL_INTERVAL" \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$BATCH_SIZE \
  trainer.policy_mini_batch_size=$BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval="$CKPT_INTERVAL" \
  trainer.hf_save_interval="$HF_SAVE_INTERVAL" \
  trainer.max_ckpts_to_keep="$MAX_CKPTS_TO_KEEP" \
  trainer.max_prompt_length="$MAX_PROMPT_LENGTH" \
  generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.enable_prefix_caching=true \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.sequence_parallel_size=$SP_SIZE \
  trainer.ref.sequence_parallel_size=$SP_SIZE \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.loss_reduction="seq_mean_token_sum_norm" \
  trainer.algorithm.max_seq_len="$MAX_SEQ_LEN" \
  trainer.algorithm.grpo_norm_by_std=false \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=null \
  generator.n_samples_per_prompt="$N_SAMPLES_PER_PROMPT" \
  generator.eval_n_samples_per_prompt="$EVAL_N_SAMPLES_PER_PROMPT" \
  generator.inference_engine.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  trainer.logger="$LOGGER" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_path="$CKPT_DIR" \
  trainer.export_path="$EXPORT_DIR" \
  trainer.dump_data_batch=true \
  generator.inference_engine.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
  $@
