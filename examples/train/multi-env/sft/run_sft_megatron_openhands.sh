#!/bin/bash
set -xeou pipefail

# SFT (Megatron) for willhx/Qwen3-30B-A3B_base_math_search on the OpenHands SWE
# split of nvidia/Nemotron-SFT-SWE-v3, in native Qwen <tool_call> format.
#
# Pipeline that produced the data (see this folder):
#   prepare_openhands.py  -> openhands.parquet        (filter to OpenHands + reasoning)
#   generate_sft.py       -> train.parquet            (messages + tools)
#   filter_by_length.py --truncate -> train.parquet   (longest assistant-ending prefix < 32k)
#
# Usage:
#   export WANDB_API_KEY=<key>            # only if logger=wandb
#   bash examples/train/multi-env/sft/run_sft_megatron_openhands.sh [extra hydra overrides...]
#   # e.g. smoke test:  NUM_STEPS=2 bash ...run_sft_megatron_openhands.sh dataset_split="train[:64]"
#
# Official Megatron container (deps -- TransformerEngine wheel etc. -- are prebuilt, so
# `uv run --isolated --extra megatron` below reuses them rather than rebuilding):
#   docker run -it --runtime=nvidia --gpus all --shm-size=64g --ipc=host \
#     -v /mnt/swe:/mnt/swe -e HF_HOME=/mnt/swe/.cache/huggingface \
#     novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8-megatron /bin/bash
#   # then inside (login shell so .bashrc sets RAY_RUNTIME_ENV_HOOK for the uv runtime env):
#   cd /mnt/swe/SkyRL && git checkout multi-env-sft
#   bash examples/train/multi-env/sft/run_sft_megatron_openhands.sh
# SkyRL auto-sets the needed Megatron env (NVTE_FUSED_ATTN=0, CUDA_DEVICE_MAX_CONNECTIONS=1).
#
# Model note: Qwen3-30B-A3B is a Mixture-of-Experts model (30B total / ~3B active,
# 128 experts). It needs real multi-GPU parallelism + (ideally) expert parallelism.
# The PARALLELISM block below is a STARTING POINT for one 8x H200 node -- tune it
# (and enable EP) for your hardware; if you OOM, raise TP/PP or enable recompute.

# --- paths (override via env) ----------------------------------------------
: "${DATA_DIR:=/mnt/swe/data/nemotron_sft_swe_v3_openhands_sft_trunc32k}"   # dir holding train.parquet
: "${CKPT_PATH:=/mnt/swe/ckpts/skyrl_sft_openhands}"                        # resumable Megatron ckpts
: "${EXPORT_PATH:=/mnt/swe/exports/skyrl_sft_openhands_hf}"                 # final HF export (for RL/vLLM)
: "${MODEL_PATH:=willhx/Qwen3-30B-A3B_base_math_search}"

# --- parallelism (TUNE to your GPUs/model) ---------------------------------
# world_size = NUM_GPUS = TP * PP * CP * DP  (DP is derived, not set). EP shards MoE experts.
# We prefer HIGH DP so the (always-on) distributed optimizer shards optimizer state widely:
#   DP = NUM_GPUS / (TP*PP*CP) = 8 / (2*1*1) = 4.
# TP=2 keeps params sharded enough for a 30B model to fit on 143GB H200s while maximizing DP.
# Qwen3-30B-A3B has 4 KV heads -> TP must divide 4 (TP<=4). To push DP to 8, add EP (e.g.
# TP=1, EP=8 shards experts) -- validate that combo with the smoke test first.
# VALIDATED fit for this 30B MoE at 32k on one 8xH200 node (TP=1 PP=1 CP=4 EP=8 ETP=1 -> DP=2):
#   EP=8   shards the 128 experts (the bulk of the 30B); EP=1 OOMs (every GPU holds all experts).
#   CP=4   shards the 32k-token ACTIVATION 4x -- required: at CP=1 the ~18.5GB activation OOMs on
#          top of the ~96GB static footprint (Megatron reserves ~25GB that expandable_segments
#          can't reclaim). CP counts toward the expert grid, so EP*ETP(=8) must divide TP*CP*DP(=8).
#   bf16 optimizer states (flags below) cut the static optimizer memory ~1/3.
# Do NOT use optimizer_cpu_offload (slow) or raise DP to 8 (OOMs). seq 32768 % (2*CP=8) == 0. OK.
: "${NUM_GPUS:=8}"
: "${TP:=1}"          # tensor model parallel (must divide the model's 4 KV heads, so <=4)
: "${PP:=1}"          # pipeline model parallel
: "${CP:=4}"          # context parallel: shards the 32k activation (required to fit; needs seq%(2*CP)==0)
: "${EP:=8}"          # expert model parallel (shards the 128 MoE experts; 8 divides 128)
: "${ETP:=1}"         # expert tensor parallel (=1 for fine-grained MoE; keeps EP*ETP dividing TP*CP*DP)

# --- MoE load-balancing aux loss -------------------------------------------
# Keeps expert routing balanced during SFT (off by default in SkyRL). Match the
# model's pretraining recipe if you can; 0 disables. Logged via get_moe_metrics.
MOE_LB_TYPE=aux_loss              # aux_loss | seq_aux_loss | global_aux_loss
: "${MOE_AUX_LOSS_COEFF:=1e-3}"   # coefficient; set 0 to turn the aux loss off

# --- batch / length --------------------------------------------------------
MAX_LENGTH=32768                  # matches the filter_by_length budget; rows are all < this
MAX_TOKENS_PER_MICROBATCH=32768   # packing bin capacity; must be a multiple of MAX_LENGTH
BATCH_SIZE=32                     # global mini-batch (sequences); tune to throughput
MICRO_BSZ_PER_GPU=1               # with packing, the token budget above governs the micro size

# Train duration: cosine (any decaying scheduler) needs the real step HORIZON, so we pass
# num_steps -- with num_epochs the scheduler's num_steps is None and the LR would not decay.
# Derive it from the row count so it tracks BATCH_SIZE/NUM_EPOCHS; set NUM_STEPS to override.
# NOTE: Megatron's OptimizerParamScheduler asserts num_warmup_steps < num_steps (warmup must fit
# inside the decay horizon). The full run (thousands of steps) is fine, but a tiny smoke NUM_STEPS
# must also lower the warmup, e.g.  NUM_STEPS=4 ... optimizer_config.num_warmup_steps=1.
: "${DATASET_ROWS:=34685}"        # rows in DATA_DIR/train.parquet
: "${NUM_EPOCHS:=1}"             # typical SFT 2-4 epochs
: "${NUM_STEPS:=}"              # set NUM_STEPS=<N> to override directly (e.g. smoke test)
: "${WARMUP_STEPS:=100}"         # override for shorter runs so warmup scales with the horizon
if [ -z "${NUM_STEPS}" ]; then NUM_STEPS=$(( ((DATASET_ROWS + BATCH_SIZE - 1) / BATCH_SIZE) * NUM_EPOCHS )); fi
DURATION="num_steps=${NUM_STEPS}"

# Checkpointing: like ALL reference SkyRL Megatron SFT scripts we use ckpt_interval=0 (the resumable
# dist-checkpoint deadlocks under Ray+EP in this stack -- AsyncCallsQueue.close() hangs) and save the
# model via HF export instead (hf_save_interval -> export_path/global_step_N, a loadable HF model for
# the RL/vLLM phase). Each HF save is ~48s / ~57GB, so save a few times over the run (default ~3x).
: "${HF_SAVE_INTERVAL:=$(( (NUM_STEPS + 2) / 3 ))}"   # 0 = save only at the end

: "${LOGGER:=console}"            # set LOGGER=wandb (and WANDB_API_KEY) to log to wandb

# Guard against silently clobbering a previous run's HF exports. With no resumable checkpoint
# (ckpt_interval=0), each run re-writes EXPORT_PATH/global_step_N with the SAME step numbers, so a
# re-run into the same dir overwrites earlier exports. Refuse unless the operator picks a fresh
# EXPORT_PATH or explicitly opts in with FORCE_EXPORT=1.
if [ -z "${FORCE_EXPORT:-}" ] && compgen -G "${EXPORT_PATH}/global_step_*" > /dev/null 2>&1; then
    echo "ERROR: ${EXPORT_PATH} already holds global_step_* exports from a previous run." >&2
    echo "       Set EXPORT_PATH=<new dir> for a fresh run, or FORCE_EXPORT=1 to overwrite." >&2
    exit 1
fi

# Launcher. Default: `uv run --isolated` builds a throwaway venv PER invocation -- and Ray's
# RAY_RUNTIME_ENV_HOOK replays that for every worker, so on a high-core box (e.g. 192 CPUs ->
# ~192 workers) hundreds of venvs are assembled concurrently and contend on the uv cache lock,
# stalling startup past the process-group barrier (DistStoreError). Set USE_PREBUILT_VENV=1 after
#   uv sync --extra megatron           # build ONE .venv
#   source .venv/bin/activate && unset RAY_RUNTIME_ENV_HOOK
# so all workers inherit the single prebuilt venv (workers use the driver's sys.executable). No
# per-worker builds, deterministic fast startup regardless of core count.
if [ "${USE_PREBUILT_VENV:-0}" = "1" ]; then
    LAUNCH=(python -m skyrl.train.main_sft)
else
    LAUNCH=(uv run --isolated --extra megatron --python 3.12 python -m skyrl.train.main_sft)
fi
"${LAUNCH[@]}" \
    strategy=megatron \
    model.path="$MODEL_PATH" \
    dataset_name="$DATA_DIR" \
    dataset_split="train" \
    messages_key=messages \
    tools_key=tools \
    system_key="" \
    train_on_what="all_assistant_messages" \
    max_length=$MAX_LENGTH \
    $DURATION \
    batch_size=$BATCH_SIZE \
    micro_train_batch_size_per_gpu=$MICRO_BSZ_PER_GPU \
    remove_microbatch_padding=true \
    use_sequence_packing=true \
    max_tokens_per_microbatch=$MAX_TOKENS_PER_MICROBATCH \
    seed=42 \
    optimizer_config.lr=1e-5 \
    optimizer_config.weight_decay=1e-2 \
    optimizer_config.max_grad_norm=1.0 \
    optimizer_config.num_warmup_steps=$WARMUP_STEPS \
    optimizer_config.scheduler=cosine \
    optimizer_config.min_lr=1e-6 \
    placement.num_nodes=1 \
    placement.num_gpus_per_node=$NUM_GPUS \
    megatron_config.tensor_model_parallel_size=$TP \
    megatron_config.pipeline_model_parallel_size=$PP \
    megatron_config.context_parallel_size=$CP \
    megatron_config.expert_model_parallel_size=$EP \
    megatron_config.expert_tensor_parallel_size=$ETP \
    megatron_config.moe_grouped_gemm=true \
    megatron_config.moe_router_load_balancing_type=$MOE_LB_TYPE \
    megatron_config.moe_aux_loss_coeff=$MOE_AUX_LOSS_COEFF \
    megatron_config.ddp_config.overlap_grad_reduce=false \
    megatron_config.ddp_config.overlap_param_gather=false \
    megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=true \
    megatron_config.optimizer_config_kwargs.exp_avg_dtype=bf16 \
    megatron_config.optimizer_config_kwargs.exp_avg_sq_dtype=bf16 \
    logger="$LOGGER" \
    project_name=skyrl_sft \
    run_name=skyrl_sft_openhands_qwen3_30b_a3b \
    ckpt_path="$CKPT_PATH" \
    ckpt_interval=0 \
    resume_from="" \
    hf_save_interval=$HF_SAVE_INTERVAL \
    export_path="$EXPORT_PATH" \
    "$@"

# Notes:
# - system_key="" because each row already carries Nemotron's per-row system prompt as
#   messages[0] (there is no separate `system` column). tools_key=tools forwards the 4
#   OpenHands schemas so the prompt renders <tools>...</tools>, matching the rollout agent.
# - train_on_what=all_assistant_messages: supervise every assistant turn's <think>+tool_call
#   (the agent sees prior thinking at rollout). See filter_by_length.py for why length<32k.
# - overlap_grad_reduce/overlap_param_gather are OFF (not the Megatron default true): with MoE
#   expert parallelism a batch can route ZERO tokens to some expert, so that grad-reduce bucket's
#   backward hook never fires and finalize_model_grads asserts "Communication call has not been
#   issued for this bucket" -- data-dependent, so it survives many steps then crashes (seen at
#   step 109). OFF makes finish_grad_sync issue every bucket's all-reduce synchronously. At DP=2
#   the throughput cost is negligible. Do NOT re-enable for this MoE recipe.
# - Saves HF snapshots to EXPORT_PATH/global_step_N every HF_SAVE_INTERVAL steps (loadable HF model
#   for the RL/vLLM phase). ckpt_interval=0 + resume_from="" (no resumable dist-checkpoint: it hangs
#   under Ray+EP; this matches every reference SkyRL Megatron SFT script). NOTE: with no checkpoint,
#   a crash/preemption restarts from scratch -- there is no resume.
# - Re-running clobbers same-numbered exports, so the launcher ABORTS if EXPORT_PATH already holds
#   global_step_* dirs. Pass a fresh EXPORT_PATH=<dir> for each run, or FORCE_EXPORT=1 to overwrite.
# - Teardown: with CP/EP the process HANGS at exit after the final HF save (the model IS written
#   first). Just kill it -- run in a container started with `--init` so zombies are reaped and the
#   GPUs free cleanly. Do NOT kill mid-save (leaves stuck CUDA contexts).
# - Run inside the prebuilt venv to avoid per-worker uv builds stalling startup on high-core hosts:
#     uv sync --extra megatron && source .venv/bin/activate && unset RAY_RUNTIME_ENV_HOOK
#     USE_PREBUILT_VENV=1 bash run_sft_megatron_openhands.sh
#   and cap Ray's worker pool on many-core boxes:  docker run ... --cpus=32 ...
# - Smoke test first:  NUM_STEPS=4 bash run_sft_megatron_openhands.sh dataset_split="train[:64]" \
#       optimizer_config.num_warmup_steps=1   # warmup must be < num_steps (Megatron asserts this)
