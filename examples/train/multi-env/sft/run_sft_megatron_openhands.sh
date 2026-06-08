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
NUM_GPUS=8
TP=2          # tensor model parallel (<=4 for this model's 4 KV heads)
PP=1          # pipeline model parallel
CP=1          # context parallel (raise for very long seqs if attention OOMs)
EP=1          # expert model parallel (raise to shard MoE experts; lets you lower TP for even higher DP)

# --- batch / length --------------------------------------------------------
MAX_LENGTH=32768                  # matches the filter_by_length budget; rows are all < this
MAX_TOKENS_PER_MICROBATCH=32768   # packing bin capacity; must be a multiple of MAX_LENGTH
BATCH_SIZE=32                     # global mini-batch (sequences); tune to throughput
MICRO_BSZ_PER_GPU=1               # with packing, the token budget above governs the micro size

# Train duration: by epochs unless NUM_STEPS is set (config errors if BOTH are passed).
: "${NUM_EPOCHS:=3}"              # typical SFT 2-4 epochs
: "${NUM_STEPS:=}"               # set NUM_STEPS=<N> to train by steps instead (e.g. smoke test)
if [ -n "${NUM_STEPS}" ]; then DURATION="num_steps=${NUM_STEPS}"; else DURATION="num_epochs=${NUM_EPOCHS}"; fi

: "${LOGGER:=console}"            # set LOGGER=wandb (and WANDB_API_KEY) to log to wandb

uv run --isolated --extra megatron --python 3.12 \
    python -m skyrl.train.main_sft \
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
    optimizer_config.num_warmup_steps=20 \
    optimizer_config.scheduler=constant_with_warmup \
    placement.num_nodes=1 \
    placement.num_gpus_per_node=$NUM_GPUS \
    megatron_config.tensor_model_parallel_size=$TP \
    megatron_config.pipeline_model_parallel_size=$PP \
    megatron_config.context_parallel_size=$CP \
    megatron_config.expert_model_parallel_size=$EP \
    megatron_config.ddp_config.overlap_grad_reduce=true \
    megatron_config.ddp_config.overlap_param_gather=true \
    logger="$LOGGER" \
    project_name=skyrl_sft \
    run_name=skyrl_sft_openhands_qwen3_30b_a3b \
    ckpt_path="$CKPT_PATH" \
    ckpt_interval=200 \
    max_ckpts_to_keep=1 \
    resume_from="latest" \
    hf_save_interval=0 \
    export_path="$EXPORT_PATH" \
    "$@"

# Notes:
# - system_key="" because each row already carries Nemotron's per-row system prompt as
#   messages[0] (there is no separate `system` column). tools_key=tools forwards the 4
#   OpenHands schemas so the prompt renders <tools>...</tools>, matching the rollout agent.
# - train_on_what=all_assistant_messages: supervise every assistant turn's <think>+tool_call
#   (the agent sees prior thinking at rollout). See filter_by_length.py for why length<32k.
# - resume_from="latest": safe to re-run; it picks up the newest checkpoint in CKPT_PATH.
# - To export an HF model for the RL/vLLM phase, set hf_save_interval=<steps> (writes HF
#   snapshots to EXPORT_PATH). Leave 0 for ckpt-only runs.
# - Smoke test first:  NUM_STEPS=2 bash run_sft_megatron_openhands.sh dataset_split="train[:64]"
