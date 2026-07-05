#!/bin/bash
# Shared env for the multi-env SFT run on THIS host (8x H200, no /mnt/swe).
# The scripts default their paths to /mnt/swe, which does not exist here; we
# keep the repo at /home/tovi/SkyRL and put data/ckpts/exports under /home/tovi
# (the 953GB / volume). Source this before the data scripts / run script, or
# pass these as env when invoking run_sft_megatron_openhands.sh.
export IMAGE=novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8-megatron
export CONTAINER_NAME=skyrl-mega
export HF_HOME=/home/tovi/.cache/huggingface

# --- how the container was launched on this host -----------------------------
# GPU passthrough needs nvidia-container-toolkit (installed) -> use `--gpus all`
# (the `nvidia` runtime is now registered). Container runs as user `ray` (uid 1000);
# host /home/tovi is uid 1007, so we chmod'd o+x on /home/tovi and o+rwX on the
# repo + data/ckpts/exports so `ray` can traverse/write. RAY_RUNTIME_ENV_HOOK lives
# in ray's interactive .bashrc, so we bake it into the container env for exec use.
#
# --init  -> tini reaps zombie GPU workers (a killed worker otherwise leaks CUDA
#            contexts -> GPUs wedge -> reboot). --cpus=32 caps Ray workers so the
#            uv cache lock does not thrash (192 CPUs -> DistStoreError init timeout).
# -v /data -> the 27TB instance-store RAID0 (md0); ckpts/exports live under
#            /data/tovi so the 953GB EBS root (/) is not overloaded. NOTE: /data is
#            INSTANCE-STORE -> wiped on stop/start (survives reboot). Copy the final
#            HF export back to /home/tovi (EBS) or S3 to persist it.
#
#   docker run -d --name skyrl-mega --init --gpus all --ipc=host \
#     --shm-size=64g --cpus=32 \
#     -v /home/tovi:/home/tovi -v /data:/data \
#     -e HF_HOME=/home/tovi/.cache/huggingface \
#     -e RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook \
#     $IMAGE sleep infinity
#   docker exec skyrl-mega bash -lc 'git config --global --add safe.directory /home/tovi/SkyRL'
#
# Drive it with:  docker exec skyrl-mega bash -lc '<cmd>'

# run_sft_megatron_openhands.sh overrides.
# Checkpoints/exports go to /data/tovi (instance-store RAID0, 27TB, fast NVMe) so
# the EBS root drive is not overloaded. DATA_DIR stays on /home/tovi (EBS) - the
# dataset is small and read-only during training.
#
# These are the HARNESS-PROMPT run: the dataset whose messages[0] was swapped to the
# rollout harness's 386-char system prompt, with FRESH ckpt/export names. They MUST match
# run_sft_recipe.sh so the two entry points agree; the "_harnessprompt" suffix keeps the
# pre-swap dataset and old exports (/data/tovi/exports/skyrl_sft_openhands_hf/global_step_*)
# intact and prevents resume-from / overwrite of them.
export DATA_DIR=/home/tovi/data/nemotron_sft_swe_v3_openhands_sft_harnessprompt_trunc32k
export CKPT_PATH=/data/tovi/ckpts/skyrl_sft_openhands_harnessprompt
export EXPORT_PATH=/data/tovi/exports/skyrl_sft_openhands_hf_harnessprompt
export MODEL_PATH=willhx/Qwen3-30B-A3B_base_math_search

# intermediate data-pipeline dirs (stages 1-3). OH_RAW/OH_FILTERED are shared (the
# harness-prompt pipeline reuses the same filtered openhands.parquet); OH_SFT is the
# post-swap generate_sft output that feeds filter_by_length -> DATA_DIR.
export OH_RAW=/home/tovi/data/nemotron_sft_swe_v3_raw
export OH_FILTERED=/home/tovi/data/nemotron_sft_swe_v3_openhands
export OH_SFT=/home/tovi/data/nemotron_sft_swe_v3_openhands_sft_harnessprompt
