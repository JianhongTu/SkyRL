# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Scope: the **multi-env RL project, SWE phase**. This is the SFT stage that produces the
initialization for a later RL phase. Root repo guidance in `/CLAUDE.md` still applies (always
`uv run --isolated`; never bare `python`/`pip`; read `.claude/docs/*` before troubleshooting).

## Goal

SFT `willhx/Qwen3-30B-A3B_base_math_search` into an OpenHands SWE agent, then hand the exported
HF checkpoint to the RL phase. The model is **Qwen3-30B-A3B, a MoE** (30B total / ~3B active, 128
experts) — it needs real multi-GPU parallelism and MoE-aware config, not a dense-model recipe.

## Pipeline (run in order)

```
prepare_openhands.py   → openhands.parquet   filter Nemotron-SFT-SWE-v3 to OpenHands + reasoning
generate_sft.py        → train.parquet       attach tool schemas, clean tool-call args
filter_by_length.py    → train.parquet       fit rows under the 32k token budget (--truncate)
run_sft_megatron_openhands.sh                 Megatron SFT launcher
```

```bash
# 1. filter to OpenHands harness (tool-interface based, not prompt-string based)
uv run --isolated examples/train/multi-env/sft/prepare_openhands.py            # add --max-shards 1 --single-file to smoke test
# 2. make it trainable (tools column + arg cleaning)
uv run --isolated examples/train/multi-env/sft/generate_sft.py \
    --input ~/data/nemotron_sft_swe_v3_openhands/openhands.parquet \
    --output-dir ~/data/nemotron_sft_swe_v3_openhands_sft
# 3. fit to 32k (truncate to longest assistant-ending prefix)
uv run --isolated examples/train/multi-env/sft/filter_by_length.py --truncate \
    --input ~/data/.../train.parquet --output-dir ~/data/..._trunc32k
# 4. train (smoke test first)
NUM_STEPS=2 bash examples/train/multi-env/sft/run_sft_megatron_openhands.sh dataset_split="train[:64]"
```

Each data script writes a JSON report sidecar (`generate_sft_report.json`,
`filter_by_length_report.json`) — read it to verify drop/keep/truncate counts after a run.

## Non-obvious constraints (why the code is shaped this way)

- **Tool signature must be exact.** The rollout agent's parser HARD-ERRORS on unexpected tool-call
  args (`security_risk`, `timeout`, …) and on `str_replace_editor` edits. `generate_sft.py` strips
  non-schema args, coerces bool enum args to `"true"`/`"false"` (all OpenHands args are string-typed),
  and drops trajectories using `task_tracker` or with **>1 tool call per assistant turn** (the parser
  and the shared `id="toolu_01"` both assume exactly one). The 4 tool schemas are import-first from
  the OpenHands fork (byte-match with `_get_tools()`), falling back to checked-in `openhands_tools.json`.
- **Length filtering is trainer-exact, not a single `apply_chat_template`.** With
  `train_on_what=all_assistant_messages` the trainer encodes leading turns with `tools=`, then each
  later message via a fixed-base encoder (`skyrl/train/generators/utils.py::encode_messages_subset`)
  that renders every assistant turn as `loop.last` — injecting `<think></think>` on no-reasoning
  turns, which a one-shot render would not. `filter_by_length.py` replicates this (per-message deltas
  are additive → exact truncation) and `--verify-rows` asserts its fast path == the exact encoder.
  It vendors trainer helpers line-for-line from `skyrl/train/sft_trainer.py`; **keep those in sync.**
- **Config coupling in `run_sft_megatron_openhands.sh`:**
  - `system_key=""` — each Nemotron row carries its own system prompt as `messages[0]`; there is no
    separate `system` column. `tools_key=tools` forwards the schemas so the prompt renders `<tools>…`.
  - Cosine LR needs a real horizon → the script derives `num_steps` from `DATASET_ROWS`/`BATCH_SIZE`/
    `NUM_EPOCHS`. Using `num_epochs` instead would leave the scheduler horizon unset and the LR flat.
    **If you change the dataset, update `DATASET_ROWS`** (default 34685) or pass `NUM_STEPS=`.
  - `max_length` (32768) must match the `filter_by_length` budget; `max_tokens_per_microbatch` must
    be a multiple of it.
- **MoE specifics:** `moe_grouped_gemm=true` and the aux-loss load balancer
  (`moe_router_load_balancing_type=aux_loss`, coeff `1e-3`) are on to keep expert routing balanced
  during SFT (off by default in SkyRL). Parallelism (`TP=2 PP=1 CP=1 EP=1` → `DP=4` on one 8×H200
  node) is a **starting point** — `TP` must divide the model's 4 KV heads (TP≤4); enable `EP` to shard
  the 128 experts and push DP higher. See the `parallelism-strategies` skill before resizing.

## SFT→agent alignment: diagnostic findings (2026-07-04) & next step

A first diagnostic eval of the SFT checkpoint (`global_step_1084`) in the **real** OpenHands
mounted runtime (r2e harness, model served via vLLM) found the agent scores 0 and degenerates
(no valid tool calls; reasoning collapses into repetition/gibberish). Root causes, ranked:

**Export/config bugs — NOT SFT; fix the Megatron→HF export:**
- **Wrong `eos_token_id`.** `generation_config.json` has `151643` (`<|endoftext|>`, the pretrain eos)
  instead of the chat turn-end `<|im_end|>` (`151645`), so vLLM/HF `generate` never stop at the turn
  boundary → repetition / many tool-calls per turn. Proven NOT an SFT bug: `<|im_end|>` is
  loss-masked=1 (trained) and the model DOES emit it (stops cleanly with `stop_token_ids:[151645]`).
  Fix the export to `eos_token_id=[151645,151643]`.
- **Malformed `tokenizer_config.json`:** `extra_special_tokens` is a LIST (should be dict/absent) →
  breaks `AutoTokenizer`. Bogus `transformers_version: 5.8.0` stamped in configs. (memory:
  `sft-ckpt-eos-export-bug`.)

**Prompt/distribution mismatches — the real blocker (this is the next-step focus):**
- **System prompt mismatch (biggest lever).** SFT trained near-uniformly — **98% of 34,685 rows** —
  on the FULL OpenHands CodeAct system prompt (`"You are OpenHands agent…"`, **median 12,842 chars**,
  with `<ROLE>`/`<EFFICIENCY>`/`<FILE_SYSTEM_GUIDELINES>`/…). The eval/RL harness sends a **386-char**
  `"You are a programming agent…"` prompt — totally different, ~33× shorter. The model conditions on
  the long prompt it learned and goes off-distribution on the short one. (Corrects the earlier
  "per-row personas" note above/in memory: it is ONE dominant STATIC prompt, not varied personas.)
- **Workspace path mismatch.** SFT repos are at `/workspace/{repo}__{version}` (SWE-bench style); the
  r2e runtime puts the repo at `/testbed` → learned exploration paths miss.
- **Tool-set mismatch.** 4 trained (`execute_bash, think, finish, str_replace_editor`) vs 5 offered
  (adds `search`).

**Symptom & sampling:** on the eval prompt the model emits empty/doubled `<tool_call>` nested in
`<think>` and plans-instead-of-acting; temp=0 collapses into multilingual gibberish. temp≈0.95
removes the gibberish (Qwen3 thinking is temp-sensitive) but the malformed-tool-call behavior
persists → sampling is only a partial fix; the prompt mismatch is the root cause.

**Runtime/env findings (harness, not model):** mounted runtime (`feature/mounted-runtime-r2e`) does
NOT auto-pull base images (`/start` hard-fails if absent — pre-pull them); the prebuilt bundle was
missing `agent_skills` deps (`openai`, doc readers) → sandboxes crashed until installed into the
bundle env. The Qwen `<tool_call>` content-parser is on `main` but was removed on
`feature/mounted-runtime-r2e` (which relies on vLLM `--tool-call-parser hermes`, itself
Qwen-compatible) — parser choice was NOT the bottleneck; the model's output is malformed.

### NEXT STEP — simplify the SFT system prompt & align it with the real agent harness

The SFT baked in a 12.8k-char OpenHands system prompt the RL/eval harness never sends. Instead of
forcing the harness to replay that giant prompt, **regenerate the SFT data with the harness's actual
(short) system prompt as `messages[0]`**, and align `/workspace`→`/testbed` and the 4↔5 tool set, so
the agent sees at rollout what it saw at train time. (`system_key=""` already takes the prompt from
`messages[0]`, so this is a data-prep change in `prepare_openhands.py`/`generate_sft.py`, not a
launcher change.)

## Env / runtime

Runs in the official Megatron container (`novaskyai/skyrl-train-ray-*-megatron`) where
TransformerEngine etc. are prebuilt, so `uv run --isolated --extra megatron` reuses them. Use a
login shell so `.bashrc` sets `RAY_RUNTIME_ENV_HOOK`. SkyRL auto-sets `NVTE_FUSED_ATTN=0` and
`CUDA_DEVICE_MAX_CONNECTIONS=1`. `resume_from=latest` makes the launcher safe to re-run.
