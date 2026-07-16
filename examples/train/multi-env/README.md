# Multi-Env RL

Codebase for the SWE phase of the multi-env RL project.

Pipeline: **SFT → eval (baseline) → RL**, all on one base model
**`willhx/Qwen3-30B-A3B_base_math_search`** (Qwen3-30B-A3B MoE, 30B total / ~3B active).

This README locks the **recipe** — which datasets, images, and settings each stage uses.
Operational details and gotchas live in [`CLAUDE.md`](./CLAUDE.md).

Eval and RL use **completely separate dataset/image lineages — do not mix them.**

---

## ⚠️ Open issues — RL training-loop smoke (resume 2026-07-10)

Env is fixed (modern `skyrl[fsdp]`: vllm 0.23 / torch 2.11; use `USE_PREBUILT_VENV=1` — build one
venv, `source rl/.venv/bin/activate && unset RAY_RUNTIME_ENV_HOOK` — to avoid per-worker Ray venv
rebuilds on this 192-CPU box). **Two blockers** before the loop reaches an optimizer step:

1. **Colocated engine emits garbage for the MoE — NOT a TP bug.** Every rollout: 0 `<tool_call>`,
   `"],`-repetition, `stop reason length` → all `CONTEXT_WINDOW_EXCEEDED` → trainer crash
   `NoneType has no len()` (`trainer.py:138`). **Standalone vLLM TP=1 on the same ckpt is CLEAN;
   colocated SkyRL garbles at BOTH TP=1 and TP=4** → it's the **FSDP→vLLM weight sync**, not TP
   sharding / CUDA graphs / the eos-tokenizer patch.
   **Root cause = exact match: [vLLM #42821](https://github.com/vllm-project/vllm/issues/42821) /
   [SkyRL #1680](https://github.com/NovaSky-AI/SkyRL/issues/1680).** Fused-MoE `load_weights` is
   **non-idempotent**: vLLM's FlashInfer-CUTLASS/TRTLLM MoE backend runs `swap_w13_to_w31` (swaps the
   fused `[gate;up]` halves) **once at engine init**; the RL sync is the **2nd** `load_weights` →
   writes raw `[gate;up]` into the swapped buffer → gate/up mis-slotted → gibberish from token 1.
   Standalone loads once (clean); colocated sync = 2nd load (garbage) — TP/graph-independent. **Fixed
   by SkyRL PRs #1685+#1737 and our `.venv` HAS them** (`new_inference_worker_wrap.py` +
   `layerwise_reload.py`; `update_weights_nccl` bracketed by `start/finish_weight_update` +
   `set_current_vllm_config`) → we're hitting a **residual case vs vLLM 0.23.0**. Next:
   (a) confirm vLLM 0.23.0 includes [PR #44814](https://github.com/vllm-project/vllm/pull/44814)
   (`patch_numel_loaded`); (b) confirm the nccl sync goes through the bracketed
   `NewInferenceWorkerWrap.update_weights_nccl`, not a native `/update_weights`;
   (c) **workaround — force a non-FlashInfer fused-MoE backend (Triton)** so `swap_w13_to_w31` never
   runs. (Related same-family: [verl #6847](https://github.com/verl-project/verl/issues/6847).)
   **Export ruled out (validated 2026-07-10):** the ckpt is standard HF Qwen3MoE — separate
   `gate_proj`/`up_proj` per expert (0 fused keys), all 6144 expert tensors present, shapes correct
   (`[768,2048]`/`[2048,768]`, router `[128,2048]`). No re-export needed; the swap is vLLM-internal.
2. **Mounted runtime never active in the training path.** `rl_train_entry.py` patches
   `get_default_sandbox_config_for_eval`, but the training rollout builds `SandboxConfig` elsewhere →
   per-instance `Building image: xingyaoww/runtime` (slow). Patch the training-path SandboxConfig source.

Relaunch (inside `skyrl-mega`): `USE_PREBUILT_VENV=1 NUM_INFERENCE_ENGINES=8 TP_SIZE=1
DATA_DIR=/home/tovi/data/r2e-smoke TASK_YAML=…/rl/skyrl_swe_30b_smoke.yaml LOGGER=console
bash rl/run_skyrl_swe_30b.sh`. Temp is 0.6 in the smoke yaml (temp 1.0 also degenerates, but
temp does NOT fix issue 1).

---

## SFT

- **Dataset:** `nvidia/Nemotron-SFT-SWE-v3`, filtered to **OpenHands trajectories only**. We
  strip unexpected tool-call args (`security_risk`, `timeout`, …) and drop trajectories that use
  the unsupported `task_tracker` tool or emit >1 tool call per turn.
- **Tool set (4):** `execute_bash, think, finish, str_replace_editor`. The system prompt is
  aligned to the rollout harness.
- **Pipeline** ([`sft/`](./sft)): `prepare_openhands.py` → `generate_sft.py` →
  `filter_by_length.py` → `run_sft_megatron_openhands.sh`.
- **Checkpoint caution:** HF exports must use `<|im_end|>` (ID `151645`) as the primary EOS in `tokenizer_config.json`, `config.json`, and `generation_config.json`; leaving the pretraining EOS (`151643`) causes rollouts to ignore assistant turn boundaries and run to `max_tokens`.

---

## Eval — SWE-bench Verified

Two phases use **two different image families** (see parity note).

| Phase | Dataset | Images | Repo path |
|-------|---------|--------|-----------|
| **Rollout** (agent solves) | `princeton-nlp/SWE-bench_Verified` / `test` | `xingyaoww/sweb.eval.x86_64.<id>` (`__`→`_s_`) — OpenHands fork | `/workspace/{repo}` |
| **Scoring** (apply patch, run tests) | same | `swebench/sweb.eval.x86_64.<id>` (`__`→`_1776_`) — official | `/testbed` |

**Images do NOT auto-pull — pre-pull the sampled set.**

- **Instance selection:** `--eval-n-limit N` does `dataset.sample(N, random_state=42)` — a
  seed-42 random draw, **not first-N** (n=1 ≠ first of n=100). Replicate exactly to know which
  images to pull; validate with `sample(1)`.
- **Rollout harness:** OpenHands `run_infer.py` with mounted remote runtime (`RUNTIME=remote`),
  model on vLLM (`--tool-call-parser hermes --enable-auto-tool-choice`). Requires
  **`native_tool_calling = true`** and **`temperature = 0.6`** — the model was SFT'd on hermes
  JSON tool calls, so non-native calling and low temps make it loop.
- **Scoring harness:** official `swebench` v3.0.17:
  ```bash
  python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified --split test \
    --predictions_path preds.jsonl --max_workers 8 --cache_level env --run_id <id>
  ```
  where preds = `{instance_id, model_patch=test_result.git_patch, model_name_or_path}`.
  Read `resolved_instances` from `<model>.<run_id>.json`.
- **Parity (verified safe):** `xingyaoww/` sits on the canonical `base_commit`; `swebench/` is
  that same commit + one setup commit. Same source, so rollout patches apply cleanly at scoring.
- **Baseline** (`global_step_1084`, 100 @ seed 42): **23/100 resolved (23%)** — 64 non-empty
  patches, 36 empty (looped/max-iter), 4 apply errors.

---

## RL — R2E-Gym (train) + SWE-bench Verified (val)

| Split | Dataset | `data_source` | Images |
|-------|---------|---------------|--------|
| **Train** | `R2E-Gym/R2E-Gym-Subset` / `train` | `r2e-gym` | **`instance_id` IS the image:** `namanjain12/<repo>_final:<commit>` |
| **Val** | `princeton-nlp/SWE-bench_Verified` / `test` | `swe-bench` | `sweb.eval.x86_64.<id>` (as in eval) |

- **Build data:** [`rl/prepare_rl_data.py`](./rl/prepare_rl_data.py) →
  `/home/tovi/data/r2e-all/{train,validation}.parquet`. `--val-size` subsamples the val split
  (the whole split is rolled out every `eval_interval`).
- **Run:** [`rl/run_skyrl_swe_30b.sh`](./rl/run_skyrl_swe_30b.sh) + config
  [`rl/skyrl_swe_30b.yaml`](./rl/skyrl_swe_30b.yaml). Tools match the SFT set
  (`enable_think=true`, `enable_search=false`). Rollouts run on the **remote runtime**
  (`SANDBOX_REMOTE_RUNTIME_API_URL`), not the GPU box.
- **Remote runtime connection (canonical):** configure the GPU client through
  the repo-root `.env`, following `skyrl-agent/.env.template`:
  `ALLHANDS_API_KEY=<shared-key>` and
  `SANDBOX_REMOTE_RUNTIME_API_URL=http://<cpu-box>:3000`. The recipe validates this file and
  forwards both values to Ray workers. Do not use `SANDBOX_API_KEY`, `OPENHANDS_API_KEY`, task
  YAML, or ad-hoc shell exports for the client connection; server-side configuration is separate.
- **Scale:** train = **4,578 instances across 10 repos** (pandas 1444, numpy 781, pillow 620,
  orange3 482, aiohttp 299, tornado 261, scrapy 215, pyramid 189, datalad 179, coveragepy 108).
  At `train_batch_size=64`, `n_samples_per_prompt=8`: ~72 steps/epoch, ~37k trajectories/epoch.
- **Image storage:** ~1.2–2.6 GB/image, **full set ≈ 7 TB** (pandas+orange3+numpy+pillow = ~81%).
  Can't bulk pre-pull on a 4 TB disk → **expand disk, subset, or on-demand-pull + LRU evict.** A
  training run needs its instances' images pre-pulled AND the train parquet filtered to that subset.

---

## At a glance

| | Eval rollout | Eval scoring | RL train | RL val |
|---|---|---|---|---|
| Dataset | SWE-bench_Verified/test | SWE-bench_Verified/test | R2E-Gym-Subset/train | SWE-bench_Verified/test |
| Images | `xingyaoww/…_s_…` | `swebench/…_1776_…` | `namanjain12/<repo>_final:<commit>` | `sweb.eval…` |
| data_source | — | — | `r2e-gym` | `swe-bench` |
| Count | 100 (seed 42) | 100 | 4,578 (10 repos) | `--val-size` (default 100) |

---

## RL environment (known-good build)

The upstream `uv run --isolated --extra skyrl-train` from `skyrl-agent` **is unbuildable** — fresh
resolution fails in a cascade. **Do NOT edit `skyrl-agent`** (upstream reference); all fixes live in
**our** [`rl/pyproject.toml`](./rl/pyproject.toml), so **run RL/smoke from `rl/`**.

| # | Conflict | Fix (in `rl/pyproject.toml`) |
|---|----------|------------------------------|
| 1 | `ImportError: resolve_policy_model_name` (SkyRL@main stale) | `skyrl` → local `path` + editable |
| 2 | protobuf: vllm≥0.19 needs ≥5.29.6, openhands 0.38 pins <5 | `override: protobuf>=5.29.6` (runs fine on protobuf 6) |
| 3 | huggingface-hub: transformers 5.x needs ≥1.5, browsergym (unused) needs <1.0 | `override: huggingface-hub>=1.5.0` |
| 4 | verl dead-pins `vllm==0.8.5` | `environments=[…x86_64]` + `override: vllm>=0.8.5` |
| 5 | `OpenAIBackend … tokenizer` (smoke only) | monkeypatch in `smoke_rollout.py` (real RL uses `SkyRLBackend`) |

Not a version-picking problem: **no vllm has both protobuf-4 and `resolve_policy_model_name`**.
`run_skyrl_swe_30b.sh` needs the same env (point it at our pyproject) or it hits the same cascade.

### Smoke test (preflight)

Runs **one** r2e rollout through the real `AutoAgentRunner.run()` path against a served vLLM —
confirms the hermes `OHCodeActAgent` emits `<tool_call>` (not `<function>`) and doesn't loop.

```bash
cd examples/train/multi-env/rl
SMOKE_API_URL=http://<vllm-host>:8010 SMOKE_MODEL_NAME=sft \
uv run --isolated --env-file "$(git rev-parse --show-toplevel)/.env" python smoke_rollout.py
```
