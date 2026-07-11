# Multi-env RL phase — status & open issues

RL from the SFT checkpoint on the SWE task (r2e-gym train / SWE-bench-Verified val),
hermes tool-calling, colocated FSDP2 + vLLM on one 8×H200 node, remote OpenHands
runtime for sandboxes.

## Diagnosis so far (256-rollout dry run, before any weight update)

The hermes rollout itself is **clean** — well-formed single tool calls, stops cleanly,
no fabrication/format-drift. The blocker is **elsewhere**: the batch delivers almost
**zero trainable signal**, and the sandbox tier saturates well below the configured
rollout concurrency. Reward eval could not complete at batch scale in the standalone
harness. Details in `diagnostics/` (below).

## Open issues (prioritized)

**P0 — training signal (decides whether RL learns at all)**
- Turn-cap trajectories are now trainable; only infrastructure/runtime, evaluation, malformed-response,
  loop, and command-timeout failures are masked. Re-measure effective gradient coverage.
- Finish recognition accepts Hermes and legacy syntax, and a valid training rollout receives a small
  `+0.05` finish bonus. Re-measure whether this improves termination without premature submission. **[4]**
- **GRPO groups collapse** when a prompt's samples fail to start (sandbox failures cluster
  per prompt) → exactly zero RLOO advantage. Use the structured group report to decide whether
  dynamic filtering/resampling is affordable.

**P1 — rollout infra & capacity**
- Sandbox runtime **saturates far below** the configured concurrency (cold starts +
  capacity). Some base images are cached; finish caching the rest, **apply crun**, and
  **measure the real sustainable concurrency**. **[1] [2]**
- **Reward-eval reliability at batch scale** — R2E reward runs inline before sandbox teardown,
  and incomplete test commands are now classified as evaluation errors; validate under concurrency.

**P1 — model behavior (wastes the turn budget → feeds P0)**
- **Blocking foreground commands** — the task now instructs backgrounding/interruption; measure whether
  a shorter enforced command timeout is still needed.
- **Reasoning spirals** — per-turn generation is now truly capped at 4K; re-measure before lowering it.
- **Workspace-path mismatch** — R2E now exposes the learned `/workspace/...` path as an alias to `/testbed`.

**P2 — config & tuning**
- **Temperature** — launcher sampling now reaches the rollout sampler and defaults to `0.6/0.95`;
  use the corrected group report before tuning. **[3]**
- **Fit check** — confirm FSDP + colocated vLLM memory fits the 8×H200 node at the target
  batch/context. **[5]**

**P3 — nice-to-have**
- Relaxed tool-call parser + stop-at-tool-close: cheap insurance, but touches ~3% of turns;
  not the lever. Deprioritized.

_(Bracketed tags map to the original issue list 1–5.)_

## Reproduce the diagnosis

Environment is **locked**: env pins in `pyproject.toml` / `uv.lock`; the MoE weight-sync fix
lives in the skyrl source and is installed editable, so it needs no re-applying.

Hermes rollouts now record exact sampled input/output token IDs; training no longer reconstructs
them through the generic message-template fallback. Structured records include task and shaped
rewards, finish bonus, mask status, token counts, stop reasons, and generation/evaluation timing.

`diagnostics/` holds the dry-run harness:
- `collect_batch.py` — derive the real task config, reproduce the seeded first 32×8
  rollout batch against a served vLLM, and dump structured trajectory/RLOO metrics.
- `parse_log.py` — summarize the rollout log into the symptoms above (end-reason, turn-length,
  finish rate, sandbox-failure counts).

All inputs are passed via env/args (each script's docstring lists them); the scripts
reference nothing outside the repo.
