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
- Most trajectories are **loss-masked out** (they hit the turn cap) → ≈0 gradient; even
  *solved* trajectories are discarded. Revisit the mask-out policy.
- The model **almost never emits `finish`**, so it runs to the turn cap. And when it does,
  the trainer's finish check doesn't recognize the hermes finish call → also masked. **[4]**
- **GRPO groups collapse** when a prompt's samples fail to start (sandbox failures cluster
  per prompt) → degenerate/undefined advantage; re-sample or drop such prompts.

**P1 — rollout infra & capacity**
- Sandbox runtime **saturates far below** the configured concurrency (cold starts +
  capacity). Some base images are cached; finish caching the rest, **apply crun**, and
  **measure the real sustainable concurrency**. **[1] [2]**
- **Reward-eval reliability at batch scale** — confirm reward is computed inline (before
  sandbox teardown) and holds under concurrency.

**P1 — model behavior (wastes the turn budget → feeds P0)**
- **Blocking foreground commands** hang turns (agent waits out timeouts).
- **Reasoning spirals** — over-thinks to the token cap without acting.
- **Workspace-path mismatch** — probes the SFT path, not the runtime path; wastes a turn.

**P2 — config & tuning**
- **Temperature** — pick a value for healthy reward entropy **and** confirm it actually
  reaches the rollout sampler (task-config vs launcher layering). **[3]**
- **Reward shaping** to encourage a `finish` call (paired with the finish-detection fix). **[4]**
- **Fit check** — confirm FSDP + colocated vLLM memory fits the 8×H200 node at the target
  batch/context. **[5]**

**P3 — nice-to-have**
- Relaxed tool-call parser + stop-at-tool-close: cheap insurance, but touches ~3% of turns;
  not the lever. Deprioritized.

_(Bracketed tags map to the original issue list 1–5.)_

## Reproduce the diagnosis

Environment is **locked**: env pins in `pyproject.toml` / `uv.lock`; the MoE weight-sync fix
lives in the skyrl source and is installed editable, so it needs no re-applying.

`diagnostics/` holds the dry-run harness:
- `collect_batch.py` + `collect_batch.yaml` — run the 32×8 rollout batch against a served
  vLLM (hermes agent = real RL config), dump per-trajectory rewards/metrics.
- `parse_log.py` — summarize the rollout log into the symptoms above (end-reason, turn-length,
  finish rate, sandbox-failure counts).

All inputs are passed via env/args (each script's docstring lists them); the scripts
reference nothing outside the repo.
