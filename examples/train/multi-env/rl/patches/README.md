# MoE weight-sync patch (REQUIRED for the colocated RL run)

## Why
`transformers>=5` stores Qwen3-MoE experts **fused/grouped** — one
`...mlp.experts.gate_up_proj` `[E, 2I, H]` param (gate+up for all experts) and
`...mlp.experts.down_proj` `[E, H, I]`. SkyRL's FSDP policy is a transformers
model, so its `state_dict()` ships those **fused** names verbatim on the
FSDP→vLLM weight sync (`save_weights_for_sampler` at trainer setup). But vLLM's
`Qwen3MoE.load_weights` expects **separate** per-expert
`experts.{n}.gate_proj/up_proj/down_proj.weight` (the on-disk HF checkpoint
format it loads cleanly at init) and explicitly skips `mlp.experts` from its
fused path. The fused names mis-map → the colocated engine's MoE experts are
corrupted → **deterministic gibberish from token 1** (byte-identical across
temperatures — the tell that it's wrong weights, not sampling).

We **cannot** just downgrade transformers to the v4 line (which keeps separate
experts): SkyRL's FSDP backend calls the **v5-only** `named_non_persistent_buffers()`
in `init_model` (v4 → `AttributeError`), and vLLM warns the v4 codepath is being
removed. So we stay on transformers 5.x and fix the sync at the extractor.

## What these scripts do
Split the fused expert tensor into separate per-expert tensors right after it is
gathered (full plain tensor → safe to slice), in SkyRL's FSDP weight extractor:
`gate = t[n,:I,:]`, `up = t[n,I:,:]`, `down = down[n]` (verified byte-exact vs the
checkpoint). vLLM then loads by name via the idempotent `model.load_weights`.

- `apply_moe_split_grouped.py` → patches `weight_sync/weight_extractor_utils.py`
  `yield_module_grouped_chunks`. **This is the one that matters** — colocated
  same-node sync uses CUDA-IPC → `group_by_module=True` → this grouped path.
- `apply_moe_split_simple.py` → patches `workers/fsdp/fsdp_worker.py`
  `extract_weights` (+ `get_weight_metadata`), the non-grouped branch. For
  completeness; not on the colocated hot path.

Both are **no-ops** if the fused key is absent (idempotent; safe to re-run).

## When to run
After **any** `uv sync` / venv rebuild (it reinstalls skyrl and wipes the edit).
Run from inside the container as root (the venv site-packages may be root-owned
after a prior patch; running as root is always safe):

```bash
docker exec -u root skyrl-mega python3 \
  /home/tovi/SkyRL/examples/train/multi-env/rl/patches/apply_moe_split_grouped.py
docker exec -u root skyrl-mega python3 \
  /home/tovi/SkyRL/examples/train/multi-env/rl/patches/apply_moe_split_simple.py
```

Confirmed working (rl_smoke4): 0 `%",` garbage, coherent multi-turn `execute_bash`
/ `str_replace_editor` tool calls, `avg_turn_assistant > 1`.
