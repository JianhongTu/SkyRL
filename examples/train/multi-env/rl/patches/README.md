# MoE weight-sync fix — NOW IN SOURCE (these scripts are legacy)

**As of the merge to `NovaSky/SkyRL@af36dc04`, the fix lives in the skyrl source**
and skyrl is installed **editable from this local repo**, so nothing here needs
re-applying. The scripts below are kept only for reference / for a non-editable
(copy) skyrl install.

## The fix (in source, committed)
`transformers>=5` stores MoE experts **fused/grouped** (e.g. Qwen3-MoE
`experts.gate_up_proj` `[E,2I,H]`, `experts.down_proj` `[E,H,I]`), but vLLM's
`load_weights` expects **separate** per-expert `experts.{n}.gate_proj/up_proj/
down_proj.weight`. SkyRL's FSDP policy ships the fused `state_dict()` verbatim on
the colocated FSDP→vLLM sync → vLLM mis-maps → deterministic gibberish. The fix
splits fused→separate in the weight extractor:

- `skyrl/backends/skyrl_train/weight_sync/weight_extractor_utils.py` →
  `yield_module_grouped_chunks` (**the CUDA-IPC / `group_by_module=True` path the
  colocated sync actually uses**)
- `skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py` → `extract_weights`
  (simple path) + `get_weight_metadata`

Split (verified byte-exact vs the checkpoint): `gate = t[n,:I,:]`,
`up = t[n,I:,:]`, `down = down[n]`. No-op on transformers<5. vLLM loads by name
via the idempotent `model.load_weights`.

## Install source (why it's durable now)
`skyrl-agent/pyproject.toml` and `rl/pyproject.toml` pin
`skyrl = { path = "/home/tovi/SkyRL", editable = true }` (was `git NovaSky main`).
So skyrl runs from this local tree — the fix is version-controlled and survives
`uv sync`. Confirmed: `import skyrl...weight_extractor_utils` resolves to
`/home/tovi/SkyRL/skyrl/...`. Gotcha: if a prior *copy* install left a
`site-packages/skyrl/` shell, remove it — it shadows the editable finder.

## Legacy scripts (only for a non-editable/copy skyrl)
`apply_moe_split_grouped.py` / `apply_moe_split_simple.py` patch a venv-copy of
skyrl in place. Not needed with the editable install above.

Validated (rl_smoke6, editable source): 0 `%",` garbage, coherent multi-turn
`execute_bash`/`str_replace_editor`, `avg_turn_assistant` 1.5.
