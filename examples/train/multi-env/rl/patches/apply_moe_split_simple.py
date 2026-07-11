"""Patch skyrl FSDPWeightExtractor to split transformers>=5 fused Qwen3-MoE
experts into vLLM's expected separate per-expert tensors. Run as root in-container."""
import sys

F = "/home/tovi/SkyRL/examples/train/multi-env/rl/.venv/lib/python3.12/site-packages/skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py"
src = open(F).read()

if "mlp.experts.gate_up_proj" in src:
    print("ALREADY PATCHED — no change"); sys.exit(0)

OLD_EXTRACT = '''        if not self.group_by_module:
            # Simple path: yield one chunk per parameter
            for name, param in params.items():
                tensor = self._gather_tensor(param).to(dtype).detach().contiguous()
                yield WeightChunk(
                    names=[name],
                    dtypes=[str(dtype)],
                    shapes=[list(tensor.shape)],
                    tensors=[tensor],
                )
        else:'''

NEW_EXTRACT = '''        if not self.group_by_module:
            # Simple path: yield one chunk per parameter.
            # transformers>=5 stores Qwen3-MoE experts FUSED/grouped
            # (experts.gate_up_proj [E,2I,H]; experts.down_proj [E,H,I]), but
            # vLLM's load_weights expects SEPARATE per-expert tensors
            # experts.{n}.gate_proj/up_proj/down_proj.weight (the on-disk HF
            # checkpoint format). Split here so the FSDP->vLLM weight sync maps
            # correctly; otherwise the fused names mis-map and the colocated
            # engine emits deterministic gibberish. No-op on transformers<5
            # (no fused key -> falls through to the plain else branch).
            for name, param in params.items():
                if name.endswith("mlp.experts.gate_up_proj"):
                    full = self._gather_tensor(param).to(dtype).detach()  # [E, 2I, H]
                    E, twoI = full.shape[0], full.shape[1]
                    I = twoI // 2
                    pre = name[: -len("gate_up_proj")]  # "...mlp.experts."
                    for n in range(E):
                        for part, sl in (("gate_proj", full[n, :I, :]), ("up_proj", full[n, I:, :])):
                            t = sl.contiguous()
                            yield WeightChunk(
                                names=[f"{pre}{n}.{part}.weight"],
                                dtypes=[str(dtype)],
                                shapes=[list(t.shape)],
                                tensors=[t],
                            )
                elif name.endswith("mlp.experts.down_proj"):
                    full = self._gather_tensor(param).to(dtype).detach()  # [E, H, I]
                    E = full.shape[0]
                    pre = name[: -len("down_proj")]
                    for n in range(E):
                        t = full[n].contiguous()
                        yield WeightChunk(
                            names=[f"{pre}{n}.down_proj.weight"],
                            dtypes=[str(dtype)],
                            shapes=[list(t.shape)],
                            tensors=[t],
                        )
                else:
                    tensor = self._gather_tensor(param).to(dtype).detach().contiguous()
                    yield WeightChunk(
                        names=[name],
                        dtypes=[str(dtype)],
                        shapes=[list(tensor.shape)],
                        tensors=[tensor],
                    )
        else:'''

OLD_META = '''        for name, param in self.model.state_dict().items():
            names.append(f"{self.weight_prefix}{name}" if self.weight_prefix else name)
            dtype_names.append(dtype_name)
            shapes.append(list(param.shape))
        return {"names": names, "dtype_names": dtype_names, "shapes": shapes}'''

NEW_META = '''        for name, param in self.model.state_dict().items():
            nm = f"{self.weight_prefix}{name}" if self.weight_prefix else name
            # Mirror extract_weights: split fused transformers>=5 Qwen3-MoE experts
            # into separate per-expert names/shapes (order must match extract_weights).
            if nm.endswith("mlp.experts.gate_up_proj"):
                E, twoI = param.shape[0], param.shape[1]
                I = twoI // 2
                H = param.shape[2]
                pre = nm[: -len("gate_up_proj")]
                for n in range(E):
                    for part in ("gate_proj", "up_proj"):
                        names.append(f"{pre}{n}.{part}.weight")
                        dtype_names.append(dtype_name)
                        shapes.append([I, H])
            elif nm.endswith("mlp.experts.down_proj"):
                E, H, I = param.shape[0], param.shape[1], param.shape[2]
                pre = nm[: -len("down_proj")]
                for n in range(E):
                    names.append(f"{pre}{n}.down_proj.weight")
                    dtype_names.append(dtype_name)
                    shapes.append([H, I])
            else:
                names.append(nm)
                dtype_names.append(dtype_name)
                shapes.append(list(param.shape))
        return {"names": names, "dtype_names": dtype_names, "shapes": shapes}'''

assert src.count(OLD_EXTRACT) == 1, f"OLD_EXTRACT matched {src.count(OLD_EXTRACT)} times"
assert src.count(OLD_META) == 1, f"OLD_META matched {src.count(OLD_META)} times"
src = src.replace(OLD_EXTRACT, NEW_EXTRACT).replace(OLD_META, NEW_META)
open(F, "w").write(src)
print("PATCHED OK")
