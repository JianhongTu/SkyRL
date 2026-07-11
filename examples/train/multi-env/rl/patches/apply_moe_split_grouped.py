"""Patch yield_module_grouped_chunks (the CUDA-IPC / group_by_module path) to split
transformers>=5 fused Qwen3-MoE experts into vLLM's separate per-expert tensors.
This is the path colocated same-node sync actually uses. Run as root in-container."""
import sys
F = "/home/tovi/SkyRL/examples/train/multi-env/rl/.venv/lib/python3.12/site-packages/skyrl/backends/skyrl_train/weight_sync/weight_extractor_utils.py"
src = open(F).read()
if "mlp.experts.gate_up_proj" in src:
    print("ALREADY PATCHED"); sys.exit(0)

OLD = '''        for param_name in param_names:
            param = params[param_name]
            tensor = gather_tensor_fn(param)
            tensor = tensor.to(dtype).detach().contiguous()
            shape = get_shape_fn(param_name, param, tensor)
            module_tensors.append(tensor)
            module_names.append(param_name)
            module_shapes.append(shape)
            module_dtypes.append(str(dtype))
            module_size += tensor.nbytes'''

NEW = '''        for param_name in param_names:
            param = params[param_name]
            tensor = gather_tensor_fn(param)
            tensor = tensor.to(dtype).detach().contiguous()
            # transformers>=5 stores Qwen3-MoE experts FUSED/grouped
            # (experts.gate_up_proj [E,2I,H]; experts.down_proj [E,H,I]), but vLLM's
            # load_weights expects SEPARATE per-expert experts.{n}.gate_proj/up_proj/
            # down_proj.weight (the on-disk HF checkpoint format it loads cleanly at
            # init). Split the gathered full tensor here so the FSDP->vLLM sync maps
            # correctly; otherwise the fused names mis-map and the colocated engine
            # emits deterministic gibberish. No-op on transformers<5 (no fused key).
            if param_name.endswith("mlp.experts.gate_up_proj"):
                E, twoI = tensor.shape[0], tensor.shape[1]
                I = twoI // 2
                pre = param_name[: -len("gate_up_proj")]
                for n in range(E):
                    for part, sl in (("gate_proj", tensor[n, :I, :]), ("up_proj", tensor[n, I:, :])):
                        t = sl.contiguous()
                        module_tensors.append(t)
                        module_names.append(f"{pre}{n}.{part}.weight")
                        module_shapes.append(list(t.shape))
                        module_dtypes.append(str(dtype))
                        module_size += t.nbytes
                continue
            if param_name.endswith("mlp.experts.down_proj"):
                E = tensor.shape[0]
                pre = param_name[: -len("down_proj")]
                for n in range(E):
                    t = tensor[n].contiguous()
                    module_tensors.append(t)
                    module_names.append(f"{pre}{n}.down_proj.weight")
                    module_shapes.append(list(t.shape))
                    module_dtypes.append(str(dtype))
                    module_size += t.nbytes
                continue
            shape = get_shape_fn(param_name, param, tensor)
            module_tensors.append(tensor)
            module_names.append(param_name)
            module_shapes.append(shape)
            module_dtypes.append(str(dtype))
            module_size += tensor.nbytes'''

assert src.count(OLD) == 1, f"OLD matched {src.count(OLD)} times"
open(F, "w").write(src.replace(OLD, NEW))
print("PATCHED OK")
