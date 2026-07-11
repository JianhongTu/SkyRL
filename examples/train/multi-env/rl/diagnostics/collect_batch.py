"""Diagnostic: collect a NPROMPTS x 8 rollout batch (before any weight update) via
the faithful hermes rollout path against a *served* vLLM, and dump per-trajectory
rewards + rollout metrics. This is the real RL rollout config (hermes agent +
SWEBench task) driven by a served engine, so it is backend-independent for
reward / end-reason / format analysis.

Usage (writes ./collect_out.json in the current directory):

    MODEL=<hf ckpt dir> DATA=<swe train.parquet> \
    SMOKE_API_URL=http://<host>:<port> SMOKE_MODEL_NAME=<served-name> \
    SANDBOX_RUNTIME_BUNDLE_HOST_PATH=<mounted openhands bundle> \
    SANDBOX_REMOTE_RUNTIME_API_URL=... OPENHANDS_API_KEY=... \
    python collect_batch.py

Env:
    MODEL     (required) HF checkpoint dir (tokenizer source)
    DATA      (required) parquet of SWE instances (train split)
    NPROMPTS  prompts to sample, x8 trajectories each (default 32)
    OUT       output json (default ./collect_out.json)
    SMOKE_API_URL / SMOKE_MODEL_NAME       served vLLM endpoint + name
    SANDBOX_RUNTIME_BUNDLE_HOST_PATH       mounted OpenHands runtime bundle
    + the sandbox vars the SWEBench task needs (remote runtime URL + key)
"""
import asyncio, json, os, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RL = _HERE.parent  # the rl/ dir holds hermes_codeact_agent.py
sys.path.insert(0, str(RL))

# 1) activate hermes agent (rebinds OHCodeActAgent) -- matches rl_train_entry
import hermes_codeact_agent  # noqa: E402,F401

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from skyrl_agent import AutoAgentRunner  # noqa: E402

# 2) openai backend accepts+ignores tokenizer kwarg (smoke shim)
from skyrl_agent.integrations import openai as _oai  # noqa: E402
_orig = _oai.OpenAIBackend.__init__
def _patched(self, infer_engine, cfg=None, tokenizer=None, **_k):  # noqa: ANN001
    _orig(self, infer_engine, cfg)
_oai.OpenAIBackend.__init__ = _patched

# 3) force mounted remote runtime (no per-instance image build)
from skyrl_agent.tasks.swebench import utils as _swe  # noqa: E402
_o = _swe.get_default_sandbox_config_for_eval
def _mounted():
    cfg = _o(); cfg.runtime_mode = "mounted"
    bundle = os.environ.get("SANDBOX_RUNTIME_BUNDLE_HOST_PATH")
    if bundle:
        cfg.runtime_bundle_host_path = bundle
    cfg.runtime_bundle_container_path = "/opt/openhands-runtime"
    return cfg
_swe.get_default_sandbox_config_for_eval = _mounted

os.environ.setdefault("OPENAI_API_KEY", "sc")
os.environ.setdefault("RUNTIME", "remote")
os.environ.setdefault("SMOKE_MODEL_NAME", "sft")
os.environ.setdefault("SMOKE_API_URL", "http://localhost:8010")

MODEL = os.environ.get("MODEL")
DATA = os.environ.get("DATA")
NPROMPTS = int(os.environ.get("NPROMPTS", "32"))
OUT = os.environ.get("OUT", "collect_out.json")
YAML = str(_HERE / "collect_batch.yaml")

def main():
    if not MODEL or not DATA:
        sys.exit("ERROR: set MODEL=<hf ckpt dir> and DATA=<train.parquet> (see module docstring)")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("parquet", data_files=DATA)["train"].select(range(NPROMPTS))
    print(f"[collect] {NPROMPTS} prompts x 8 traj = {NPROMPTS*8} trajectories; yaml={YAML}", flush=True)
    runner = AutoAgentRunner.from_task(YAML, infer_engine=None, tokenizer=tok)
    # val_mode=False -> training-style rollout: 8 samples/prompt, per-traj reward + full metrics
    output = asyncio.run(runner.run(ds, val_mode=False))
    print(f"\n[collect] done in {(time.time()-t0)/60:.1f} min. keys: {list(output.keys())}", flush=True)
    print("[collect] rewards (first 40):", (output.get("rewards") or [])[:40], flush=True)
    print("[collect] rollout_metrics:", json.dumps(output.get("rollout_metrics", {}), indent=2, default=str)[:4000], flush=True)
    slim = {k: v for k, v in output.items() if k != "batch"}
    with open(OUT, "w") as f:
        json.dump(slim, f, default=str)
    print(f"[collect] wrote {os.path.abspath(OUT)}", flush=True)

if __name__ == "__main__":
    main()
