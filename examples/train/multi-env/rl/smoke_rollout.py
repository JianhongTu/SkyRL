"""Standalone rollout smoke for HermesOHCodeActAgent (option B).

Runs the EXACT training rollout code path (AutoAgentRunner.run -> CodeActTrajectory
-> run_controller -> agent.step()) against a served vLLM via the openai_server
backend. Importing ``hermes_codeact_agent`` makes the class selected by the task
config available, so this exercises our Hermes subclass directly.

PURPOSE: confirm the model now emits clean <tool_call> (not <function>/escaped),
makes a real edit, and does NOT loop — the fix validated in the eval.

PREREQUISITES (all on the box you run this from, reaching the two services):
  1. vLLM serving the SFT export, OpenAI-compatible:
       vllm serve $MODEL --served-model-name sft --port 8010 ...
     -> SMOKE_API_URL=http://<host>:8010  (NO /v1 suffix), SMOKE_MODEL_NAME=sft
  2. remote_runtime_server up on a3255b8 (auto-pull) + mounts + OPENHANDS_API_KEY,
     reachable at SANDBOX_REMOTE_RUNTIME_API_URL. docker login done (so the r2e
     image can be pulled on cache-miss).
  3. Run in the skyrl-train env (skyrl_agent + openhands importable):
       cd /home/tovi/SkyRL
       uv run --isolated --extra skyrl-train python \
         examples/train/multi-env/rl/smoke_rollout.py

ENV VARS (with defaults):
  SMOKE_MODEL_NAME=sft
  SMOKE_API_URL=http://localhost:8010
  SMOKE_MODEL_PATH=/data/tovi/exports/skyrl_sft_openhands_hf_harnessprompt/global_step_1084/policy
  SMOKE_DATA=/home/tovi/data/r2e-all/train.parquet
  SMOKE_INSTANCE_INDEX=0            # which r2e row to roll out
  SANDBOX_REMOTE_RUNTIME_API_URL, OPENHANDS_API_KEY / SANDBOX_API_KEY / ALLHANDS_API_KEY
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# --- make the subclass importable ---------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import hermes_codeact_agent  # noqa: E402

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from skyrl_agent import AutoAgentRunner  # noqa: E402

# skyrl-agent's build_backend passes tokenizer=... to every backend ctor, but the
# openai_server OpenAIBackend signature is (infer_engine, cfg) — it never got the
# tokenizer arg the SkyRLBackend (real-RL) has. Only bites the openai_server path
# we use for the smoke; patch it to accept+ignore tokenizer. (skyrl-agent untouched.)
from skyrl_agent.integrations import openai as _oai_backend  # noqa: E402

_oai_orig_init = _oai_backend.OpenAIBackend.__init__


def _oai_patched_init(self, infer_engine, cfg=None, tokenizer=None, **_kw):  # noqa: ANN001
    _oai_orig_init(self, infer_engine, cfg)


_oai_backend.OpenAIBackend.__init__ = _oai_patched_init

os.environ.setdefault("SANDBOX_RUNTIME_MODE", "mounted")
os.environ.setdefault(
    "SANDBOX_RUNTIME_BUNDLE_HOST_PATH", "/opt/openhands-runtime/current"
)
os.environ.setdefault(
    "SANDBOX_RUNTIME_BUNDLE_CONTAINER_PATH", "/opt/openhands-runtime"
)

# --- env / defaults -----------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY", "sc")          # OpenAIBackend asserts this
os.environ.setdefault("RUNTIME", "remote")             # use the remote runtime
os.environ.setdefault("SMOKE_MODEL_NAME", "sft")
os.environ.setdefault("SMOKE_API_URL", "http://localhost:8010")
MODEL_PATH = os.environ.get(
    "SMOKE_MODEL_PATH",
    "/data/tovi/exports/skyrl_sft_openhands_hf_harnessprompt/global_step_1084/policy",
)
DATA = os.environ.get("SMOKE_DATA", "/home/tovi/data/r2e-all/train.parquet")
IDX = int(os.environ.get("SMOKE_INSTANCE_INDEX", "0"))
YAML = str(_HERE / "smoke_hermes.yaml")

# sanity: the sandbox api key must match the runtime server's OPENHANDS_API_KEY
for k in ("SANDBOX_REMOTE_RUNTIME_API_URL",):
    if not os.environ.get(k):
        print(f"[WARN] {k} is not set — the r2e sandbox will fail to start.")


def main():
    print(f"== smoke: agent={hermes_codeact_agent.HermesOHCodeActAgent.__name__} ==")
    print(f"   model_name={os.environ['SMOKE_MODEL_NAME']}  api_url={os.environ['SMOKE_API_URL']}")
    print(f"   yaml={YAML}")
    print(f"   data={DATA}  instance_index={IDX}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    dataset = load_dataset("parquet", data_files=DATA)["train"].select([IDX])
    inst = dataset[0]
    print(f"   instance_id={inst.get('instance_id')}  data_source={inst.get('data_source')}")

    # infer_engine=None -> the openai_server backend is built from the yaml config
    agent_generator = AutoAgentRunner.from_task(YAML, infer_engine=None, tokenizer=tokenizer)
    output = asyncio.run(agent_generator.run(dataset, val_mode=True))

    print("\n==== ROLLOUT OUTPUT ====")
    print("keys:", list(output.keys()))
    if "rewards" in output:
        print("rewards:", output["rewards"])
    if "rollout_metrics" in output:
        print("rollout_metrics:", json.dumps(output["rollout_metrics"], indent=2, default=str))

    out_path = Path(os.environ.get("SMOKE_OUT", "/tmp/smoke_rollout_output.json"))
    with open(out_path, "w") as f:
        json.dump(output, f, default=str)
    print(f"\nfull output -> {out_path}")
    print(
        "\nVALIDATION: grep this run's stdout for lines starting `response ` — the raw\n"
        "model generations. PASS = they contain <tool_call>{...}</tool_call> (hermes),\n"
        "the agent runs execute_bash/str_replace_editor, and it does NOT repeat the\n"
        "same action to a loop. FAIL = <function=...>/escaped args or immediate loop."
    )


if __name__ == "__main__":
    main()
