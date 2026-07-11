"""Collect the trainer's first seeded rollout batch before any weight update.

The diagnostic derives its task configuration from ``../skyrl_swe_30b.yaml`` and
only replaces the internal SkyRL backend with an OpenAI-compatible served vLLM.
It emits structured per-trajectory records and loop/RLOO group statistics.

Usage (writes ./collect_out.json in the current directory):

    MODEL=<hf ckpt dir> DATA=<swe train.parquet> \
    SMOKE_API_URL=http://<host>:<port> SMOKE_MODEL_NAME=<served-name> \
    SANDBOX_RUNTIME_BUNDLE_HOST_PATH=<mounted openhands bundle> \
    SANDBOX_REMOTE_RUNTIME_API_URL=... ALLHANDS_API_KEY=... \
    python collect_batch.py

Env:
    MODEL     (required) HF checkpoint dir (tokenizer source)
    DATA      (required) parquet of SWE instances (train split)
    NPROMPTS / BATCH_SIZE  first-batch prompt count (default 32)
    OUT       output json (default ./collect_out.json)
    SEED      dataloader seed (default 1, matching training)
    DIAGNOSTIC_MODE  fidelity (default, training concurrency) or quality (32 agents)
    MAX_PARALLEL_AGENTS  explicit concurrency override
    TEMP / TOP_P         sampling overrides shared with the training launcher
    SMOKE_API_URL / SMOKE_MODEL_NAME       served vLLM endpoint + name
    SANDBOX_RUNTIME_BUNDLE_HOST_PATH       mounted OpenHands runtime bundle
    + the sandbox vars the SWEBench task needs (remote runtime URL + key)
"""

import asyncio
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from analysis import summarize_rloo_groups

_HERE = Path(__file__).resolve().parent
RL = _HERE.parent  # the rl/ dir holds hermes_codeact_agent.py
sys.path.insert(0, str(RL))

os.environ.setdefault("OPENAI_API_KEY", "sc")
os.environ.setdefault("RUNTIME", "remote")
os.environ.setdefault("SMOKE_MODEL_NAME", "sft")
os.environ.setdefault("SMOKE_API_URL", "http://localhost:8010")

MODEL = os.environ.get("MODEL")
DATA = os.environ.get("DATA")
NPROMPTS = int(os.environ.get("NPROMPTS", os.environ.get("BATCH_SIZE", "32")))
SEED = int(os.environ.get("SEED", os.environ.get("seed", "1")))
OUT = os.environ.get("OUT", "collect_out.json")
TRAINING_YAML = RL / "skyrl_swe_30b.yaml"


def _require_environment():
    missing = [
        name
        for name in (
            "SANDBOX_RUNTIME_BUNDLE_HOST_PATH",
            "SANDBOX_REMOTE_RUNTIME_API_URL",
            "ALLHANDS_API_KEY",
        )
        if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit(
            f"ERROR: missing required environment variables: {', '.join(missing)}"
        )


def _build_resolved_config():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(TRAINING_YAML)
    cfg.generator.infer_backend = "openai_server"
    cfg.generator.backend_config = {
        "model_name": os.environ["SMOKE_MODEL_NAME"],
        "api_url": os.environ["SMOKE_API_URL"],
        "model_max_len": 32768,
        "require_token_ids": True,
    }
    if os.environ.get("TEMP"):
        cfg.generator.sampling_params.temperature = float(os.environ["TEMP"])
    if os.environ.get("TOP_P"):
        cfg.generator.sampling_params.top_p = float(os.environ["TOP_P"])

    mode = os.environ.get("DIAGNOSTIC_MODE", "fidelity")
    if mode not in {"fidelity", "quality"}:
        raise SystemExit("ERROR: DIAGNOSTIC_MODE must be 'fidelity' or 'quality'")
    if mode == "quality":
        cfg.dispatcher.max_parallel_agents = 32
        cfg.dispatcher.max_eval_parallel_agents = 32
    if os.environ.get("MAX_PARALLEL_AGENTS"):
        concurrency = int(os.environ["MAX_PARALLEL_AGENTS"])
        cfg.dispatcher.max_parallel_agents = concurrency
        cfg.dispatcher.max_eval_parallel_agents = concurrency
    return cfg, mode


def _first_training_batch(tokenizer, max_prompt_length):
    import torch
    from torchdata.stateful_dataloader import StatefulDataLoader

    from skyrl.train.dataset import PromptDataset

    dataset = PromptDataset(
        datasets=DATA,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        num_workers=8,
    )
    if len(dataset) < NPROMPTS:
        raise SystemExit(
            f"ERROR: filtered dataset has {len(dataset)} rows, fewer than NPROMPTS={NPROMPTS}"
        )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=NPROMPTS,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=8,
        drop_last=True,
        generator=generator,
        multiprocessing_context="spawn",
    )
    batch = next(iter(dataloader))
    return [item["env_extras"] for item in batch], [item["uid"] for item in batch]


def main():
    if not MODEL or not DATA:
        sys.exit(
            "ERROR: set MODEL=<hf ckpt dir> and DATA=<train.parquet> (see module docstring)"
        )
    _require_environment()

    # Activate HermesOHCodeActAgent before AgentRunner resolves its trajectory class.
    importlib.import_module("hermes_codeact_agent")

    from omegaconf import OmegaConf
    from skyrl_agent import AutoAgentRunner

    # Match rl_train_entry.py's mounted remote runtime behavior.
    from skyrl_agent.tasks.swebench import utils as _swe
    from transformers import AutoTokenizer

    original_sandbox_config = _swe.get_default_sandbox_config_for_eval

    def _mounted():
        sandbox_cfg = original_sandbox_config()
        sandbox_cfg.runtime_mode = "mounted"
        sandbox_cfg.runtime_bundle_host_path = os.environ[
            "SANDBOX_RUNTIME_BUNDLE_HOST_PATH"
        ]
        sandbox_cfg.runtime_bundle_container_path = "/opt/openhands-runtime"
        return sandbox_cfg

    _swe.get_default_sandbox_config_for_eval = _mounted

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg, mode = _build_resolved_config()
    filter_max_prompt_length = int(
        os.environ.get("MAX_PROMPT_LEN", cfg.generator.max_prompt_length)
    )
    batch, batch_uids = _first_training_batch(
        tok, max_prompt_length=filter_max_prompt_length
    )
    selected_instance_ids = [row["instance"]["instance_id"] for row in batch]
    resolved = {
        "mode": mode,
        "model": MODEL,
        "data": DATA,
        "seed": SEED,
        "prompt_count": NPROMPTS,
        "batch_uids": batch_uids,
        "instance_ids": selected_instance_ids,
        "num_trajectories": cfg.generator.num_trajectories,
        "max_iterations": cfg.generator.max_iterations,
        "max_prompt_length": cfg.generator.max_prompt_length,
        "filter_max_prompt_length": filter_max_prompt_length,
        "sampling_params": OmegaConf.to_container(
            cfg.generator.sampling_params, resolve=True
        ),
        "max_parallel_agents": cfg.dispatcher.max_parallel_agents,
        "runtime_api_url": os.environ["SANDBOX_REMOTE_RUNTIME_API_URL"],
        "runtime_bundle": os.environ["SANDBOX_RUNTIME_BUNDLE_HOST_PATH"],
        "served_model": os.environ["SMOKE_MODEL_NAME"],
        "served_api_url": os.environ["SMOKE_API_URL"],
    }
    print("[collect] resolved config:\n" + json.dumps(resolved, indent=2), flush=True)

    with tempfile.TemporaryDirectory(prefix="skyrl-collect-") as temp_dir:
        config_path = Path(temp_dir) / "resolved_task.yaml"
        OmegaConf.save(cfg, config_path)
        runner = AutoAgentRunner.from_task(
            str(config_path), infer_engine=None, tokenizer=tok
        )
        output = asyncio.run(runner.run(batch, val_mode=False))

    prompt_uid_by_instance = dict(zip(selected_instance_ids, batch_uids))
    for record in output.get("trajectory_records", []):
        record["prompt_uid"] = prompt_uid_by_instance.get(record["instance_id"])
    group_summary = summarize_rloo_groups(output.get("trajectory_records", []))
    output["diagnostic"] = {"resolved_config": resolved, "rloo_summary": group_summary}
    print(
        f"\n[collect] done in {(time.time() - t0) / 60:.1f} min. keys: {list(output.keys())}",
        flush=True,
    )
    print("[collect] trajectory rewards:", output.get("traj_rewards", []), flush=True)
    print(
        "[collect] RLOO summary:",
        json.dumps(
            {key: value for key, value in group_summary.items() if key != "groups"},
            indent=2,
        ),
        flush=True,
    )
    print(
        "[collect] rollout_metrics:",
        json.dumps(output.get("rollout_metrics", {}), indent=2, default=str)[:4000],
        flush=True,
    )
    with open(OUT, "w") as f:
        json.dump(output, f, default=str)
    print(f"[collect] wrote {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()
