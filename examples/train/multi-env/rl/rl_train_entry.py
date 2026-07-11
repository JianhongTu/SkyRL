"""Real-RL entrypoint wrapper for the multi-env SWE run.

``run_skyrl_swe_30b.sh`` invokes THIS instead of ``skyrl_train_main`` directly,
so the two rollout patches the training path needs are active before skyrl-train
starts — without editing skyrl-agent (kept as the reference). Mirrors what
``smoke_rollout.py`` did for the openai_server smoke, minus the OpenAIBackend
shim (real RL uses SkyRLBackend, which already accepts the tokenizer arg):

  1. HermesOHCodeActAgent  — importing ``hermes_codeact_agent`` rebinds
     ``OHCodeActAgent`` so rollouts parse/emit Qwen hermes ``<tool_call>``
     (matches how the model was SFT'd; non-native tool syntax corrupts edits).
  2. mounted remote runtime — force ``SandboxConfig.runtime_mode="mounted"`` so
     the r2e sandbox mounts the prebuilt bundle and builds NO per-instance image
     (``SWEBenchTask`` otherwise defaults to slow image-build mode).

Then it hands off to ``skyrl_agent.integrations.skyrl_train.skyrl_train_main`` as
if ``-m`` had launched it — argv[1:] already holds the CLI overrides, which
``main()`` reads via ``from_cli_overrides(sys.argv[1:])``.
"""

import os
import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# 1. activate the hermes agent subclass (monkeypatch on import)
import hermes_codeact_agent  # noqa: E402,F401  (side effect: rebinds OHCodeActAgent)

# 2. force MOUNTED remote runtime (no per-instance image build)
from skyrl_agent.tasks.swebench import utils as _swe_utils  # noqa: E402

_swe_orig_sbcfg = _swe_utils.get_default_sandbox_config_for_eval


def _swe_mounted_sbcfg():
    cfg = _swe_orig_sbcfg()
    cfg.runtime_mode = "mounted"
    cfg.runtime_bundle_host_path = os.environ.get(
        "SANDBOX_RUNTIME_BUNDLE_HOST_PATH",
        "/home/ec2-user/tovi/openhands-runtime-bundles/d3c99cd",
    )
    cfg.runtime_bundle_container_path = "/opt/openhands-runtime"
    print(
        f"[RL] mounted sandbox cfg: runtime_mode={cfg.runtime_mode} "
        f"bundle={cfg.runtime_bundle_host_path}",
        flush=True,
    )
    return cfg


_swe_utils.get_default_sandbox_config_for_eval = _swe_mounted_sbcfg

os.environ.setdefault("RUNTIME", "remote")  # use the remote runtime, not local docker

# 3. hand off to the real entrypoint (argv[1:] already holds the Hydra overrides)
runpy.run_module(
    "skyrl_agent.integrations.skyrl_train.skyrl_train_main",
    run_name="__main__",
    alter_sys=True,
)
