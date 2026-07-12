"""Real-RL entrypoint wrapper for the multi-env SWE run.

The task config selects the Hermes agent directly, and the launcher exports the
mounted-runtime settings inherited by Ray workers. This wrapper only ensures the
local RL modules are importable before handing off to skyrl-train.
"""

import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Hand off to the real entrypoint (argv[1:] already holds the Hydra overrides).
runpy.run_module(
    "skyrl_agent.integrations.skyrl_train.skyrl_train_main",
    run_name="__main__",
    alter_sys=True,
)
