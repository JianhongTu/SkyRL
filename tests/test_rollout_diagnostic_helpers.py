import ast
import asyncio
import copy
import importlib.util
import re
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_eval_only_disables_colocation_before_starting_inference():
    launcher = (
        ROOT / "examples/train/multi-env/rl/run_skyrl_swe_30b.sh"
    ).read_text()

    assert "COLOCATE_ALL=true" in launcher
    assert re.search(
        r'if \[ "\$EVAL_ONLY" = "true" \]; then\s+'
        r'ENTRYPOINT=\(rl_eval_entry\.py\)\s+'
        r'COLOCATE_ALL=false\s+fi',
        launcher,
    )
    assert "trainer.placement.colocate_all=$COLOCATE_ALL" in launcher

    eval_entrypoint = (
        ROOT / "examples/train/multi-env/rl/rl_eval_entry.py"
    ).read_text()
    assert "if cfg.trainer.placement.colocate_all:" in eval_entrypoint
    assert "Eval-only requires trainer.placement.colocate_all=false" in eval_entrypoint


def test_rl_launcher_moves_openhands_file_store_off_tmp():
    launcher = (
        ROOT / "examples/train/multi-env/rl/run_skyrl_swe_30b.sh"
    ).read_text()

    assert (
        "export OPENHANDS_FILE_STORE_PATH=${OPENHANDS_FILE_STORE_PATH:-"
        "/data/tovi/openhands_file_store}"
    ) in launcher


def test_multi_env_rl_uses_uniform_temperature():
    launcher = (
        ROOT / "examples/train/multi-env/rl/run_skyrl_swe_30b.sh"
    ).read_text()
    assert "generator.sampling_params.temperature=${TEMP:-1.0}" in launcher
    assert "generator.eval_sampling_params.temperature=${EVAL_TEMP:-1.0}" in launcher

    task_config = (
        ROOT / "examples/train/multi-env/rl/skyrl_swe_30b.yaml"
    ).read_text()
    assert len(re.findall(r"^    temperature: 1\.0$", task_config, re.MULTILINE)) == 1
    assert len(re.findall(r"^      temperature: 1\.0$", task_config, re.MULTILINE)) == 1


def test_multi_env_rl_defaults_to_final_intact_sft_checkpoint():
    launcher = (
        ROOT / "examples/train/multi-env/rl/run_skyrl_swe_30b.sh"
    ).read_text()

    assert (
        "MODEL=${MODEL:-/data/tovi/exports/"
        "skyrl_sft_openhands_hf_harnessprompt_intact/global_step_496/policy}"
        in launcher
    )

    wrapper = (
        ROOT / "examples/train/multi-env/run_rl_recipe.sh"
    ).read_text()
    assert (
        "MODEL=/data/tovi/exports/"
        "skyrl_sft_openhands_hf_harnessprompt_intact/global_step_496/policy"
        in wrapper
    )


def test_multi_env_rl_defaults_to_repo_env_file():
    launcher = (
        ROOT / "examples/train/multi-env/rl/run_skyrl_swe_30b.sh"
    ).read_text()

    assert "ENV_FILE=${ENV_FILE:-$REPO/.env}" in launcher

    wrapper = (
        ROOT / "examples/train/multi-env/run_rl_recipe.sh"
    ).read_text()
    assert 'ENV_FILE="$REPO/.env"' in wrapper
    assert '$RL_DIR/.env' not in wrapper

    readme = (ROOT / "examples/train/multi-env/README.md").read_text()
    assert '--env-file "$(git rev-parse --show-toplevel)/.env"' in readme


def test_multi_env_rl_defaults_to_prebuilt_venv():
    launcher = (
        ROOT / "examples/train/multi-env/rl/run_skyrl_swe_30b.sh"
    ).read_text()

    assert 'if [ "${USE_PREBUILT_VENV:-1}" = "1" ]; then' in launcher
    assert "unset RAY_RUNTIME_ENV_HOOK" in launcher
    assert 'LAUNCH=("$RL_DIR/.venv/bin/python" "${ENTRYPOINT[@]}")' in launcher


def test_multi_env_rl_defaults_to_64_prompt_batch():
    launcher = (
        ROOT / "examples/train/multi-env/rl/run_skyrl_swe_30b.sh"
    ).read_text()

    assert "BATCH_SIZE=${BATCH_SIZE:-64}" in launcher


def test_multi_env_rl_uses_native_context_window_split():
    prompt_tokens = 30_720
    generation_tokens = 2_048
    assert prompt_tokens + generation_tokens == 32_768

    for relative_path in (
        "examples/train/multi-env/rl/skyrl_swe_30b.yaml",
        "examples/train/multi-env/rl/skyrl_swe_30b_smoke.yaml",
    ):
        source = (ROOT / relative_path).read_text()
        assert re.search(r"^  max_prompt_length: 30720$", source, re.MULTILINE)
        assert len(re.findall(r"^\s+max_tokens: 2048$", source, re.MULTILINE)) == 2

    task_config = (
        ROOT / "examples/train/multi-env/rl/skyrl_swe_30b.yaml"
    ).read_text()
    assert task_config.count('stop: ["</tool_call>"]') == 2

    launcher = (
        ROOT / "examples/train/multi-env/rl/run_skyrl_swe_30b.sh"
    ).read_text()
    assert "trainer.max_prompt_length=${MAX_PROMPT_LEN:-30720}" in launcher
    assert "generator.sampling_params.max_generate_length=2048" in launcher
    assert "generator.eval_sampling_params.max_generate_length=2048" in launcher


def test_rl_recipe_enforces_validated_runtime_limits():
    task_config = (
        ROOT / "examples/train/multi-env/rl/skyrl_swe_30b.yaml"
    ).read_text()
    assert re.search(r"^  max_parallel_agents: 32$", task_config, re.MULTILINE)
    assert re.search(r"^  max_eval_parallel_agents: 32$", task_config, re.MULTILINE)

    launcher = (
        ROOT / "examples/train/multi-env/rl/run_skyrl_swe_30b.sh"
    ).read_text()
    assert "NOFILE_LIMIT=${NOFILE_LIMIT:-524288}" in launcher
    assert 'ulimit -n "$NOFILE_LIMIT"' in launcher
    assert 'mktemp "$_d/.skyrl-write-test.XXXXXX"' in launcher
    assert "EVAL_INTERVAL=${EVAL_INTERVAL:-10}" in launcher
    assert "EVAL_BEFORE_TRAIN=${EVAL_BEFORE_TRAIN:-false}" in launcher
    assert "CKPT_INTERVAL=${CKPT_INTERVAL:-20}" in launcher
    assert 'VAL_DATA=${VAL_DATA:-"${DATA_DIR}/validation.parquet"}' in launcher

    smoke_launcher = (
        ROOT / "examples/train/multi-env/rl/run_skyrl_swe_30b_smoke.sh"
    ).read_text()
    assert "BATCH_SIZE=${BATCH_SIZE:-8}" in smoke_launcher
    assert "MAX_TRAINING_STEPS=${MAX_TRAINING_STEPS:-2}" in smoke_launcher
    assert "EVAL_INTERVAL=${EVAL_INTERVAL:-9999}" in smoke_launcher
    assert "CKPT_INTERVAL=${CKPT_INTERVAL:-1}" in smoke_launcher
    assert 'trainer.max_training_steps="$MAX_TRAINING_STEPS"' in smoke_launcher

    assert "/opt/openhands-runtime/current" in launcher


def test_rl_recipe_activates_hermes_and_mounted_runtime_in_ray_worker():
    task_config = (
        ROOT / "examples/train/multi-env/rl/skyrl_swe_30b.yaml"
    ).read_text()
    assert "agent_cls: hermes_codeact_agent.HermesOHCodeActAgent" in task_config

    mapping = (
        ROOT / "skyrl-agent/skyrl_agent/agents/mapping.py"
    ).read_text()
    assert mapping.count('"hermes_codeact_agent.HermesOHCodeActAgent"') == 2

    runner = (
        ROOT / "skyrl-agent/skyrl_agent/agents/oh_codeact/codeact_runner.py"
    ).read_text()
    assert "self.agent = self.agent_cls(" in runner

    launcher = (
        ROOT / "examples/train/multi-env/rl/run_skyrl_swe_30b.sh"
    ).read_text()
    assert "SKYRL_PYTHONPATH_EXPORT=${SKYRL_PYTHONPATH_EXPORT:-1}" in launcher
    assert "SANDBOX_RUNTIME_MODE=${SANDBOX_RUNTIME_MODE:-mounted}" in launcher
    assert 'SANDBOX_RUNTIME_BUNDLE_HOST_PATH:-/opt/openhands-runtime/current' in launcher

    ray_runtime = (ROOT / "skyrl/train/utils/utils.py").read_text()
    for name in (
        "SANDBOX_RUNTIME_MODE",
        "SANDBOX_RUNTIME_BUNDLE_HOST_PATH",
        "SANDBOX_RUNTIME_BUNDLE_CONTAINER_PATH",
    ):
        assert name in ray_runtime

    swe_utils = (
        ROOT / "skyrl-agent/skyrl_agent/tasks/swebench/utils.py"
    ).read_text()
    assert 'os.environ.get("SANDBOX_RUNTIME_MODE")' in swe_utils
    assert "runtime_bundle_host_path" in swe_utils


def test_swebench_mounted_runtime_config_from_env():
    source_path = ROOT / "skyrl-agent/skyrl_agent/tasks/swebench/utils.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_default_sandbox_config_for_eval"
    )

    class SandboxConfig(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    namespace = {
        "SandboxConfig": SandboxConfig,
        "os": SimpleNamespace(
            environ={
                "ALLHANDS_API_KEY": "test-key",
                "SANDBOX_REMOTE_RUNTIME_API_URL": "http://runtime:3000",
                "SANDBOX_RUNTIME_MODE": "mounted",
                "SANDBOX_RUNTIME_BUNDLE_HOST_PATH": "/opt/openhands-runtime/current",
            }
        ),
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"
        ),
        namespace,
    )

    config = namespace["get_default_sandbox_config_for_eval"]()

    assert config.runtime_mode == "mounted"
    assert config.runtime_bundle_host_path == "/opt/openhands-runtime/current"
    assert config.runtime_bundle_container_path == "/opt/openhands-runtime"

    del namespace["os"].environ["SANDBOX_RUNTIME_BUNDLE_HOST_PATH"]
    try:
        namespace["get_default_sandbox_config_for_eval"]()
    except ValueError as exc:
        assert "SANDBOX_RUNTIME_BUNDLE_HOST_PATH must be set" in str(exc)
    else:
        raise AssertionError("Expected mounted runtime configuration to require a host path")


def test_swebench_uses_separate_rollout_and_scoring_images():
    source_path = ROOT / "skyrl-agent/skyrl_agent/tasks/swebench/utils.py"
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"get_instance_docker_image", "get_swebench_scoring_image"}
    }
    namespace = {"DOCKER_IMAGE_PREFIX": "docker.io/xingyaoww/"}
    exec(
        compile(
            ast.Module(body=list(functions.values()), type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    instance = {"instance_id": "astropy__astropy-13033"}

    assert namespace["get_instance_docker_image"](instance, "swe-bench") == (
        "docker.io/xingyaoww/sweb.eval.x86_64.astropy_s_astropy-13033"
    )
    assert namespace["get_swebench_scoring_image"](instance) == (
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-13033"
    )


def test_swebench_disables_jupyter_only_for_scoring():
    source_path = ROOT / "skyrl-agent/skyrl_agent/tasks/swebench/utils.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SWEBenchTask"
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_config"
    )

    class AppConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.agent_config = SimpleNamespace(enable_jupyter=True)

        def get_agent_config(self):
            return self.agent_config

        def set_agent_config(self, agent_config):
            self.agent_config = agent_config

    class SandboxConfig(SimpleNamespace):
        pass

    minimal_class = ast.ClassDef(
        name="SWEBenchTask",
        bases=[],
        keywords=[],
        body=[method_node],
        decorator_list=[],
    )
    ast.fix_missing_locations(minimal_class)
    namespace = {
        "AppConfig": AppConfig,
        "get_default_sandbox_config_for_eval": SandboxConfig,
        "get_instance_docker_image": lambda _instance, _source: "rollout-image",
        "get_swebench_scoring_image": lambda _instance: "scoring-image",
        "logger": SimpleNamespace(info=lambda *_args, **_kwargs: None),
        "os": SimpleNamespace(
            environ={
                "USE_INSTANCE_IMAGE": "true",
                "OPENHANDS_FILE_STORE_PATH": "/data/tovi/openhands_file_store",
            }
        ),
    }
    exec(
        compile(
            ast.Module(body=[minimal_class], type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )

    rollout_config = namespace["SWEBenchTask"].get_config({}, "swe-bench")
    scoring_config = namespace["SWEBenchTask"].get_config({}, "swe-bench", scoring=True)

    assert rollout_config.get_agent_config().enable_jupyter is True
    assert scoring_config.get_agent_config().enable_jupyter is False
    assert rollout_config.kwargs["file_store_path"] == "/data/tovi/openhands_file_store"
    assert scoring_config.kwargs["file_store_path"] == "/data/tovi/openhands_file_store"


def test_swebench_evaluator_cleanup_preserves_startup_error():
    source_path = ROOT / "skyrl-agent/skyrl_agent/tasks/swebench/utils.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SWEBenchTask"
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "evaluate_result"
    )

    class Runtime:
        def __init__(self):
            self.close_attempted = False
            self.event_stream = SimpleNamespace(close=lambda: None)

        async def connect(self):
            raise RuntimeError("startup failed")

        def close(self):
            self.close_attempted = True
            raise RuntimeError("cleanup failed")

    runtime = Runtime()

    class Logger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    minimal_class = ast.ClassDef(
        name="SWEBenchTask",
        bases=[],
        keywords=[],
        body=[method_node],
        decorator_list=[],
    )
    ast.fix_missing_locations(minimal_class)
    namespace = {
        "create_runtime": lambda _config: runtime,
        "logger": Logger(),
    }
    exec(
        compile(
            ast.Module(body=[minimal_class], type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    namespace["SWEBenchTask"].get_config = classmethod(
        lambda _cls, _instance, _dataset, scoring=False: SimpleNamespace(scoring=scoring)
    )

    with pytest.raises(RuntimeError, match="startup failed"):
        asyncio.run(
            namespace["SWEBenchTask"].evaluate_result(
                instance={},
                run_results={},
                instance_id="issue",
                trajectory_id=0,
                dataset="swe-bench",
            )
        )

    assert runtime.close_attempted


def test_generator_validation_accepts_missing_per_sample_logprobs():
    source_path = (
        ROOT / "skyrl-agent/skyrl_agent/integrations/skyrl_train/trainer.py"
    )
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_generator_output"
    )

    class Array:
        def sum(self):
            return 1

    namespace = {
        "GeneratorInput": dict,
        "GeneratorOutput": dict,
        "logger": SimpleNamespace(warning=lambda *_args, **_kwargs: None),
        "np": SimpleNamespace(concatenate=lambda _values: Array()),
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"
        ),
        namespace,
    )

    namespace["validate_generator_output"](
        {},
        {
            "prompt_token_ids": [[0]],
            "response_ids": [[1]],
            "loss_masks": [[1]],
            "rewards": [0.0],
            "rollout_logprobs": [None],
        },
    )


def test_agent_trainer_honors_max_training_steps():
    source_path = (
        ROOT / "skyrl-agent/skyrl_agent/integrations/skyrl_train/trainer.py"
    )
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    trainer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SkyRLAgentPPOTrainer"
    )
    init_method = next(
        node
        for node in trainer_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    train_method = next(
        node
        for node in trainer_class.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "train"
    )
    init_source = ast.unparse(init_method)
    train_source = ast.unparse(train_method)

    assert "min(self.total_training_steps, self.cfg.trainer.max_training_steps)" in init_source
    assert "stop_training = False" in train_source
    assert "self.global_step > self.cfg.trainer.max_training_steps" in train_source
    assert "if stop_training:" in train_source


def test_skyrl_backend_reuses_shared_inference_client():
    source_path = (
        ROOT
        / "skyrl-agent/skyrl_agent/integrations/skyrl_train/skyrl_train_backend.py"
    )
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    backend_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SkyRLBackend"
    )
    class AsyncInferBackend:
        pass

    class Client:
        def __init__(self):
            self.calls = 0

        async def generate(self, _input_obj, model):
            assert model == "policy"
            self.calls += 1
            return {
                "responses": ["ok"],
                "response_ids": [[1, 2]],
                "stop_reasons": ["stop"],
            }

    namespace = {
        "Any": object,
        "List": list,
        "AsyncInferBackend": AsyncInferBackend,
        "resolve_policy_model_name": lambda _cfg: "policy",
    }
    exec(
        compile(
            ast.Module(body=[backend_class], type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    backend = namespace["SkyRLBackend"].__new__(namespace["SkyRLBackend"])
    backend.policy_model_name = "policy"
    backend.client = Client()

    response, metadata = asyncio.run(
        backend.async_generate_ids([0], {"max_tokens": 2})
    )

    assert response == "ok"
    assert metadata["output_tokens"] == [1, 2]
    assert backend.client.calls == 1


def test_sync_to_async_bridge_uses_one_persistent_event_loop():
    source_path = ROOT / "skyrl-agent/skyrl_agent/dispatcher/async_utils.py"
    source = source_path.read_text()

    assert "self._submissions.put" in source
    assert "asyncio.run(arun())" not in source
    assert "asyncio.new_event_loop()" in source

    spec = importlib.util.spec_from_file_location("test_async_utils", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}
    loop_ids = set()
    results = []
    errors = []

    async def probe():
        with lock:
            loop_ids.add(id(asyncio.get_running_loop()))
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.05)
        with lock:
            state["active"] -= 1
        return "ok"

    def invoke():
        try:
            results.append(module.call_async_from_sync(probe, 2))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=invoke) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert results == ["ok"] * 8
    assert len(loop_ids) == 1
    assert state["peak"] > 1


def test_colocated_startup_closes_session_before_event_loop_exit():
    source = (ROOT / "skyrl/train/entrypoints/main_base.py").read_text()
    sleep = source.index("await client.sleep()")
    close = source.index("await client.aclose()", sleep)
    run = source.index("asyncio.run(sleep_inference_client())", close)

    assert sleep < close < run


def test_diagnostic_parser_describes_iteration_cap_as_trainable():
    source = (
        ROOT / "examples/train/multi-env/rl/diagnostics/parse_log.py"
    ).read_text()

    assert "iteration cap (prefix remains trainable)" in source
    assert "max_iterations -> MASKED OUT" not in source


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_generation_limit_preserves_configured_cap():
    helpers = _load_module(
        "rollout_diagnostic_utils",
        "skyrl-agent/skyrl_agent/agents/rollout_diagnostic_utils.py",
    )

    assert helpers.bounded_max_tokens(configured=4000, remaining=28000) == 4000
    assert helpers.bounded_max_tokens(configured=4000, remaining=2500) == 2500


def test_native_model_budget_uses_full_encoded_input_length():
    helpers = _load_module(
        "rollout_native_budget_utils",
        "skyrl-agent/skyrl_agent/agents/rollout_diagnostic_utils.py",
    )

    assert helpers.remaining_generation_tokens(30_720, 32_768) == 2_048
    assert helpers.remaining_generation_tokens(32_000, 32_768) == 768
    assert helpers.remaining_generation_tokens(32_768, 32_768) == 0
    assert helpers.remaining_generation_tokens(33_000, 32_768) == 0


def test_context_terminal_masking_policy():
    helpers = _load_module(
        "rollout_context_terminal_utils",
        "skyrl-agent/skyrl_agent/agents/rollout_diagnostic_utils.py",
    )

    assert "CONTEXT_BUDGET_REACHED" in helpers.NON_FINISH_TERMINAL_REASONS
    assert "CONTEXT_BUDGET_REACHED" not in helpers.MASK_OUT_REASONS
    assert "TRUNCATED_RESPONSE" in helpers.NON_FINISH_TERMINAL_REASONS
    assert "TRUNCATED_RESPONSE" in helpers.MASK_OUT_REASONS


@pytest.mark.parametrize(
    "content",
    [
        '<tool_call>{"name":"finish","arguments":{}}</tool_call>',
        '<tool_call>\n{"arguments": {}, "name": "finish"}\n</tool_call>',
        "<function=finish>done</function>",
    ],
)
def test_finish_detection_accepts_hermes_and_legacy(content):
    helpers = _load_module(
        "rollout_diagnostic_utils",
        "skyrl-agent/skyrl_agent/agents/rollout_diagnostic_utils.py",
    )

    assert helpers.contains_finish_call(content)


@pytest.mark.parametrize(
    "content",
    [
        '<tool_call>{"name":"execute_bash","arguments":{}}</tool_call>',
        '<tool_call>{"name":</tool_call>',
        "ordinary assistant prose",
    ],
)
def test_finish_detection_rejects_non_finish_and_malformed_calls(content):
    helpers = _load_module(
        "rollout_diagnostic_utils",
        "skyrl-agent/skyrl_agent/agents/rollout_diagnostic_utils.py",
    )

    assert not helpers.contains_finish_call(content)


def test_finish_reason_is_normalized_before_masking():
    helpers = _load_module(
        "rollout_diagnostic_utils",
        "skyrl-agent/skyrl_agent/agents/rollout_diagnostic_utils.py",
    )

    hermes = [
        {
            "role": "assistant",
            "content": '<tool_call>{"name":"finish","arguments":{}}</tool_call>',
        }
    ]
    legacy = [{"role": "assistant", "content": "<function=finish>done</function>"}]
    missing = [
        {
            "role": "assistant",
            "content": '<tool_call>{"name":"think","arguments":{}}</tool_call>',
        }
    ]

    assert helpers.normalize_finish_reason(hermes, "FINISH_TOOL") == "FINISH_TOOL"
    assert helpers.normalize_finish_reason(legacy, "FINISH_TOOL") == "FINISH_TOOL"
    assert helpers.normalize_finish_reason(missing, "FINISH_TOOL") == "BAD_LLM_RESPONSE"
    assert (
        helpers.normalize_finish_reason(
            [{"role": "user", "content": "tool output"}], "FINISH_TOOL"
        )
        == "error_runtime"
    )
    assert (
        helpers.normalize_finish_reason(missing, "max_iterations_reached")
        == "max_iterations_reached"
    )


def test_iteration_cap_is_trainable_but_still_terminal():
    helpers = _load_module(
        "rollout_diagnostic_utils",
        "skyrl-agent/skyrl_agent/agents/rollout_diagnostic_utils.py",
    )
    messages = [
        {"role": "assistant", "content": "unfinished but valid sampled behavior"}
    ]

    assert "max_iterations_reached" not in helpers.MASK_OUT_REASONS
    assert (
        helpers.normalize_finish_reason(messages, "max_iterations_reached")
        == "max_iterations_reached"
    )


def test_finish_bonus_is_small_training_only_shaping():
    helpers = _load_module(
        "rollout_diagnostic_utils",
        "skyrl-agent/skyrl_agent/agents/rollout_diagnostic_utils.py",
    )

    assert helpers.apply_finish_reward_bonus(0.0, "FINISH_TOOL", training=True) == (
        0.05,
        0.05,
    )
    assert helpers.apply_finish_reward_bonus(1.0, "FINISH_TOOL", training=True) == (
        1.05,
        0.05,
    )
    assert helpers.apply_finish_reward_bonus(
        0.0, "max_iterations_reached", training=True
    ) == (0.0, 0.0)
    assert helpers.apply_finish_reward_bonus(1.0, "FINISH_TOOL", training=False) == (
        1.0,
        0.0,
    )


def test_exact_output_tokens_require_server_token_ids():
    helpers = _load_module(
        "exact_output_token_utils",
        "skyrl-agent/skyrl_agent/agents/rollout_diagnostic_utils.py",
    )

    assert helpers.extract_exact_output_tokens({"token_ids": [1, 2, 3]}) == [1, 2, 3]
    with pytest.raises(RuntimeError, match="token_ids"):
        helpers.extract_exact_output_tokens({"text": "decoded only"})

    backend_source = (
        ROOT / "skyrl-agent/skyrl_agent/integrations/openai.py"
    ).read_text()
    diagnostic_source = (
        ROOT / "examples/train/multi-env/rl/diagnostics/collect_batch.py"
    ).read_text()
    assert 'payload["return_token_ids"] = True' in backend_source
    assert "extract_exact_output_tokens(choice)" in backend_source
    assert '"require_token_ids": True' in diagnostic_source


def test_hermes_generation_keeps_trajectory_level_training_contract():
    source_path = ROOT / "examples/train/multi-env/rl/hermes_codeact_agent.py"
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HermesOHCodeActAgent"
    )
    step = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    )

    assert "record_transition" not in source
    assert "self.transitions" not in source
    assert any(
        isinstance(node, ast.Attribute)
        and node.attr == "async_generate_ids"
        for node in ast.walk(step)
    )
    assert "if len(input_ids) > self.max_prompt_length:" in source
    assert 'thought="CONTEXT_BUDGET_REACHED"' in source
    assert "self.messages.pop()" in source
    assert 'thought="TRUNCATED_RESPONSE"' in source

    backend = (
        ROOT
        / "skyrl-agent/skyrl_agent/integrations/skyrl_train/skyrl_train_backend.py"
    ).read_text()
    assert "replace(" not in backend
    assert "await client.aclose()" not in backend
    assert source.index("tool_calls, thought = self._parse_hermes_tool_calls") < source.index(
        'if stop_reason == "length":'
    )


def test_transition_data_preserves_exact_tokens_and_action_mask():
    utils = _load_module(
        "skyrl_agent_functional_utils",
        "skyrl-agent/skyrl_agent/functional/utils.py",
    )
    transitions = [
        utils.Transition(
            ob=utils.Observation(input_ids=[1, 2]),
            ac=utils.TokensWithLogprobs(token_ids=[3]),
            reward=0.0,
            episode_done=False,
        ),
        utils.Transition(
            ob=utils.Observation(input_ids=[1, 2, 3, 4]),
            ac=utils.TokensWithLogprobs(token_ids=[5]),
            reward=0.0,
            episode_done=False,
        ),
        utils.Transition(
            ob=utils.Observation(input_ids=[1, 2, 3, 4, 5, 6]),
            ac=utils.TokensWithLogprobs(token_ids=[7, 8]),
            reward=0.0,
            episode_done=False,
            metrics={"trainable": False},
        ),
    ]

    data = utils.transitions_to_training_data(transitions)

    assert len(data) == 1
    assert data[0].input_tokens == [1, 2]
    assert data[0].response_tokens == [3, 4, 5, 6, 7, 8]
    assert data[0].response_mask == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_postprocess_keeps_cap_signal_and_separates_finish_bonus():
    helpers = _load_module(
        "postprocess_rollout_utils",
        "skyrl-agent/skyrl_agent/agents/rollout_diagnostic_utils.py",
    )
    functional = _load_module(
        "postprocess_functional_utils",
        "skyrl-agent/skyrl_agent/functional/utils.py",
    )
    source_path = ROOT / "skyrl-agent/skyrl_agent/agents/base.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AgentRunner"
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_post_process_results"
    )

    class Logger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    class OmegaConf:
        @staticmethod
        def select(*_args, **_kwargs):
            return None

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if kwargs.get("return_assistant_tokens_mask"):
                return {
                    "input_ids": [[20, 21] for _ in messages],
                    "assistant_masks": [[1, 1] for _ in messages],
                }
            return {"input_ids": [[10] for _ in messages]}

    namespace = {
        "Any": object,
        "Dict": dict,
        "defaultdict": defaultdict,
        "copy": copy,
        "pd": SimpleNamespace(Series=lambda value: dict(value)),
        "logger": Logger(),
        "OmegaConf": OmegaConf,
        "MASK_OUT_REASONS": helpers.MASK_OUT_REASONS,
        "PREFIX_TRAINABLE_TERMINAL_REASONS": (
            helpers.PREFIX_TRAINABLE_TERMINAL_REASONS
        ),
        "normalize_finish_reason": helpers.normalize_finish_reason,
        "apply_finish_reward_bonus": helpers.apply_finish_reward_bonus,
        "transitions_to_training_data": functional.transitions_to_training_data,
        "chat_template": None,
        "chat_template_qwen3_thinking": None,
    }
    exec(
        compile(
            ast.Module(body=[method_node], type_ignores=[]), str(source_path), "exec"
        ),
        namespace,
    )

    def transition(seed):
        return functional.Transition(
            ob=functional.Observation(input_ids=[seed]),
            ac=functional.TokensWithLogprobs(token_ids=[seed + 1]),
            reward=0.0,
            episode_done=False,
            metrics={
                "finish_reason": "stop",
                "generation_seconds": 0.1,
                "output_token_count": 1,
            },
        )

    finish_result = {
        "instance_id": "issue",
        "trajectory_id": 0,
        "messages": [
            {
                "role": "assistant",
                "content": '<tool_call>{"name":"finish","arguments":{}}</tool_call>',
            }
        ],
        "transitions": [transition(1)],
        "reward": 0.0,
        "finish": True,
        "finish_reason": "FINISH_TOOL",
    }
    capped_result = {
        "instance_id": "issue",
        "trajectory_id": 1,
        "messages": [{"role": "assistant", "content": "valid unfinished response"}],
        "transitions": [transition(3)],
        "reward": 1.0,
        "finish": False,
        "finish_reason": "max_iterations_reached",
    }
    legacy_error_result = {
        "instance_id": "issue",
        "trajectory_id": 2,
        "messages": [
            {"role": "user", "content": "problem"},
            {"role": "assistant", "content": "partial response"},
        ],
        "transitions": [],
        "reward": 0.0,
        "finish": False,
        "finish_reason": "error_runtime",
    }
    split_result = {
        "instance_id": "issue",
        "trajectory_id": 3,
        "messages": [{"role": "assistant", "content": "unfinished"}],
        "transitions": [
            functional.Transition(
                ob=functional.Observation(input_ids=[5]),
                ac=functional.TokensWithLogprobs(token_ids=[6]),
                reward=0.0,
                episode_done=False,
            ),
            functional.Transition(
                ob=functional.Observation(input_ids=[8]),
                ac=functional.TokensWithLogprobs(token_ids=[9, 10]),
                reward=0.0,
                episode_done=False,
            ),
        ],
        "reward": 0.0,
        "finish": False,
        "finish_reason": "max_iterations_reached",
    }
    runner = SimpleNamespace(
        cfg=SimpleNamespace(
            generator=SimpleNamespace(
                num_trajectories=4,
                val_config=SimpleNamespace(num_trajectories=1),
                max_prompt_length=1,
                remove_think_tokens=False,
            )
        ),
        trajectories={
            "issue": {
                0: SimpleNamespace(result=finish_result),
                1: SimpleNamespace(result=capped_result),
                2: SimpleNamespace(result=legacy_error_result),
                3: SimpleNamespace(result=split_result),
            }
        },
        batch=[
            {
                "instance": {"instance_id": "issue"},
                "instance_id": "issue",
                "data_source": "r2e-gym",
            }
        ],
        _get_data=lambda content: content,
        tokenizer=Tokenizer(),
    )

    output = namespace["_post_process_results"](runner, val_mode=False)

    assert output["response_ids"] == [[2], [4], [20], [6], [9]]
    assert output["rewards"] == [0.05, 1.0, 0.0, 0.0, 0.0]
    assert output["task_rewards"] == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert output["loss_masks"] == [[1.0], [1.0], [0], [0.0], [0.0]]
    assert output["trajectory_records"][0]["finish_reward_bonus"] == 0.05
    assert output["trajectory_records"][1]["finish_reason"] == "max_iterations_reached"
    assert output["trajectory_records"][3]["finish_reason"] == (
        "CONTEXT_WINDOW_EXCEEDED"
    )
    rollout_metrics = output["rollout_metrics"]
    assert rollout_metrics["rollout_metrics/reward_mean/finish_tool"] == 0.0
    assert rollout_metrics["rollout_metrics/positive_reward_rate/finish_tool"] == 0.0
    assert rollout_metrics["rollout_metrics/reward_sum/finish_tool"] == 0.0
    assert rollout_metrics["rollout_metrics/positive_reward_count/finish_tool"] == 0
    assert rollout_metrics["rollout_metrics/trajectory_count/finish_tool"] == 1
    assert rollout_metrics["rollout_metrics/reward_mean/truncated_response"] == 0.0
    assert rollout_metrics["rollout_metrics/positive_reward_rate/truncated_response"] == 0.0
    assert rollout_metrics["rollout_metrics/reward_mean/context_budget_reached"] == 0.0
    assert rollout_metrics["rollout_metrics/positive_reward_rate/context_budget_reached"] == 0.0

    legacy_runner = SimpleNamespace(
        cfg=SimpleNamespace(
            generator=SimpleNamespace(
                num_trajectories=1,
                val_config=SimpleNamespace(num_trajectories=1),
                max_prompt_length=10,
                remove_think_tokens=False,
            )
        ),
        trajectories={"issue": {0: SimpleNamespace(result=legacy_error_result)}},
        batch=runner.batch,
        _get_data=runner._get_data,
        tokenizer=Tokenizer(),
    )
    legacy_output = namespace["_post_process_results"](legacy_runner, val_mode=False)
    assert legacy_output["rollout_logprobs"] is None

    budget_result = {
        "instance_id": "issue",
        "trajectory_id": 0,
        "messages": [{"role": "assistant", "content": "valid prefix"}],
        "transitions": [transition(11)],
        "reward": 1.0,
        "finish": True,
        "finish_reason": "CONTEXT_BUDGET_REACHED",
    }
    truncated_result = {
        "instance_id": "issue",
        "trajectory_id": 1,
        "messages": [{"role": "assistant", "content": "partial tool call"}],
        "transitions": [
            transition(21),
            functional.Transition(
                ob=functional.Observation(input_ids=[21, 22, 23]),
                ac=functional.TokensWithLogprobs(token_ids=[24, 25]),
                reward=0.0,
                episode_done=False,
                metrics={"trainable": False},
            ),
        ],
        "reward": 1.0,
        "finish": True,
        "finish_reason": "TRUNCATED_RESPONSE",
    }
    prefix_runner = SimpleNamespace(
        cfg=SimpleNamespace(
            generator=SimpleNamespace(
                num_trajectories=2,
                val_config=SimpleNamespace(num_trajectories=1),
                max_prompt_length=10,
                remove_think_tokens=False,
            )
        ),
        trajectories={
            "issue": {
                0: SimpleNamespace(result=budget_result),
                1: SimpleNamespace(result=truncated_result),
            }
        },
        batch=runner.batch,
        _get_data=runner._get_data,
        tokenizer=Tokenizer(),
    )

    prefix_output = namespace["_post_process_results"](
        prefix_runner, val_mode=False
    )

    assert prefix_output["response_ids"] == [[12], [22, 23, 24, 25]]
    assert prefix_output["loss_masks"] == [[1.0], [0, 0, 0, 0]]
    assert [
        record["finish_reason"] for record in prefix_output["trajectory_records"]
    ] == ["CONTEXT_BUDGET_REACHED", "TRUNCATED_RESPONSE"]
    rollout_metrics = prefix_output["rollout_metrics"]
    assert rollout_metrics["rollout_metrics/num_mask_out"] == 1
    assert rollout_metrics["rollout_metrics/num_mask_non_zero_reward"] == 1
    assert rollout_metrics["rollout_metrics/reward_mean/finish_tool"] == 0.0
    assert rollout_metrics["rollout_metrics/positive_reward_rate/finish_tool"] == 0.0
    assert rollout_metrics["rollout_metrics/reward_mean/truncated_response"] == 1.0
    assert rollout_metrics["rollout_metrics/positive_reward_rate/truncated_response"] == 1.0
    assert rollout_metrics["rollout_metrics/reward_sum/truncated_response"] == 1.0
    assert rollout_metrics["rollout_metrics/positive_reward_count/truncated_response"] == 1
    assert rollout_metrics["rollout_metrics/trajectory_count/truncated_response"] == 1
    assert rollout_metrics["rollout_metrics/reward_mean/context_budget_reached"] == 1.0
    assert rollout_metrics["rollout_metrics/positive_reward_rate/context_budget_reached"] == 1.0


def test_inline_evaluation_propagates_finish_reason():
    source_path = ROOT / "skyrl-agent/skyrl_agent/agents/oh_codeact/codeact_runner.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CodeActTrajectory"
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "evaluate_trajectory"
    )

    class SWEBenchTask:
        pass

    class Logger:
        def info(self, *_args, **_kwargs):
            pass

        def error(self, *_args, **_kwargs):
            pass

    namespace = {
        "SWEBenchTask": SWEBenchTask,
        "pd": SimpleNamespace(Series=lambda value: value),
        "time": time,
        "logger": Logger(),
    }
    exec(
        compile(
            ast.Module(body=[method_node], type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    trajectory = SimpleNamespace(
        task=SWEBenchTask(),
        cfg=SimpleNamespace(instance_id="issue", trajectory_id=0),
        data={
            "instance_id": "issue",
            "instance": {},
            "data_source": "r2e-gym",
        },
        result={
            "finish_reason": "FINISH_TOOL",
            "results": {
                "reward": 0,
                "finish_reason": "error_evaluation",
                "evaluation_error": "run_tests_timeout",
                "reward_evaluation_seconds": 600.0,
            },
        },
    )

    asyncio.run(namespace["evaluate_trajectory"](trajectory))

    assert trajectory.result["finish_reason"] == "error_evaluation"
    assert trajectory.result["eval_error"] == "run_tests_timeout"


def test_filter_preserves_trajectory_records_for_step_rows():
    source_path = ROOT / "skyrl/train/utils/trainer_utils.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "filter_generator_output"
    )
    namespace = {"GeneratorOutput": dict, "List": list}
    exec(
        compile(
            ast.Module(body=[function_node], type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    output = {
        "prompt_token_ids": [[1], [2], [3]],
        "response_ids": [[4], [5], [6]],
        "rewards": [1.0, 1.0, 0.0],
        "task_rewards": [1.0, 1.0, 0.0],
        "loss_masks": [[1], [1], [1]],
        "stop_reasons": None,
        "rollout_logprobs": None,
        "rollout_metrics": {},
        "traj_idx": ["a-traj0", "a-traj0", "b-traj0"],
        "trajectory_records": [
            {"instance_id": "a", "trajectory_id": 0},
            {"instance_id": "b", "trajectory_id": 0},
        ],
    }

    filtered = namespace["filter_generator_output"](output, [0, 1])

    assert filtered["traj_idx"] == ["a-traj0", "a-traj0"]
    assert filtered["trajectory_records"] == [{"instance_id": "a", "trajectory_id": 0}]


def test_group_summary_reports_rloo_signal_and_masking():
    analysis = _load_module(
        "rollout_diagnostic_analysis",
        "examples/train/multi-env/rl/diagnostics/analysis.py",
    )
    records = [
        {
            "instance_id": "a",
            "trajectory_id": 0,
            "reward": 1,
            "loss_mask_nonzero": True,
        },
        {
            "instance_id": "a",
            "trajectory_id": 1,
            "reward": 1,
            "loss_mask_nonzero": True,
        },
        {
            "instance_id": "a",
            "trajectory_id": 2,
            "reward": 0,
            "loss_mask_nonzero": False,
        },
        {
            "instance_id": "a",
            "trajectory_id": 3,
            "reward": 0,
            "loss_mask_nonzero": True,
        },
        {
            "instance_id": "b",
            "trajectory_id": 0,
            "reward": 0,
            "loss_mask_nonzero": True,
        },
        {
            "instance_id": "b",
            "trajectory_id": 1,
            "reward": 0,
            "loss_mask_nonzero": True,
        },
    ]

    summary = analysis.summarize_rloo_groups(records)
    group_a, group_b = summary["groups"]

    assert group_a["instance_id"] == "a"
    assert group_a["success_count"] == 2
    assert group_a["masked_count"] == 1
    assert group_a["zero_reward_variance"] is False
    assert group_a["loo_advantages"] == pytest.approx([2 / 3, 2 / 3, -2 / 3, -2 / 3])

    assert group_b["zero_reward_variance"] is True
    assert group_b["loo_advantages"] == [0.0, 0.0]
    assert summary["num_zero_variance_groups"] == 1
    assert summary["num_groups"] == 2


def test_skyrl_generator_forwards_resolved_sampling_params():
    source_path = (
        ROOT / "skyrl-agent/skyrl_agent/integrations/skyrl_train/skyrl_train_main.py"
    )
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SkyRLAgentGenerator"
    )

    class GeneratorInterface:
        pass

    namespace = {
        "GeneratorInterface": GeneratorInterface,
        "GeneratorInput": dict,
        "GeneratorOutput": dict,
        "SkyRLAgentGeneratorConfig": object,
        "InferenceEngineInterface": object,
    }
    exec(
        compile(
            ast.Module(body=[class_node], type_ignores=[]), str(source_path), "exec"
        ),
        namespace,
    )
    generator_cls = namespace["SkyRLAgentGenerator"]

    class Runner:
        async def run(self, input_batch, val_mode=False, sampling_params=None):
            return {
                "input_batch": input_batch,
                "val_mode": val_mode,
                "sampling_params": sampling_params,
            }

    generator = generator_cls.__new__(generator_cls)
    generator.agent_generator = Runner()
    sampling_params = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 4000}
    input_batch = {
        "batch_metadata": SimpleNamespace(training_phase="train"),
        "sampling_params": sampling_params,
    }

    result = asyncio.run(generator.generate(input_batch))

    assert result["val_mode"] is False
    assert result["sampling_params"] == sampling_params
