from typing import Any, Dict, TypedDict, List, Optional, Type, Callable
from collections import defaultdict
from abc import ABC, abstractmethod
import os
import copy
from omegaconf import OmegaConf, DictConfig
import pandas as pd
from loguru import logger
from skyrl_agent.tasks.base import BaseTask
from transformers import AutoTokenizer
from dataclasses import dataclass

from skyrl_agent.integrations.base import (
    build_backend,
    build_generator_input,
    build_generator_output,
    _import_object,
    AsyncInferBackend,
)
from skyrl_agent.dispatcher.dispatchers import DISPATCHER_REGISTRY, DispatcherType
from skyrl_agent.config.configuration_utils import TASK_CONFIG_REGISTRY, get_field_from_config, TrajectoryConfig
from skyrl_agent.functional.chat_template import chat_template, chat_template_qwen3_thinking
from skyrl_agent.functional.utils import transitions_to_training_data
from skyrl_agent.agents.rollout_diagnostic_utils import (
    MASK_OUT_REASONS,
    PREFIX_TRAINABLE_TERMINAL_REASONS,
    apply_finish_reward_bonus,
    normalize_finish_reason,
)
from .mapping import AGENT_TRAJECTORY_REGISTRY


class CompleterOutput:
    pass


@dataclass
class RuntimeConfig:
    runtime_initializer: Optional[Callable]
    instruction_getter: Callable
    config_builder: Optional[Callable]
    completer: Optional[Callable]
    evaluator: Callable

    @classmethod
    def from_dict(cls, cfg: DictConfig):
        def safe_import(cfg, key):
            try:
                val = cfg.get(key, None)
                return _import_object(val) if val else None
            except AttributeError:
                return None

        runtime_initializer = safe_import(cfg, "initializer")
        config_builder = safe_import(cfg, "config_builder")
        instruction_getter = safe_import(cfg, "instruction_getter")  # If optional; else raise if missing
        completer = safe_import(cfg, "completer")
        evaluator = safe_import(cfg, "evaluator")
        return cls(
            runtime_initializer=runtime_initializer,
            config_builder=config_builder,
            instruction_getter=instruction_getter,
            completer=completer,
            evaluator=evaluator,
        )


@dataclass
class TrajectoryResult(TypedDict):
    instance_id: str
    trajectory_id: str
    messages: List[Dict[str, str]]
    state: Any
    results: Optional[CompleterOutput]
    error: Optional[str]
    finish: bool
    finish_reason: str
    reward: Optional[float]
    task_reward: Optional[float]
    finish_reward_bonus: Optional[float]
    eval_error: Optional[str]
    transitions: Optional[List[Any]]
    rollout_seconds: Optional[float]
    evaluation_seconds: Optional[float]
    reward_evaluation_seconds: Optional[float]


class BaseTrajectory(ABC):

    def __init__(
        self,
        cfg: TrajectoryConfig,
        data: Dict[str, Any],
        infer_engine: AsyncInferBackend,
        tokenizer: AutoTokenizer,
        task: BaseTask,
        val_mode: bool = False,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.data = data
        self.infer_engine = infer_engine
        self.tokenizer = tokenizer
        self.task = task
        self.val_mode = val_mode
        self.agent_cls = _import_object(cfg.agent_cls)

        self.result: TrajectoryResult = None

    @abstractmethod
    async def initialize_trajectory(self):
        pass

    @abstractmethod
    async def generate_trajectory(self):
        pass

    @abstractmethod
    async def evaluate_trajectory(self):
        pass


# TODO(csy): also specify whether loss_mask, attention_mask, etc. are needed -- for training or eval
class AgentRunner:
    def __init__(self, cfg: Dict[str, Any], infer_engine: Any, tokenizer: Any) -> None:
        """
        Initialize the CodeActGenerator with the given configuration.

        Args:
            generation_config: Configuration dictionary containing parameters like max_prompt_length, max_response_length, etc.
        """
        self.cfg = cfg

        # infer engine
        self.infer_engine = build_backend(
            cfg.generator.infer_backend,
            infer_engine=infer_engine,
            tokenizer=tokenizer,
            cfg=cfg.generator.backend_config,
        )
        self.tokenizer = tokenizer
        self.traj_cls: Type[BaseTrajectory] = _import_object(AGENT_TRAJECTORY_REGISTRY.get(cfg.agent_cls))
        self.task: BaseTask = _import_object(cfg.task)()

        # metadata
        self.trajectories: Dict[str, Dict[str, BaseTrajectory]] = {}

        # Will be set in subclasses
        self.agent_config = None

    @classmethod
    def from_task(cls, task: str, infer_engine: Any, tokenizer: Any):
        # Resolve task name or path
        if os.path.exists(task):
            config_path = task
        elif task in TASK_CONFIG_REGISTRY:
            config_path = TASK_CONFIG_REGISTRY[task]
        else:
            raise ValueError(
                f"Unknown task '{task}'. Must be a YAML path or one of: {list(TASK_CONFIG_REGISTRY.keys())}"
            )

        cfg = OmegaConf.load(config_path)

        return cls(cfg, infer_engine, tokenizer)

    def _get_data(self, content) -> Dict[str, Any]:
        """Process input data into trajectory input."""
        data_cfg = self.cfg.get("data", {})

        # Resolve instance payload
        instance = None
        if data_cfg.get("instance_key"):
            try:
                instance = get_field_from_config(data_cfg.get("instance_key"), content)
            except ValueError:
                instance = None

        # Resolve instance_id; rely on configured key and downstream default to batch_id if missing
        instance_id = None
        if data_cfg.get("instance_id_key"):
            try:
                instance_id = get_field_from_config(data_cfg.get("instance_id_key"), content)
            except ValueError:
                instance_id = None

        # Resolve data_source with default fallback
        data_source = "default"
        if data_cfg.get("data_source_key"):
            try:
                data_source = get_field_from_config(data_cfg.get("data_source_key"), content)
            except ValueError:
                data_source = "default"

        return {
            "instance": instance if instance is not None else content,
            "instance_id": instance_id,
            "data_source": data_source,
        }

    def _initialize_trajectories(self, val_mode: bool = False, sampling_params: Optional[Dict[str, Any]] = None):
        for batch_id, content in enumerate(self.batch):
            data = self._get_data(content)
            instance_id: str = data["instance_id"] if data["instance_id"] else batch_id
            self.trajectories[instance_id] = {}
            trajectory_sampling_params = sampling_params
            if trajectory_sampling_params is None:
                trajectory_sampling_params = (
                    self.cfg.generator.val_config.sampling_params if val_mode else self.cfg.generator.sampling_params
                )
                trajectory_sampling_params = OmegaConf.to_container(trajectory_sampling_params, resolve=True)
            else:
                trajectory_sampling_params = copy.deepcopy(trajectory_sampling_params)
            num_trajectories = (
                self.cfg.generator.val_config.num_trajectories if val_mode else self.cfg.generator.num_trajectories
            )

            # Simple generator-level toggles (defaults)
            profile_tools = bool(self.cfg.generator.get("profile_tools", False))
            debug_log = bool(self.cfg.generator.get("debug_log", False))
            enable_turn_reminder = bool(self.cfg.generator.get("enable_turn_reminder", False))

            for traj_id in range(num_trajectories):
                traj_cfg = TrajectoryConfig(
                    instance_id=instance_id,
                    trajectory_id=traj_id,
                    max_prompt_length=self.cfg.generator.max_prompt_length,
                    sampling_params=trajectory_sampling_params,
                    vision_is_active=self.cfg.generator.vision_is_active,
                    qwen3_enable_thinking=self.cfg.generator.qwen3_enable_thinking,
                    qwen3_acc_thinking=self.cfg.generator.qwen3_acc_thinking,
                    max_iterations=self.cfg.generator.max_iterations,
                    tools=self.cfg.tools,
                    agent_cls=self.cfg.agent_cls,
                    profile_tools=profile_tools,
                    debug_log=debug_log,
                    early_step_threshold=self.cfg.generator.get("early_step_threshold", 0),
                    enable_turn_reminder=enable_turn_reminder,
                )
                traj: BaseTrajectory = self.traj_cls(
                    cfg=traj_cfg,
                    data=data,
                    tokenizer=self.tokenizer,
                    infer_engine=self.infer_engine,
                    task=self.task,
                    val_mode=val_mode,
                )
                self.trajectories[instance_id][traj_id] = traj

    def _post_process_results(self, return_tensors=False, val_mode: bool = False) -> Dict[str, Any]:
        """
        Post-process the results to convert them into the appropriate output format.

        Returns:
            A dictionary containing the processed results.
        """
        raw_reward_list = []
        all_results = {}
        matched_results = []
        instance_list = []
        error_list = []
        resolved_list = []
        traj_reward_list = []
        has_finish_action_list = []
        finish_reason_list = []

        num_trajectories = (
            self.cfg.generator.val_config.num_trajectories if val_mode else self.cfg.generator.num_trajectories
        )

        for instance_id in self.trajectories:
            for trajectory_id in self.trajectories[instance_id]:
                all_results.setdefault(instance_id, {})[trajectory_id] = self.trajectories[instance_id][
                    trajectory_id
                ].result

        for batch_idx, content in enumerate(self.batch):
            data = self._get_data(content)
            instance = pd.Series(data["instance"])
            instance_id = data["instance_id"] if data["instance_id"] else batch_idx
            instance["instance_id"] = instance_id  # safe mutation
            trajectories = all_results.get(instance_id, {})
            matched_results.extend(trajectories.values())
            instance_list.extend([instance] * len(trajectories))

        assert len(matched_results) == num_trajectories * len(
            self.batch
        ), f"Expected number of results {num_trajectories * len(self.batch)}, got {len(matched_results)}"

        # Group results by instance_id for message handling
        results_by_instance = {}
        for i, (instance, result) in enumerate(zip(instance_list, matched_results)):
            instance_id = instance["instance_id"]
            results_by_instance.setdefault(instance_id, []).append((i, result))

        global_fallback_set = None
        for results in results_by_instance.values():
            if all(res.get("messages") for _, res in results):
                global_fallback_set = [copy.deepcopy(res) for _, res in results]
                break
        # get reward before handling empty messages
        for idx, result in enumerate(matched_results):
            reward = result.get("reward", False)
            raw_reward_list.append(reward)
        raw_reward = sum(raw_reward_list) / len(raw_reward_list)
        num_empty_messages = sum(1 for res in matched_results if not res.get("messages", []))
        # Handle empty messages by copying from another trajectory of the same instance
        for instance_id, results in results_by_instance.items():
            # Look for a non-empty base result
            fallback = next((res for _, res in results if res.get("messages")), None)
            if not fallback:
                if global_fallback_set:
                    logger.warning(
                        f"[WARN] No local fallback for instance_id {instance_id}, using global fallback set."
                    )
                    for j, (idx, res) in enumerate(results):
                        # Use corresponding global fallback result (same trajectory index)
                        fallback_res = global_fallback_set[j % len(global_fallback_set)]
                        print(f"Empty messages for instance_id {instance_id}, trajectory {idx}. Using global fallback.")
                        matched_results[idx]["messages"] = copy.deepcopy(fallback_res["messages"])
                        matched_results[idx]["finish_reason"] = "error_runtime"

                else:
                    logger.error(f"[FATAL] No fallback (local/global) for instance_id {instance_id}. Skipping.")
                    continue
            else:
                for idx, res in results:
                    if not res.get("messages", []):
                        print(f"Empty messages for instance_id {instance_id}, trajectory {idx}. Using local fallback.")
                        matched_results[idx]["messages"] = copy.deepcopy(fallback["messages"])
                        matched_results[idx]["finish_reason"] = "error_runtime"

        # error evaluation mainly due to timeout during tool execution
        mask_out_reason = MASK_OUT_REASONS
        # Get training data

        # backward compatibility for old format
        # TODO(csy): remove this after oh_agent is updated
        all_prompts = []
        all_responses = []
        legacy_results = []

        for result in matched_results:
            if result.get("transitions", []):
                continue
            messages = result.get("messages", [])
            logger.info(
                f"No transitions found for instance_id {result.get('instance_id')}, "
                f"trajectory_id {result.get('trajectory_id')}. Using messages instead."
            )
            starting_index = next(
                (i for i, message in enumerate(messages) if message["role"] == "assistant"),
                len(messages),
            )
            if starting_index == len(messages):
                print(
                    f'ERROR: Found no assistant message. len(messages) == {len(messages)} and roles are '
                    f'{[message["role"] for message in messages]}'
                )
            legacy_results.append(result)
            all_prompts.append(messages[:starting_index])
            all_responses.append(messages[starting_index:])

        legacy_training_data = {}
        if legacy_results:
            prompt_encodings = self.tokenizer.apply_chat_template(
                all_prompts,
                add_generation_prompt=True,
                return_dict=True,
            )
            response_encodings = self.tokenizer.apply_chat_template(
                all_responses,
                chat_template=chat_template_qwen3_thinking if self.cfg.generator.remove_think_tokens else chat_template,
                return_assistant_tokens_mask=True,
                add_generation_prompt=False,
                return_dict=True,
            )
            for idx, result in enumerate(legacy_results):
                legacy_training_data[id(result)] = (
                    prompt_encodings["input_ids"][idx],
                    response_encodings["input_ids"][idx],
                    response_encodings["assistant_masks"][idx],
                )

        # step-level
        prompt_input_ids = []
        response_ids = []
        response_assistant_mask = []
        logprobs = []
        steps_per_trajectory = []
        reward_list = []
        task_reward_list = []
        is_last_episode_list = []
        traj_idx_list = []
        step_finish_reason_list = []
        result_by_traj_id = {}
        finish_reason_index_by_traj_id = {}

        num_turns = []  # assistant-based turns
        for result in matched_results:
            current_traj_id = f"{result.get('instance_id')}-traj{result.get('trajectory_id')}"
            result_by_traj_id[current_traj_id] = result
            messages = result.get("messages", [])
            transitions = result.get("transitions", [])
            original_reason = result.get("finish_reason", None)
            final_reason = normalize_finish_reason(messages, original_reason)
            if final_reason != original_reason:
                print(
                    f"[WARN] Normalized finish_reason {original_reason} -> {final_reason} "
                    f"for {current_traj_id}"
                )
                result["finish_reason"] = final_reason
            task_reward = float(result.get("reward", False))
            shaped_reward, finish_bonus = apply_finish_reward_bonus(
                task_reward,
                result.get("finish_reason"),
                training=not val_mode,
            )
            result["task_reward"] = task_reward
            result["finish_reward_bonus"] = finish_bonus
            result["reward"] = shaped_reward
            # Count assistant messages as turns to match actual steps
            num_turns.append(sum(1 for m in messages if m.get("role") == "assistant"))
            # trajectory-level results
            error_list.append(result.get("error", None))
            resolved_list.append(task_reward)
            traj_reward_list.append(shaped_reward)
            has_finish_action_list.append(result.get("finish", False))
            finish_reason_index_by_traj_id[current_traj_id] = len(finish_reason_list)
            finish_reason_list.append(result.get("finish_reason", None))

            # backward compatibility for old format
            # TODO(csy): remove this after oh_agent is updated
            if not transitions:
                prompt_ids, legacy_response_ids, assistant_mask = legacy_training_data[id(result)]
                prompt_input_ids.append(prompt_ids)
                response_ids.append(legacy_response_ids)
                response_assistant_mask.append(assistant_mask)
                logprobs.append(None)
                is_last_episode_list.append(True)
                steps_per_trajectory.append(1)
                reward_list.append(result.get("reward", False))
                task_reward_list.append(result.get("task_reward", False))
                step_finish_reason_list.append(result.get("finish_reason", None))
                traj_idx_list.append(current_traj_id)
                continue

            # step-level results
            data_list = transitions_to_training_data(transitions)
            for data in data_list:
                prompt_input_ids.append(data.input_tokens)
                response_ids.append(data.response_tokens)
                logprobs.append(data.response_logprobs)
                response_assistant_mask.append(data.response_mask)
                is_last_episode_list.append(False)
            is_last_episode_list[-1] = True
            steps_per_trajectory.append(len(data_list))
            reward_list.extend([result.get("reward", False)] * len(data_list))
            task_reward_list.extend([result.get("task_reward", False)] * len(data_list))
            step_finish_reason_list.extend([result.get("finish_reason", None)] * len(data_list))
            traj_idx_list.extend([current_traj_id] * len(data_list))

        max_response_length = self.cfg.generator.max_prompt_length
        truncated_ids = []
        truncated_masks = []
        truncated_logprobs = []
        context_exceeded_traj_ids = set()

        for idx, (ids, mask, logprob, reason) in enumerate(
            zip(response_ids, response_assistant_mask, logprobs, step_finish_reason_list)
        ):
            # Check if truncation is needed
            first_nonzero = mask.index(1) if 1 in mask else len(mask)
            ids = ids[first_nonzero:]
            mask = mask[first_nonzero:]
            if logprob is not None:
                logprob = logprob[first_nonzero:]

            if len(ids) > max_response_length:
                if (
                    reason not in mask_out_reason
                    and reason not in PREFIX_TRAINABLE_TERMINAL_REASONS
                ):
                    logger.warning(
                        f"[WARN] Response length {len(ids)} > max_response_length={max_response_length} "
                        f"but finish_reason='{reason}' not in mask_out_reason={mask_out_reason}. "
                    )
                    context_exceeded_traj_ids.add(traj_idx_list[idx])
                # Truncate tokens and masks
                ids = ids[:max_response_length]
                mask = mask[:max_response_length]
                if logprob is not None:
                    logprob = logprob[:max_response_length]

            truncated_ids.append(ids)
            truncated_masks.append(mask)
            truncated_logprobs.append(logprob)

        for traj_id in context_exceeded_traj_ids:
            result = result_by_traj_id[traj_id]
            finish_reason_idx = finish_reason_index_by_traj_id[traj_id]
            finish_bonus = result.get("finish_reward_bonus", 0.0)
            result["finish_reason"] = "CONTEXT_WINDOW_EXCEEDED"
            result["finish_reward_bonus"] = 0.0
            result["reward"] = result.get("reward", 0.0) - finish_bonus
            finish_reason_list[finish_reason_idx] = "CONTEXT_WINDOW_EXCEEDED"
            traj_reward_list[finish_reason_idx] = result["reward"]
        for idx, traj_id in enumerate(traj_idx_list):
            if traj_id in context_exceeded_traj_ids:
                step_finish_reason_list[idx] = "CONTEXT_WINDOW_EXCEEDED"
                reward_list[idx] = result_by_traj_id[traj_id]["reward"]

        response_ids = truncated_ids
        response_assistant_mask = truncated_masks
        logprobs = truncated_logprobs

        loss_mask = [
            [0] * len(mask) if (reason in mask_out_reason) else mask
            for mask, reason in zip(response_assistant_mask, step_finish_reason_list)
        ]

        loss_mask_nonzero_by_traj = {}
        if len(loss_mask) == len(matched_results):
            for result, mask in zip(matched_results, loss_mask):
                key = f"{result.get('instance_id')}-traj{result.get('trajectory_id')}"
                loss_mask_nonzero_by_traj[key] = any(mask)
        else:
            for key, mask in zip(traj_idx_list, loss_mask):
                loss_mask_nonzero_by_traj[key] = loss_mask_nonzero_by_traj.get(key, False) or any(mask)

        trajectory_records = []
        for result in matched_results:
            key = f"{result.get('instance_id')}-traj{result.get('trajectory_id')}"
            messages = result.get("messages", [])
            transitions = result.get("transitions", [])
            transition_metrics = [getattr(transition, "metrics", {}) or {} for transition in transitions]
            trajectory_records.append(
                {
                    "instance_id": result.get("instance_id"),
                    "trajectory_id": result.get("trajectory_id"),
                    "reward": result.get("reward", False),
                    "task_reward": result.get("task_reward", False),
                    "finish_reward_bonus": result.get("finish_reward_bonus", 0.0),
                    "finish_reason": result.get("finish_reason"),
                    "loss_mask_nonzero": loss_mask_nonzero_by_traj.get(key),
                    "finish": result.get("finish", False),
                    "error": result.get("error"),
                    "eval_error": result.get("eval_error"),
                    "turn_count": sum(1 for message in messages if message.get("role") == "assistant"),
                    "prompt_token_count": (
                        len(transitions[0].ob.input_ids) if transitions else None
                    ),
                    "generated_token_count": sum(len(transition.ac.token_ids or []) for transition in transitions),
                    "turn_generated_tokens": [
                        int(metrics.get("output_token_count", 0)) for metrics in transition_metrics
                    ],
                    "generation_seconds": sum(
                        float(metrics.get("generation_seconds", 0.0)) for metrics in transition_metrics
                    ),
                    "turn_stop_reasons": [metrics.get("finish_reason") for metrics in transition_metrics],
                    "rollout_seconds": result.get("rollout_seconds"),
                    "evaluation_seconds": result.get("evaluation_seconds"),
                    "reward_evaluation_seconds": result.get("reward_evaluation_seconds"),
                }
            )

        rollout_metrics = {}
        # Compute assistant-based turn average and record metric
        avg_turn_assistant = (sum(num_turns) / len(num_turns)) if len(num_turns) > 0 else 0.0
        rollout_metrics["rollout_metrics/avg_turn_assistant"] = avg_turn_assistant

        # Note: no backward-compat key kept (removed per request)

        total_per_instance = defaultdict(int)
        resolved_per_instance = defaultdict(int)
        for instance, reward in zip(instance_list, resolved_list):
            instance_id = instance["instance_id"]
            total_per_instance[instance_id] += 1
            if reward > 0:
                resolved_per_instance[instance_id] += 1

        # Track how many instances have resolution rate 0% or 100%
        num_resolved_0 = 0
        num_resolved_1 = 0

        # Print ratio and update counts
        for instance in sorted(total_per_instance):
            total = total_per_instance[instance]
            resolved = resolved_per_instance[instance]

            if resolved == 0:
                num_resolved_0 += 1
            elif resolved == total:
                num_resolved_1 += 1

        rollout_metrics["rollout_metrics/num_all_resolved"] = num_resolved_1
        rollout_metrics["rollout_metrics/num_none_resolved"] = num_resolved_0
        rollout_metrics["rollout_metrics/finish_tool_ratio"] = sum(
            1 for reason in finish_reason_list if reason == "FINISH_TOOL"
        ) / len(finish_reason_list)
        context_terminal_reasons = {
            "CONTEXT_WINDOW_EXCEEDED",
            "CONTEXT_BUDGET_REACHED",
            "TRUNCATED_RESPONSE",
        }
        rollout_metrics["rollout_metrics/context_exceed_ratio"] = sum(
            1 for reason in finish_reason_list if reason in context_terminal_reasons
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/context_budget_ratio"] = sum(
            1 for reason in finish_reason_list if reason == "CONTEXT_BUDGET_REACHED"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/truncated_response_ratio"] = sum(
            1 for reason in finish_reason_list if reason == "TRUNCATED_RESPONSE"
        ) / len(finish_reason_list)
        # Ratio of trajectories stopped by iteration cap; avoid 'max' in key to prevent max-reduction
        rollout_metrics["rollout_metrics/iter_cap_ratio"] = sum(
            1 for reason in finish_reason_list if reason == "max_iterations_reached"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/stuck_in_a_loop_ratio"] = sum(
            1 for reason in finish_reason_list if reason == "stuck_in_a_loop"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/error_runtime"] = sum(
            1 for reason in finish_reason_list if reason == "error_runtime"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/error_evaluation"] = sum(
            1 for reason in finish_reason_list if reason == "error_evaluation"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/num_mask_out"] = sum(1 for mask in loss_mask if all(m == 0 for m in mask))
        rollout_metrics["rollout_metrics/num_mask_non_zero_reward"] = sum(
            1 for mask, reward in zip(loss_mask, resolved_list) if all(m == 0 for m in mask) and reward > 0
        )
        rollout_metrics["rollout_metrics/bad_llm_response"] = sum(
            1 for reason in finish_reason_list if reason == "BAD_LLM_RESPONSE"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/raw_reward"] = raw_reward
        rollout_metrics["rollout_metrics/num_empty_messages"] = num_empty_messages

        # Optional aggregation of tool-call profiling if available
        try:
            cfg_profile = None
            for path in ("generator.profile_tools", "skyrl_agent.profile_tools"):
                try:
                    cfg_profile = OmegaConf.select(self.cfg, path)
                except Exception:
                    cfg_profile = None
                if cfg_profile is not None:
                    break
            profile_enabled = bool(cfg_profile)
            tool_calls_per_traj = []
            tool_calls_per_traj_nf = []  # exclude finish
            tool_name_totals = defaultdict(int)
            if profile_enabled:
                for res in matched_results:
                    state = res.get("state") or {}
                    prof = state.get("tool_profile") if isinstance(state, dict) else None
                    if prof and isinstance(prof, dict):
                        total = prof.get("tool_calls_total")
                        by_name = prof.get("tool_calls_by_name") or {}
                        if isinstance(total, int):
                            tool_calls_per_traj.append(total)
                        if isinstance(by_name, dict):
                            # accumulate per-tool totals
                            for k, v in by_name.items():
                                try:
                                    tool_name_totals[k] += int(v)
                                except Exception:
                                    pass
                            # compute no-finish per-traj sum
                            try:
                                nf_sum = sum(int(v) for k, v in by_name.items() if k != "finish")
                                tool_calls_per_traj_nf.append(nf_sum)
                            except Exception:
                                pass

            def emit_distribution(prefix: str, vals: list[int]):
                if not vals:
                    return
                n = len(vals)
                s = sorted(vals)
                mean_val = sum(vals) / n
                mnv = s[0]
                mxv = s[-1]
                rollout_metrics[f"{prefix}_total"] = int(sum(vals))
                rollout_metrics[f"{prefix}_per_traj_mean"] = float(mean_val)
                rollout_metrics[f"{prefix}_per_traj_min"] = float(mnv)
                rollout_metrics[f"{prefix}_per_traj_max"] = float(mxv)

            # Emit distributions for overall and no-finish variants
            emit_distribution("rollout_metrics/tool_calls", tool_calls_per_traj)
            emit_distribution("rollout_metrics/tool_calls_no_finish", tool_calls_per_traj_nf)

            for name, cnt in tool_name_totals.items():
                rollout_metrics[f"rollout_metrics/tool_name/{name}"] = int(cnt)
        except Exception:
            pass

        print("rollout metrics:", rollout_metrics)

        print(f"Finish reason: {finish_reason_list}")
        # Create tensor dictionary
        output = {
            "prompt_token_ids": prompt_input_ids,
            "response_ids": response_ids,
            "rewards": reward_list,
            "task_rewards": task_reward_list,
            "traj_rewards": traj_reward_list,
            "loss_masks": loss_mask,
            "episode_nums": steps_per_trajectory,
            "is_last_episode": is_last_episode_list,
            "traj_idx": traj_idx_list,
            "stop_reasons": [None] * len(prompt_input_ids),
            "rollout_logprobs": logprobs,
            "rollout_metrics": rollout_metrics,
            "trajectory_records": trajectory_records,
        }

        return output

    async def run(
        self,
        input_batch: Any,
        val_mode: bool = False,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Generate trajectories for the given prompts using the configured agents.

        Args:
            prompts: A dictionary containing training instances.
            val_mode: Whether we're running validation.

        Returns:
            Results converted to the appropriate output format based on infer backend.
        """
        self.batch = build_generator_input(self.cfg.generator.infer_backend, input_batch).input_batch

        num_trajectories = (
            self.cfg.generator.val_config.num_trajectories if val_mode else self.cfg.generator.num_trajectories
        )
        if sampling_params is not None:
            effective_sampling_params = copy.deepcopy(sampling_params)
        elif val_mode:
            effective_sampling_params = self.cfg.generator.val_config.sampling_params
        else:
            effective_sampling_params = self.cfg.generator.sampling_params

        # Initialize agents and other components
        self._initialize_trajectories(val_mode=val_mode, sampling_params=sampling_params)

        generator_dispatcher: DispatcherType | None = DISPATCHER_REGISTRY.get(self.cfg.dispatcher.type)
        if not generator_dispatcher:
            raise ValueError(f"Unknown generator type: {self.cfg.dispatcher.type}")
        else:
            logger.info(f"Using generator dispatcher: {self.cfg.dispatcher.type}")
            init_fn = "initialize_trajectory"
            run_fn = "generate_trajectory"
            eval_fn = "evaluate_trajectory"
            if val_mode:
                max_parallel_agents = self.cfg.dispatcher.get("val_config", {}).get(
                    "max_parallel_agents", self.cfg.dispatcher.max_parallel_agents
                )
                max_eval_parallel_agents = self.cfg.dispatcher.get("val_config", {}).get(
                    "max_eval_parallel_agents", self.cfg.dispatcher.max_eval_parallel_agents
                )
                dispatcher_cfg = {
                    "sampling_params": effective_sampling_params,
                    "max_parallel_agents": max_parallel_agents,
                    "max_eval_parallel_agents": max_eval_parallel_agents,
                    "num_instances": len(self.batch),
                    "num_trajectories": num_trajectories,
                }
            else:
                dispatcher_cfg = {
                    "sampling_params": effective_sampling_params,
                    "max_parallel_agents": self.cfg.dispatcher.max_parallel_agents,
                    "max_eval_parallel_agents": self.cfg.dispatcher.max_eval_parallel_agents,
                    "num_instances": len(self.batch),
                    "num_trajectories": num_trajectories,
                }
            await generator_dispatcher(
                dispatcher_cfg, self.trajectories, init_fn=init_fn, run_fn=run_fn, eval_fn=eval_fn
            )

        output = self._post_process_results(val_mode=val_mode)

        # reset after run
        self.trajectories = {}

        return build_generator_output(self.cfg.generator.infer_backend, output).result
