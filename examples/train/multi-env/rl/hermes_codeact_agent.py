"""HermesOHCodeActAgent — RL rollout agent for the multi-env SWE phase.

WHY THIS EXISTS
---------------
skyrl-agent's reference ``OHCodeActAgent`` drives the rollout in OpenHands
**non-native** function calling: it renders the prompt as ``<function=X>
<parameter=Y>`` TEXT (``convert_fncall_messages_to_non_fncall_messages``) and
parses the model's output the same way (``convert_non_fncall_messages_to_fncall_messages``).

Our checkpoint was SFT'd on **Qwen hermes** tool calls (``<tools>`` block +
``<tool_call>{json}</tool_call>``) — see ``examples/train/multi-env/README.md``.
The eval proved the mismatch is fatal: in ``<function>`` mode the model emits
hermes-JSON-escaped args into the raw ``<parameter>`` slots, corrupts its edits,
and loops to ~0; it only worked once we forced ``native_tool_calling=true``
(23% resolve). The RL rollout would reproduce that failure.

This subclass overrides ONLY the three format-bearing pieces so the rollout is
hermes end-to-end, matching the SFT and the (fixed) eval. Everything else —
condenser, steps-remaining nudges, context-window handling, error handling,
reward/finish bookkeeping — is inherited unchanged. The reference harness is
left as-is (it serves as the upstream reference).

Wire it in by setting ``agent_cls`` in the RL yaml to this class's dotted path.

OPEN VALIDATION ITEMS (must pass a 1-instance rollout smoke before a full run):
  * Observation role: this keeps OpenHands' structured messages, so tool results
    should render as ``<tool_response>`` (tool role). Confirm OpenHands emits them
    as role="tool" here (not "user"); if "user", they still render, but verify it
    matches the SFT observation format.
  * Template parity: we encode with the training-copy template
    (``templates/qwen3_acc_thinking.jinja2``, byte-copied from ``skyrl/train``),
    which renders assistant content verbatim. Confirm this is the exact template
    the SFT trainer used (the skyrl-agent rollout copy DIFFERS — it re-parses
    <think>).
  * eos/stop: ensure the infer backend stops at ``<|im_end|>`` (151645).
"""

import copy
import json
import re
import traceback
from pathlib import Path
from typing import Any, Optional

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
    LLMMalformedActionError,
    LLMNoActionError,
    LLMResponseError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action, AgentFinishAction, MessageAction
from openhands.events.event import Event
from openhands.memory.condenser.condenser import Condensation, View

from skyrl_agent.agents.oh_codeact.codeact_agent import OHCodeActAgent
from skyrl_agent.agents.rollout_diagnostic_utils import (
    bounded_max_tokens,
    remaining_generation_tokens,
)
from skyrl_agent.dispatcher.async_utils import call_async_from_sync
from skyrl_agent.functional.function_calling import convert_str_to_completion_format
from skyrl_agent.functional.utils import record_transition

# Training-matching acc-thinking template (byte copy from skyrl/train), shipped
# next to this file so the rollout prompt is byte-faithful to the SFT.
_TEMPLATE_PATH = Path(__file__).parent / "templates" / "qwen3_acc_thinking.jinja2"

# Hermes tool-call block: <tool_call>\n{...}\n</tool_call>  (non-greedy, dotall)
_HERMES_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class HermesOHCodeActAgent(OHCodeActAgent):
    """OHCodeActAgent variant that prompts and parses in Qwen hermes format."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transitions = []

    @record_transition
    async def _generate(self, **kwargs):
        """Generate once while preserving the exact sampled input/output token IDs."""
        return await self.infer_engine.async_generate_ids(**kwargs)

    def _is_last_action_finish(self, state: State):
        """Preserve context terminal reasons instead of labeling them as finish."""
        terminal_reasons = {
            "CONTEXT_BUDGET_REACHED",
            "TRUNCATED_RESPONSE",
            "BAD_LLM_RESPONSE",
            "NO_FUNCTION_CALL",
            "cmd_timeout",
        }
        if state and state.history:
            last_action = next(
                (event for event in reversed(state.history) if isinstance(event, Action)),
                None,
            )
            if isinstance(last_action, AgentFinishAction):
                reason = (
                    last_action.thought
                    if last_action.thought in terminal_reasons
                    else "FINISH_TOOL"
                )
                return True, reason
        return False, None

    # ------------------------------------------------------------------ prompt
    def _encode_prompt(self, messages):
        """Encode with tools= so the template emits the <tools> block and renders
        assistant tool_calls / history as hermes <tool_call> JSON.

        Overrides the base, which (a) omits tools= and (b) uses the rollout-copy
        template. We pass tools=self.tools and use the training-copy template.
        """
        chat_template = _TEMPLATE_PATH.read_text()
        return self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            enable_thinking=self.qwen3_enable_thinking,
            chat_template=chat_template,
        )

    # ------------------------------------------------------------------ parsing
    def _parse_hermes_tool_calls(self, response_str: str):
        """Parse hermes <tool_call>{json}</tool_call> blocks from the model output.

        Returns (tool_calls, thought) where tool_calls is OpenAI-format
        (id/type/function{name,arguments-as-json-string}) and thought is the
        response with tool-call blocks stripped (the <think> + prose).
        """
        tool_calls = []
        for i, m in enumerate(_HERMES_TOOL_CALL_RE.finditer(response_str)):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                # malformed / truncated tool call — skip; no-action path handles it
                continue
            name = obj.get("name")
            args = obj.get("arguments", {})
            tool_calls.append(
                {
                    # SFT used a shared id "toolu_01"; keep single-call convention
                    "id": f"toolu_{i:02d}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args if isinstance(args, str) else json.dumps(args),
                    },
                }
            )
        thought = _HERMES_TOOL_CALL_RE.sub("", response_str).strip()
        return tool_calls, thought

    # --------------------------------------------------------------------- step
    def step(self, state: State) -> Action:
        """Hermes reimplementation of OHCodeActAgent.step.

        Identical to the base except: (1) NO convert_fncall->non_fncall (history
        stays hermes-structured / verbatim), (2) hermes output parsing.
        """
        self.step_count += 1
        print(f"instance id {self.instance_id}, trajectory {self.trajectory_id}, step {self.step_count}")
        if self.pending_actions:
            return self.pending_actions.popleft()

        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == "/exit":
            return AgentFinishAction()

        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events
            case Condensation(action=condensation_action):
                return condensation_action

        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(condensed_history, initial_user_message)
        messages = self.llm.format_messages_for_llm(messages)
        # NOTE (vs base): base calls convert_fncall_messages_to_non_fncall_messages
        # here to produce <function> text. We deliberately DO NOT — messages stay
        # OpenHands-structured so the template renders hermes.

        if len(self.messages) == 0:
            self.messages = messages
        else:
            obs = messages[-1]
            # base asserts obs["role"] == "user" (true only after non-fncall
            # conversion). Without conversion the observation is role "tool"
            # (rendered as <tool_response>) or "user"; accept either.
            remaining_steps = self.app_config.max_iterations - self.step_count + 1
            if isinstance(obs.get("content"), str):
                if remaining_steps > 1:
                    obs["content"] += f"\nSteps remaining: {remaining_steps}."
                else:
                    obs["content"] += (
                        "\nThis is your last step, make sure to use the finish tool "
                        "to submit your final answer."
                    )
            self.messages.append(obs)
            print(f"Obs: {obs.get('content')}")

        response_str = None
        try:
            input_ids = self._encode_prompt(self.messages)
            configured_max_tokens = int(self.sampling_params.get("max_tokens", 0))
            model_max_length = self.max_prompt_length + configured_max_tokens
            if len(self.messages) == 2:
                self.prompt_token_len = len(input_ids)
            else:
                self.response_token_len = len(input_ids) - self.prompt_token_len

            if self.response_token_len >= self.max_prompt_length - 3000:
                if isinstance(self.messages[-1].get("content"), str):
                    self.messages[-1]["content"] += (
                        "\nNote: You are running out of tokens, submit your solution "
                        "through finish tool now."
                    )
                input_ids = self._encode_prompt(self.messages)
                self.response_token_len = len(input_ids) - self.prompt_token_len
            if len(input_ids) > self.max_prompt_length:
                return AgentFinishAction(thought="CONTEXT_BUDGET_REACHED")

            sampling_params = copy.deepcopy(self.sampling_params)
            remaining_tokens = remaining_generation_tokens(
                len(input_ids), model_max_length
            )
            if remaining_tokens == 0:
                return AgentFinishAction(thought="CONTEXT_BUDGET_REACHED")
            sampling_params["max_tokens"] = bounded_max_tokens(
                configured=configured_max_tokens,
                remaining=remaining_tokens,
            )

            response_str, meta_info = call_async_from_sync(
                self._generate,
                input_ids=input_ids,
                sampling_params=sampling_params,
                request_id=self.agent_id,
            )
            stop_reason = meta_info.get("finish_reason")
            print(
                f"instance id {self.instance_id}, trajectory {self.trajectory_id}, "
                f"response {response_str} stop reason {stop_reason}"
            )

            if not response_str:
                return AgentFinishAction(thought="BAD_LLM_RESPONSE")

            # store the assistant turn VERBATIM (raw <think> + <tool_call>) so the
            # template replays it byte-for-byte on subsequent turns (World-A KEEP).
            self.messages.append({"role": "assistant", "content": response_str})

            # ---- hermes parse (vs base: convert_non_fncall_messages_to_fncall_messages)
            tool_calls, thought = self._parse_hermes_tool_calls(response_str)
            if not tool_calls:
                if stop_reason == "length":
                    self.transitions[-1].metrics["trainable"] = False
                    return AgentFinishAction(thought="TRUNCATED_RESPONSE")
                # no valid <tool_call> — mirror base's no-action handling; the
                # codeact_user_response nudge ("No function call detected...") will
                # prompt a retry on the next turn.
                raise LLMNoActionError("No hermes <tool_call> found in response")

            fn_call_messages = [
                {"role": "assistant", "content": thought, "tool_calls": tool_calls}
            ]
            actions = codeact_function_calling.response_to_actions(
                convert_str_to_completion_format(fn_call_messages),
                mcp_tool_names=list(self.mcp_tools.keys()),
            )
            print(f"Take action: {[type(action) for action in actions]}")
            if len(actions) > 1:
                logger.warning("Multiple actions detected, only the first action will be executed.")
            self.pending_actions.append(actions[0])

        except (
            LLMMalformedActionError,
            LLMNoActionError,
            LLMResponseError,
            FunctionCallValidationError,
            FunctionCallNotExistsError,
        ):
            raise

        except Exception as e:
            logger.error(f"Error in agent step: {str(e)}")
            logger.debug(f"{traceback.format_exc()}")
            self.pending_actions.append(
                MessageAction(
                    content=f"An error: {str(e)} encountered. Please try a different approach.",
                )
            )

        if not self.pending_actions:
            return AgentFinishAction()
        return self.pending_actions.popleft()

    # ------------------------------------------------------ trajectory logging
    def get_final_messages(self, state: State):
        """Same as base but WITHOUT convert_fncall->non_fncall, so the logged
        trajectory is the hermes conversation actually seen/produced."""
        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events
            case Condensation(action=condensation_action):
                return condensation_action

        # self.messages already holds the running hermes conversation; return it
        # directly (verbatim assistant turns + observations) for RL training.
        return self.messages


Agent.register("HermesOHCodeActAgent", HermesOHCodeActAgent)
