"""Small, dependency-free helpers shared by rollout collection and post-processing."""

import json
import re
from typing import Any

MASK_OUT_REASONS = frozenset(
    {
        "CONTEXT_WINDOW_EXCEEDED",
        "error_runtime",
        "error_evaluation",
        "BAD_LLM_RESPONSE",
        "stuck_in_a_loop",
        "cmd_timeout",
    }
)
PREFIX_TRAINABLE_TERMINAL_REASONS = frozenset(
    {"CONTEXT_BUDGET_REACHED", "TRUNCATED_RESPONSE"}
)
NON_FINISH_TERMINAL_REASONS = MASK_OUT_REASONS | {
    "error_initialization",
    "max_iterations_reached",
} | PREFIX_TRAINABLE_TERMINAL_REASONS
FINISH_REWARD_BONUS = 0.05

_HERMES_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_LEGACY_FINISH_RE = re.compile(
    r"<function=finish(?:\s[^>]*)?>.*?</function>", re.DOTALL
)


def bounded_max_tokens(configured: int, remaining: int) -> int:
    """Respect the configured per-turn cap and the remaining context budget."""
    return min(configured, remaining)


def remaining_generation_tokens(input_length: int, model_max_length: int) -> int:
    """Return the native model budget remaining after the full encoded input."""
    return max(0, model_max_length - input_length)


def extract_exact_output_tokens(choice: dict[str, Any]) -> list[int]:
    """Return server-sampled token IDs or fail instead of silently re-tokenizing."""
    token_ids = choice.get("token_ids")
    if token_ids is None:
        raise RuntimeError(
            "Inference response omitted token_ids required for exact rollout diagnostics"
        )
    return list(token_ids)


def contains_finish_call(content: str) -> bool:
    """Recognize both legacy OpenHands and Hermes finish-call syntax."""
    if _LEGACY_FINISH_RE.search(content):
        return True
    for match in _HERMES_TOOL_CALL_RE.finditer(content):
        try:
            call = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if call.get("name") == "finish":
            return True
    return False


def normalize_finish_reason(
    messages: list[dict[str, Any]], finish_reason: str | None
) -> str | None:
    """Return the final reason used by both loss masking and rollout metrics."""
    if not messages or finish_reason in NON_FINISH_TERMINAL_REASONS:
        return finish_reason

    last_message = messages[-1]
    role = last_message.get("role")
    if role == "assistant":
        content = last_message.get("content")
        if not isinstance(content, str) or not contains_finish_call(content):
            return "BAD_LLM_RESPONSE"
    elif role == "user":
        return "error_runtime"
    return finish_reason


def apply_finish_reward_bonus(
    task_reward: float, finish_reason: str | None, *, training: bool
) -> tuple[float, float]:
    """Return shaped reward and bonus while leaving evaluation rewards untouched."""
    bonus = FINISH_REWARD_BONUS if training and finish_reason == "FINISH_TOOL" else 0.0
    return float(task_reward) + bonus, bonus
