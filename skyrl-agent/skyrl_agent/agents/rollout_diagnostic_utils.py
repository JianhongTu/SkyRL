"""Small, dependency-free helpers shared by rollout collection and post-processing."""

import json
import re
from typing import Any

MASK_OUT_REASONS = frozenset(
    {
        "CONTEXT_WINDOW_EXCEEDED",
        "TRUNCATED_RESPONSE",
        "error_runtime",
        "error_evaluation",
        "BAD_LLM_RESPONSE",
        "stuck_in_a_loop",
        "cmd_timeout",
    }
)
PREFIX_TRAINABLE_TERMINAL_REASONS = frozenset(
    {"CONTEXT_BUDGET_REACHED"}
)
NON_FINISH_TERMINAL_REASONS = MASK_OUT_REASONS | {
    "error_initialization",
    "max_iterations_reached",
} | PREFIX_TRAINABLE_TERMINAL_REASONS
FINISH_REWARD_BONUS = 0.05

_REPETITION_WINDOW_TOKENS = 128
_REPETITION_MAX_PERIOD = 32
_REPETITION_MIN_MATCH_FRACTION = 0.90
_REPETITION_NGRAM_SIZE = 4
_REPETITION_MIN_DUPLICATE_NGRAM_FRACTION = 0.80

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


def _has_repetitive_suffix(token_ids: list[int]) -> bool:
    if len(token_ids) < _REPETITION_WINDOW_TOKENS:
        return False

    suffix = token_ids[-_REPETITION_WINDOW_TOKENS:]
    for period in range(1, _REPETITION_MAX_PERIOD + 1):
        comparable = len(suffix) - period
        matches = sum(
            suffix[index] == suffix[index - period]
            for index in range(period, len(suffix))
        )
        if matches / comparable >= _REPETITION_MIN_MATCH_FRACTION:
            return True

    ngrams = {
        tuple(suffix[index : index + _REPETITION_NGRAM_SIZE])
        for index in range(len(suffix) - _REPETITION_NGRAM_SIZE + 1)
    }
    num_ngrams = len(suffix) - _REPETITION_NGRAM_SIZE + 1
    duplicate_fraction = 1 - len(ngrams) / num_ngrams
    return duplicate_fraction >= _REPETITION_MIN_DUPLICATE_NGRAM_FRACTION


def has_repetitive_assistant_turn(
    token_ids: list[int], assistant_mask: list[int]
) -> bool:
    """Detect periodic or low-diversity repetition in an assistant-turn suffix."""
    if len(token_ids) != len(assistant_mask):
        raise ValueError("token_ids and assistant_mask must have the same length")

    assistant_turn: list[int] = []
    for token_id, is_assistant in zip(token_ids, assistant_mask):
        if is_assistant:
            assistant_turn.append(token_id)
        elif assistant_turn:
            if _has_repetitive_suffix(assistant_turn):
                return True
            assistant_turn = []
    return _has_repetitive_suffix(assistant_turn)


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
