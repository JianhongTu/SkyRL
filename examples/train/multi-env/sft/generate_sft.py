# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Turn the OpenHands-filtered Nemotron-SFT-SWE-v3 split into a SkyRL SFT dataset.

Input: the parquet produced by ``prepare_openhands.py`` (column ``messages`` =
list<struct<role, content, reasoning_content, tool_calls:str>>, plus ``uuid``,
``license``). That output is filtered but not yet trainable: it has no tool
schemas (the model would never see tool definitions) and carries tool-call args
the rollout agent's parser rejects.

This script produces ``train.parquet`` with columns:
  * ``messages`` — same struct, with ``tool_calls`` rewritten (non-schema args
    stripped) and ``messages[0]`` (the lone system turn) **replaced** by the exact
    system prompt the rollout harness sends (``get_system_message()``). The Nemotron
    rows bake in a ~12.8k-char OpenHands CodeAct prompt the rollout/eval harness never
    emits (it sends a 386-char prompt); training on the baked-in prompt is pure
    train/rollout skew, so we overwrite it. No separate ``system`` column.
  * ``tools``    — JSON string, identical for every row: the 4 OpenHands tool
    schemas in ``_get_tools()`` order [execute_bash, think, finish,
    str_replace_editor]. The SFT trainer's ``_coerce_tools`` parses it and
    forwards it as ``tools=`` to the chat template.
  * ``uuid``     — carried through for provenance.

Transforms (see the plan):
  1. Drop trajectories that use ``task_tracker`` (not an agent tool) — the only
     transform that discards whole trajectories.
  2. Strip tool-call args not in the tool's declared schema (e.g. ``security_risk``,
     ``timeout``). Trajectory is kept; only the offending arg is removed. Required:
     the rollout parser HARD-ERRORS on unexpected args for str_replace_editor edits.
  3. Overwrite ``messages[0]`` with the harness system prompt (consistency swap;
     see above). Trajectory is kept; only the system turn's content changes.

Train it with ``train_on_what=all_assistant_messages`` (the agent sees prior
turns' thinking, so every assistant turn is supervised).

Usage:
    uv run --isolated examples/train/multi-env/sft/prepare_openhands.py --max-shards 1 --single-file
    uv run --isolated examples/train/multi-env/sft/generate_sft.py \
        --input ~/data/nemotron_sft_swe_v3_openhands/openhands.parquet \
        --output-dir ~/data/nemotron_sft_swe_v3_openhands_sft
"""

import argparse
import json
import os
from collections import Counter
from glob import glob

import pyarrow.parquet as pq

# OpenHands agent toolset (NOTE: tighter than prepare_openhands — drops task_tracker).
OPENHANDS_AGENT_TOOLS = frozenset({"execute_bash", "str_replace_editor", "finish", "think"})

# Hardcoded param allow-lists, asserted against the live/loaded schemas to catch drift.
_EXPECTED_PARAMS = {
    "execute_bash": {"command", "is_input"},
    "think": {"thought"},
    "finish": {"message", "task_completed"},
    "str_replace_editor": {
        "command",
        "path",
        "file_text",
        "old_str",
        "new_str",
        "insert_line",
        "view_range",
        "concise",
    },
}

_TOOLS_JSON_FALLBACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openhands_tools.json")
_SYSTEM_PROMPT_FALLBACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openhands_system_prompt.txt")

# The rollout harness is OUR OpenHands fork (../SkyRL-OpenHands, branch feature/mounted-runtime-r2e):
# its codeact_agent/prompts/system_prompt.j2 renders the SHORT 386-char "You are a programming agent…"
# harness prompt. UPSTREAM OpenHands renders a ~12.8k-char "You are OpenHands agent…" CodeAct prompt —
# importing THAT would silently re-bake the exact train/rollout skew this swap exists to remove. So the
# import path is bounded by these sanity limits and falls back to the checked-in copy if they're violated.
_HARNESS_PROMPT_PREFIX = "You are a programming agent"
_HARNESS_PROMPT_MAX_CHARS = 2000


def _is_harness_prompt(s: str) -> bool:
    """True iff ``s`` looks like our fork's short harness prompt (not upstream's ~12.8k CodeAct one)."""
    return bool(s) and s.startswith(_HARNESS_PROMPT_PREFIX) and len(s) <= _HARNESS_PROMPT_MAX_CHARS


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------


def build_tools(tools_json_path: str | None = None) -> list[dict]:
    """Return the 4 OpenHands tool schemas in ``_get_tools()`` order.

    Import-first (guarantees a byte-match with the rollout agent's ``_get_tools()``);
    falls back to a checked-in JSON if the OpenHands fork isn't importable.
    """
    if tools_json_path:
        with open(os.path.expanduser(tools_json_path)) as f:
            return json.load(f)
    try:
        from openhands.agenthub.codeact_agent.tools import (
            FinishTool,
            ThinkTool,
            create_cmd_run_tool,
            create_str_replace_editor_tool,
        )

        tools = [
            create_cmd_run_tool(use_short_description=False),
            ThinkTool,
            FinishTool,
            create_str_replace_editor_tool(use_short_description=False),
        ]
        return json.loads(json.dumps(tools, default=lambda o: dict(o)))
    except ImportError:
        print(f"NOTE: openhands not importable; using checked-in tool schemas at {_TOOLS_JSON_FALLBACK}")
        with open(_TOOLS_JSON_FALLBACK) as f:
            return json.load(f)


def build_system_prompt(system_prompt_path: str | None = None) -> str:
    """Return the system message the rollout harness sends, for messages[0].

    Import-first from OUR OpenHands fork (byte-match with its ``get_system_message()``), falling back to
    the checked-in ``openhands_system_prompt.txt`` if the fork isn't importable. Both branches ``.strip()``
    so the value equals ``get_system_message()`` exactly (``PromptManager`` renders ``system_prompt.j2``
    and strips).

    GUARD: the import path trusts whatever ``openhands`` is installed, and UPSTREAM OpenHands renders a
    ~12.8k-char "You are OpenHands agent…" CodeAct prompt — NOT our fork's 386-char harness prompt. Baking
    that would silently reintroduce the train/rollout skew this swap removes, so we validate the imported
    string looks like the fork's harness prompt (``_is_harness_prompt``) and fall back to the checked-in
    copy if not. The final assert then guarantees the returned prompt is always the harness prompt (also
    catching a stale/corrupt fallback). An explicit ``--system-prompt`` override is trusted as-is (only
    checked non-empty) — that's the escape hatch for deliberately training on a different prompt.
    """
    if system_prompt_path:
        with open(os.path.expanduser(system_prompt_path)) as f:
            prompt = f.read().strip()
        assert prompt, f"--system-prompt file is empty: {system_prompt_path}"
        return prompt

    prompt = None
    try:
        import openhands.agenthub.codeact_agent as _codeact
        from openhands.utils.prompt import PromptManager

        prompt_dir = os.path.join(os.path.dirname(_codeact.__file__), "prompts")
        prompt = PromptManager(prompt_dir=prompt_dir).get_system_message()
    except Exception as e:  # not importable / API drift — use the checked-in copy
        print(f"NOTE: openhands system prompt unavailable ({type(e).__name__}); using {_SYSTEM_PROMPT_FALLBACK}")

    # Import must yield our fork's SHORT prompt; if it's upstream's ~12.8k one (or import failed), fall back.
    if not _is_harness_prompt(prompt or ""):
        if prompt is not None:
            print(
                f"NOTE: imported system prompt is not the fork's harness prompt "
                f"({len(prompt)} chars, starts {prompt[:32]!r}); using {_SYSTEM_PROMPT_FALLBACK}"
            )
        with open(_SYSTEM_PROMPT_FALLBACK) as f:
            prompt = f.read().strip()

    assert _is_harness_prompt(prompt), (
        f"system prompt failed validation ({len(prompt)} chars, starts {prompt[:40]!r}); expected our "
        f"OpenHands fork's harness prompt (prefix {_HARNESS_PROMPT_PREFIX!r}, <{_HARNESS_PROMPT_MAX_CHARS} "
        f"chars). Check {_SYSTEM_PROMPT_FALLBACK} matches ../SkyRL-OpenHands codeact_agent system_prompt.j2."
    )
    return prompt


def param_allowlist(tools: list[dict]) -> dict[str, frozenset]:
    """Allowed arg keys per tool, derived from the schemas (single source of truth)."""
    allow = {t["function"]["name"]: frozenset(t["function"]["parameters"].get("properties", {})) for t in tools}
    # Guard against schema drift relative to what this script was written for.
    for name, expected in _EXPECTED_PARAMS.items():
        assert name in allow, f"tool schema missing {name!r}: got {sorted(allow)}"
        assert (
            allow[name] == expected
        ), f"{name} params drifted: schema={sorted(allow[name])} expected={sorted(expected)}"
    return allow


# ---------------------------------------------------------------------------
# Tool-call transforms
# ---------------------------------------------------------------------------


def _toolset(messages) -> set:
    """Set of tool names called in a trajectory (tool_calls is a JSON string per message)."""
    names = set()
    for m in messages:
        tc = m.get("tool_calls")
        if not tc:
            continue
        try:
            calls = json.loads(tc) if isinstance(tc, str) else tc
        except json.JSONDecodeError:
            continue
        if isinstance(calls, dict):
            calls = [calls]
        if not isinstance(calls, list):
            continue
        for c in calls:
            fn = c.get("function", c) if isinstance(c, dict) else None
            name = fn.get("name") if isinstance(fn, dict) else None
            # A malformed call (no usable name) gets a sentinel so the
            # subset-check in transform_row drops the row instead of emitting
            # a {"name": null, ...} tool call.
            names.add(name if name else "<invalid_tool_call>")
    return names


def _count_assistant_tool_calls(messages) -> int:
    """Max number of tool_calls in any single assistant message (for the single-call check)."""
    worst = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        tc = m.get("tool_calls")
        if not tc:
            continue
        try:
            calls = json.loads(tc) if isinstance(tc, str) else tc
        except json.JSONDecodeError:
            continue
        if isinstance(calls, dict):
            calls = [calls]
        if isinstance(calls, list):
            worst = max(worst, len(calls))
    return worst


class _UnparseableToolArgs(Exception):
    """A tool call whose ``arguments`` can't be resolved to a dict.

    Raised by ``rewrite_tool_calls`` and caught in ``transform_row`` to DROP the
    whole trajectory. Emitting the call with empty ``{}`` args instead would teach
    the model a tool call missing its required args (e.g. ``str_replace_editor``
    without ``command``/``path``), which the rollout parser then hard-errors on.
    """


def rewrite_tool_calls(tool_calls_field, allowlist: dict, stats: dict) -> tuple[str, bool]:
    """Strip non-schema args (and coerce bool enum args) from a message's tool_calls.

    Returns ``(json_string, modified)``. Keeps the OpenAI-style shape with
    ``arguments`` as a JSON string (parquet-friendly; the trainer re-parses it).
    Mirrors ``sft_trainer._normalize_tool_call_payload`` for the parsing direction.

    All OpenHands tool args are string-typed in the schemas (e.g. ``is_input`` /
    ``task_completed`` are string enums), and the rollout parser/runtime compares
    them as strings, so a JSON bool in the source is coerced to ``"true"``/
    ``"false"`` to keep SFT and rollout consistent.

    Raises ``_UnparseableToolArgs`` if a call's ``arguments`` can't be resolved to a
    dict (unparseable JSON string, or a non-dict payload) so the caller drops the row.
    """
    if not tool_calls_field or tool_calls_field in ("[]", ""):
        return tool_calls_field or "", False
    try:
        calls = json.loads(tool_calls_field) if isinstance(tool_calls_field, str) else tool_calls_field
    except json.JSONDecodeError:
        return tool_calls_field, False
    if isinstance(calls, dict):
        calls = [calls]
    if not isinstance(calls, list):
        return tool_calls_field, False

    modified = False
    out = []
    for call in calls:
        if not isinstance(call, dict):
            out.append(call)
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        name = fn.get("name")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as e:
                raise _UnparseableToolArgs(name or "<unknown>") from e
        if not isinstance(args, dict):
            raise _UnparseableToolArgs(name or "<unknown>")
        allowed = allowlist.get(name, frozenset())
        clean = {}
        for k, v in args.items():
            if k not in allowed:
                stats["dropped_args"][k] += 1
                modified = True
                continue
            if isinstance(v, bool):  # schema enum args are strings; rollout compares as str
                v = str(v).lower()
                stats["coerced_bool_args"][k] += 1
                modified = True
            clean[k] = v
        out.append(
            {
                "id": call.get("id", "toolu_01"),
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(clean, ensure_ascii=False)},
            }
        )
    return json.dumps(out, ensure_ascii=False), modified


def transform_row(messages, allowlist: dict, system_prompt: str, stats: dict):
    """Transform one trajectory. Returns new messages list, or None to drop."""
    messages = [dict(m) for m in messages]

    # Drop #1: trajectories touching a tool the agent doesn't have (task_tracker)
    # or containing a malformed/unnamed tool call (the "<invalid_tool_call>" sentinel).
    bad_tools = _toolset(messages) - OPENHANDS_AGENT_TOOLS
    if bad_tools:
        stats["rows_dropped_unsupported_tools"] += 1
        for b in bad_tools:
            stats["dropped_tool_names"][b] += 1
        return None

    # Drop #2: an assistant turn with >1 tool call (the rollout parser/converter
    # enforce exactly one per turn; >1 also breaks the shared id="toolu_01").
    if _count_assistant_tool_calls(messages) > 1:
        stats["rows_dropped_multi_toolcall"] += 1
        return None

    # Track rows that wouldn't tokenize at train time (no trailing assistant turn).
    trimmed = list(messages)
    while trimmed and trimmed[-1].get("role") == "tool":
        trimmed.pop()
    if not trimmed or trimmed[-1].get("role") != "assistant":
        stats["rows_no_trailing_assistant"] += 1

    # Strip non-schema args (and coerce bool enum args) on every assistant tool_call.
    # Drop #3: a tool call whose arguments can't be resolved to a dict (would otherwise
    # be emitted with empty {} args and hard-error the rollout parser at train time).
    row_modified = False
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            try:
                new_tc, modified = rewrite_tool_calls(m["tool_calls"], allowlist, stats)
            except _UnparseableToolArgs as bad:
                stats["rows_dropped_unparseable_args"] += 1
                stats["unparseable_arg_tools"][str(bad)] += 1
                return None
            m["tool_calls"] = new_tc
            row_modified = row_modified or modified
        # Normalize struct fields to non-null strings (assign unconditionally:
        # setdefault would leave a present-but-None value as None).
        m["content"] = m.get("content") or ""
        m["reasoning_content"] = m.get("reasoning_content") or ""
        m["tool_calls"] = m.get("tool_calls") or ""
    if row_modified:
        stats["trajectories_modified"] += 1

    # Consistency swap: overwrite the baked-in (~12.8k-char) OpenHands system prompt
    # with the exact string the rollout harness sends, so the model conditions at
    # train time on the same system turn it sees at rollout. messages[0] is always the
    # lone system turn (verified: exactly 1 system msg/row, always at position 0).
    if messages and messages[0].get("role") == "system":
        if messages[0].get("content") != system_prompt:
            stats["rows_system_prompt_replaced"] += 1
        messages[0]["content"] = system_prompt
    else:
        stats["rows_no_system_msg"] += 1
    return messages


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate the OpenHands SFT dataset from prepare_openhands output.")
    parser.add_argument(
        "--input",
        default="~/data/nemotron_sft_swe_v3_openhands",
        help="prepare_openhands output: a .parquet file or a dir of *.parquet shards.",
    )
    parser.add_argument("--output-dir", default="~/data/nemotron_sft_swe_v3_openhands_sft")
    parser.add_argument("--output-file", default="train.parquet", help="Output filename (stem becomes the HF split).")
    parser.add_argument(
        "--tools-json", default=None, help="Override path to tool schemas JSON (else import-then-fallback)."
    )
    parser.add_argument(
        "--system-prompt", default=None, help="Override path to system prompt text (else import-then-fallback)."
    )
    parser.add_argument("--num-rows", type=int, default=None, help="Cap rows processed (smoke test).")
    args = parser.parse_args()

    input_path = os.path.expanduser(args.input)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    shards = (
        [input_path]
        if input_path.endswith(".parquet")
        else sorted(glob(os.path.join(input_path, "**", "*.parquet"), recursive=True))
    )
    if not shards:
        raise FileNotFoundError(f"No .parquet found at {input_path}")

    tools = build_tools(args.tools_json)
    allowlist = param_allowlist(tools)
    tools_json = json.dumps(tools, ensure_ascii=False)
    tool_order = [t["function"]["name"] for t in tools]
    print(f"tools ({len(tools)}): {tool_order}")

    system_prompt = build_system_prompt(args.system_prompt)
    print(f"system prompt ({len(system_prompt)} chars): {system_prompt[:70]!r}...")

    stats = {
        "rows_in": 0,
        "rows_out": 0,
        "rows_dropped_unsupported_tools": 0,
        "rows_dropped_multi_toolcall": 0,
        "rows_dropped_unparseable_args": 0,
        "rows_no_trailing_assistant": 0,
        "rows_system_prompt_replaced": 0,
        "rows_no_system_msg": 0,
        "trajectories_modified": 0,
        "dropped_args": Counter(),
        "coerced_bool_args": Counter(),
        "dropped_tool_names": Counter(),
        "unparseable_arg_tools": Counter(),
    }
    out_rows = []
    done = False
    for shard in shards:
        if done:
            break
        df = pq.read_table(shard).to_pandas()
        for _, row in df.iterrows():
            if args.num_rows is not None and stats["rows_in"] >= args.num_rows:
                done = True
                break
            stats["rows_in"] += 1
            new_messages = transform_row(row["messages"], allowlist, system_prompt, stats)
            if new_messages is None:
                continue
            uuid = row.get("uuid", "")
            out_rows.append(
                {
                    "messages": new_messages,
                    "tools": tools_json,
                    "uuid": "" if uuid is None or (isinstance(uuid, float)) else str(uuid),
                }
            )
            stats["rows_out"] += 1

    if not out_rows:
        raise RuntimeError(
            f"No rows survived filtering (rows_in={stats['rows_in']}, "
            f"dropped_unsupported_tools={stats['rows_dropped_unsupported_tools']}, "
            f"dropped_multi_toolcall={stats['rows_dropped_multi_toolcall']}, "
            f"dropped_unparseable_args={stats['rows_dropped_unparseable_args']}). "
            "Refusing to write an empty/schema-less parquet."
        )

    # Write via datasets so the parquet has a clean list<struct> schema.
    from datasets import Dataset

    out_path = os.path.join(output_dir, args.output_file)
    Dataset.from_list(out_rows).to_parquet(out_path)

    # ---- data accounting (stdout + JSON sidecar) ----
    report = {
        "input": input_path,
        "output": out_path,
        "tool_order": tool_order,
        "rows_in": stats["rows_in"],
        "rows_out": stats["rows_out"],
        "rows_dropped_unsupported_tools": stats["rows_dropped_unsupported_tools"],
        "dropped_tool_names": dict(stats["dropped_tool_names"].most_common()),
        "rows_dropped_multi_toolcall": stats["rows_dropped_multi_toolcall"],
        "rows_dropped_unparseable_args": stats["rows_dropped_unparseable_args"],
        "unparseable_arg_tools": dict(stats["unparseable_arg_tools"].most_common()),
        "rows_no_trailing_assistant_train_time_drop": stats["rows_no_trailing_assistant"],
        "system_prompt_chars": len(system_prompt),
        "rows_system_prompt_replaced": stats["rows_system_prompt_replaced"],
        "rows_no_system_msg": stats["rows_no_system_msg"],
        "trajectories_modified": stats["trajectories_modified"],
        "args_stripped_by_key": dict(stats["dropped_args"].most_common()),
        "bool_args_coerced_by_key": dict(stats["coerced_bool_args"].most_common()),
    }
    with open(os.path.join(output_dir, "generate_sft_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== generate_sft report ===")
    print(f"rows_in:  {report['rows_in']}")
    print(f"rows_out: {report['rows_out']}")
    print(
        f"DROPPED unsupported tools (trajectory loss): {report['rows_dropped_unsupported_tools']} {report['dropped_tool_names']}"
    )
    print(f"DROPPED multi-tool-call turns (trajectory loss): {report['rows_dropped_multi_toolcall']}")
    print(
        f"DROPPED unparseable tool-call args (trajectory loss): {report['rows_dropped_unparseable_args']} {report['unparseable_arg_tools']}"
    )
    print(f"would drop at train time (no trailing assistant): {report['rows_no_trailing_assistant_train_time_drop']}")
    print(
        f"SYSTEM PROMPT swapped to harness prompt ({report['system_prompt_chars']} chars) on "
        f"{report['rows_system_prompt_replaced']} rows; rows with no system msg: {report['rows_no_system_msg']}"
    )
    print(f"MODIFIED (kept) trajectories: {report['trajectories_modified']}")
    print(f"  args stripped by key: {report['args_stripped_by_key']}")
    print(f"  bool args coerced by key: {report['bool_args_coerced_by_key']}")
    print(f"Output: {out_path}")
    print("NOTE: stripping/coercion edit tool calls in place; those trajectories are KEPT, not discarded.")


if __name__ == "__main__":
    main()
