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
    stripped). The per-row system prompt stays as ``messages[0]`` (we keep the
    Nemotron personas; no separate ``system`` column).
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


def rewrite_tool_calls(tool_calls_field, allowlist: dict, stats: dict) -> tuple[str, bool]:
    """Strip non-schema args (and coerce bool enum args) from a message's tool_calls.

    Returns ``(json_string, modified)``. Keeps the OpenAI-style shape with
    ``arguments`` as a JSON string (parquet-friendly; the trainer re-parses it).
    Mirrors ``sft_trainer._normalize_tool_call_payload`` for the parsing direction.

    All OpenHands tool args are string-typed in the schemas (e.g. ``is_input`` /
    ``task_completed`` are string enums), and the rollout parser/runtime compares
    them as strings, so a JSON bool in the source is coerced to ``"true"``/
    ``"false"`` to keep SFT and rollout consistent.
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
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
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


def transform_row(messages, allowlist: dict, stats: dict):
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
    row_modified = False
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            new_tc, modified = rewrite_tool_calls(m["tool_calls"], allowlist, stats)
            m["tool_calls"] = new_tc
            row_modified = row_modified or modified
        # Normalize struct fields to non-null strings (assign unconditionally:
        # setdefault would leave a present-but-None value as None).
        m["content"] = m.get("content") or ""
        m["reasoning_content"] = m.get("reasoning_content") or ""
        m["tool_calls"] = m.get("tool_calls") or ""
    if row_modified:
        stats["trajectories_modified"] += 1
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

    stats = {
        "rows_in": 0,
        "rows_out": 0,
        "rows_dropped_unsupported_tools": 0,
        "rows_dropped_multi_toolcall": 0,
        "rows_no_trailing_assistant": 0,
        "trajectories_modified": 0,
        "dropped_args": Counter(),
        "coerced_bool_args": Counter(),
        "dropped_tool_names": Counter(),
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
            new_messages = transform_row(row["messages"], allowlist, stats)
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
            f"dropped_multi_toolcall={stats['rows_dropped_multi_toolcall']}). "
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
        "rows_no_trailing_assistant_train_time_drop": stats["rows_no_trailing_assistant"],
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
    print(f"would drop at train time (no trailing assistant): {report['rows_no_trailing_assistant_train_time_drop']}")
    print(f"MODIFIED (kept) trajectories: {report['trajectories_modified']}")
    print(f"  args stripped by key: {report['args_stripped_by_key']}")
    print(f"  bool args coerced by key: {report['bool_args_coerced_by_key']}")
    print(f"Output: {out_path}")
    print("NOTE: stripping/coercion edit tool calls in place; those trajectories are KEPT, not discarded.")


if __name__ == "__main__":
    main()
