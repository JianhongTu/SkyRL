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
Download nvidia/Nemotron-SFT-SWE-v3 and keep only the OpenHands-harness split.

The dataset mixes many agent harnesses (OpenHands, SWE-agent-style, opencode,
Codex CLI, ...), each identified by its *tool interface*. OpenHands' CodeActAgent
uses the distinctive tool names ``execute_bash`` + ``str_replace_editor`` (plus
``finish`` / ``think`` / ``task_tracker``), which no other harness in this dataset
uses. Filtering by tool interface is more robust than string-matching the
"You are OpenHands agent" system prompt, because the dataset prompt-augments
OpenHands (same tools, varied persona text).

A row is kept iff its tool universe satisfies::

    {execute_bash, str_replace_editor}  ⊆  tools  ⊆  OPENHANDS_TOOLS

The upper bound makes the tool signature **consistent by construction**: any row
that also calls a foreign harness's tool is dropped, so every kept trajectory
speaks exactly the OpenHands tool API. The script additionally **asserts the
Arrow schema is identical across all shards** so the output concatenates cleanly.

Reasoning is bimodal in this dataset: ~51% of OpenHands trajectories carry
``reasoning_content`` on ~every assistant turn, ~49% carry none. By default we
keep only the reasoning trajectories (``--reasoning only``); use ``none`` for the
non-reasoning split or ``any`` to keep both.

This script streams shard-by-shard: download -> filter -> write -> delete raw
(unless --keep-raw), so peak disk stays at one raw shard (~120 MB) plus the
growing OpenHands output (~4 GB total), instead of the full ~11.7 GB dataset.

Usage:
    # Full run: all 96 shards -> ~/data/nemotron_sft_swe_v3_openhands
    uv run --isolated examples/train/multi-env/sft/prepare_openhands.py

    # Smoke test on the first shard, into one combined parquet
    uv run --isolated examples/train/multi-env/sft/prepare_openhands.py \
        --max-shards 1 --single-file

Set ``HF_TOKEN`` if you hit Hub rate limits.
"""

import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

# OpenHands CodeActAgent tool interface. REQUIRED must be present; the full set
# is the allowed upper bound that guarantees a consistent tool signature.
OPENHANDS_REQUIRED_TOOLS = frozenset({"execute_bash", "str_replace_editor"})
OPENHANDS_TOOLS = frozenset({"execute_bash", "str_replace_editor", "finish", "think", "task_tracker"})


def _iter_tool_calls(messages):
    """Yield (name, arguments_dict) for every tool call in a trajectory."""
    for m in messages:
        tc = m.get("tool_calls")
        if not tc:
            continue
        if isinstance(tc, str):
            try:
                tc = json.loads(tc)
            except json.JSONDecodeError:
                continue
        if isinstance(tc, dict):
            tc = [tc]
        if not isinstance(tc, list):
            continue
        for c in tc:
            if not isinstance(c, dict):
                continue
            fn = c.get("function", c)
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            if name:
                yield name, args


def _toolset(messages) -> frozenset:
    return frozenset(name for name, _ in _iter_tool_calls(messages))


def is_openhands(messages) -> bool:
    """Keep iff REQUIRED ⊆ tools ⊆ OPENHANDS_TOOLS (consistent OpenHands signature)."""
    ts = _toolset(messages)
    return OPENHANDS_REQUIRED_TOOLS.issubset(ts) and ts.issubset(OPENHANDS_TOOLS)


def has_reasoning(messages) -> bool:
    """True if any assistant turn carries non-empty ``reasoning_content``.

    Reasoning is bimodal per-trajectory in this dataset: a trajectory either has
    reasoning on ~every assistant turn or none at all, so a single non-empty turn
    cleanly classifies the whole trajectory.
    """
    for m in messages:
        if m.get("role") == "assistant" and (m.get("reasoning_content") or "").strip():
            return True
    return False


def prepare(
    dataset_name: str,
    output_dir: str,
    raw_dir: str,
    revision: str | None,
    keep_raw: bool,
    single_file: bool,
    max_shards: int | None,
    reasoning: str,
) -> None:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import HfApi, hf_hub_download

    output_dir = os.path.expanduser(output_dir)
    raw_dir = os.path.expanduser(raw_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset", revision=revision)
    shards = sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))
    if max_shards is not None:
        shards = shards[:max_shards]
    if not shards:
        raise FileNotFoundError(f"No data/*.parquet shards found in {dataset_name}")
    print(f"{dataset_name}: {len(shards)} shard(s) to process")
    print(f"Filter: {{execute_bash, str_replace_editor}} ⊆ tools ⊆ {set(OPENHANDS_TOOLS)}")
    print(f"Reasoning: {reasoning}  (only=keep reasoning trajectories, none=keep non-reasoning, any=keep both)")
    print()

    reference_schema = None
    total_in = total_out = 0
    rows_with_reasoning = 0
    from collections import Counter, defaultdict

    tool_calls = Counter()  # name -> count across kept rows
    arg_keys = defaultdict(Counter)  # name -> Counter(arg_key)
    writer = None
    single_path = os.path.join(output_dir, "openhands.parquet")

    for shard in shards:
        local = hf_hub_download(
            repo_id=dataset_name,
            filename=shard,
            repo_type="dataset",
            revision=revision,
            local_dir=raw_dir,
        )
        table = pq.read_table(local)

        # --- schema consistency assertion ---
        if reference_schema is None:
            reference_schema = table.schema
        elif not table.schema.equals(reference_schema):
            raise ValueError(f"Schema mismatch in {shard}.\nExpected:\n{reference_schema}\nGot:\n{table.schema}")

        df = table.to_pandas()
        keep_idx = []
        for i, msgs in enumerate(df["messages"]):
            if not is_openhands(msgs):
                continue
            reasons = has_reasoning(msgs)
            if reasoning == "only" and not reasons:
                continue
            if reasoning == "none" and reasons:
                continue
            keep_idx.append(i)
            if reasons:
                rows_with_reasoning += 1
            for name, args in _iter_tool_calls(msgs):
                tool_calls[name] += 1
                for k in args:
                    arg_keys[name][k] += 1
        total_in += len(df)
        total_out += len(keep_idx)

        # Build an explicitly-typed index array: an empty Python list would make
        # pyarrow infer a null-typed array and `take` would raise (some shards keep
        # zero OpenHands trajectories).
        kept = table.take(pa.array(keep_idx, type=pa.int64()))
        if single_file:
            if writer is None:
                writer = pq.ParquetWriter(single_path, kept.schema)
            if kept.num_rows:
                writer.write_table(kept)
        else:
            pq.write_table(kept, os.path.join(output_dir, os.path.basename(shard)))

        if not keep_raw:
            os.remove(local)

        print(f"  {os.path.basename(shard)}: {len(keep_idx):5d}/{len(df):5d} kept")

    if writer is not None:
        writer.close()

    # --- tool-signature consistency check (kept rows only use OpenHands tools) ---
    seen_tools = set(tool_calls)
    assert seen_tools.issubset(OPENHANDS_TOOLS), f"Unexpected tools in output: {seen_tools - OPENHANDS_TOOLS}"

    print()
    pct = (100 * total_out / total_in) if total_in else 0.0
    print(f"Done. Kept {total_out}/{total_in} OpenHands trajectories ({pct:.1f}%).")
    rpct = (100 * rows_with_reasoning / total_out) if total_out else 0.0
    print(f"  with reasoning_content: {rows_with_reasoning}/{total_out} ({rpct:.1f}%)")
    print(f"Output: {single_path if single_file else output_dir + '/'}")
    print()
    print("Schema (consistent across all shards):")
    print(reference_schema)
    print()
    print("Tool signature (kept rows) — arg keys as % of that tool's calls:")
    for name in sorted(tool_calls, key=lambda n: -tool_calls[n]):
        n = tool_calls[name]
        sig = ", ".join(f"{k}:{100 * v / n:.0f}%" for k, v in arg_keys[name].most_common())
        print(f"  {name:<20} ({n} calls) -> {sig}")
    # Flag rare arg keys (<0.5% of a tool's calls) as likely generation typos.
    rare = []
    for name, keys in arg_keys.items():
        n = tool_calls[name]
        for k, v in keys.items():
            if v < max(1, 0.005 * n):
                rare.append(f"{name}.{k} ({v})")
    if rare:
        print()
        print(f"NOTE: {len(rare)} rare arg key(s) (<0.5%), likely generation typos: {', '.join(sorted(rare))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Nemotron-SFT-SWE-v3 and keep only OpenHands trajectories.")
    parser.add_argument("--dataset", default="nvidia/Nemotron-SFT-SWE-v3", help="HF dataset repo id.")
    parser.add_argument(
        "--output-dir",
        default="~/data/nemotron_sft_swe_v3_openhands",
        help="Directory to write the filtered OpenHands parquet shards into.",
    )
    parser.add_argument(
        "--raw-dir",
        default="~/data/nemotron_sft_swe_v3_raw",
        help="Scratch dir for raw shard downloads (deleted per-shard unless --keep-raw).",
    )
    parser.add_argument("--revision", default=None, help="Branch/tag/commit to download (default: latest).")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw shards after filtering (default: delete).")
    parser.add_argument(
        "--single-file", action="store_true", help="Write one combined openhands.parquet instead of per-shard files."
    )
    parser.add_argument("--max-shards", type=int, default=None, help="Only process the first N shards (for testing).")
    parser.add_argument(
        "--reasoning",
        choices=["only", "none", "any"],
        default="only",
        help="Keep reasoning trajectories only (default), non-reasoning only, or both.",
    )
    args = parser.parse_args()

    prepare(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        revision=args.revision,
        keep_raw=args.keep_raw,
        single_file=args.single_file,
        max_shards=args.max_shards,
        reasoning=args.reasoning,
    )
