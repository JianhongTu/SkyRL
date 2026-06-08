#!/usr/bin/env python3
"""Fit the OpenHands SFT dataset to a token budget.

Third step of the multi-env SFT pipeline::

    prepare_openhands.py  ->  openhands.parquet
    generate_sft.py       ->  train.parquet          (messages + tools)
    filter_by_length.py   ->  train.parquet          (rows fit under --max-length)

Two modes for handling rows over budget:

* default (drop): keep only rows whose full tokenized length is **strictly below**
  ``--max-length`` (default 32768); drop the rest.
* ``--truncate``: keep every row but cut each over-long trajectory to the
  **longest prefix that stays strictly below --max-length and ends on an
  assistant action** (i.e. keep as many whole ``assistant -> observation`` steps
  as fit; the trainer trims the trailing observation). A row whose very first
  step already exceeds the budget is dropped.

Why this isn't a single ``apply_chat_template`` call
----------------------------------------------------
With ``train_on_what=all_assistant_messages`` the trainer
(``skyrl/train/sft_trainer.py::_tokenize_chat_all_assistants``) does NOT render
the whole conversation once. It encodes the leading system/user turns with
``apply_chat_template(..., tools=tools)`` and then encodes every later message
with the per-message *fixed-base* encoder
(``skyrl/train/generators/utils.py::encode_messages_subset``). The two differ:
the fixed-base encoder renders each assistant turn as ``loop.last``, so an empty
``<think></think>`` block is injected on *every* no-reasoning assistant turn,
whereas a single full render injects it at most once. So we replicate the
trainer's exact measure::

    length = len(apply_chat_template(messages[:first_assistant], tools=tools))
           + sum_over_later_messages( fixed_base_token_delta(message) )

The per-message deltas are **additive and position-independent**, which is what
makes truncation exact: the tokenized length of any assistant-ending prefix is
just the running sum of deltas, and equals what the trainer will produce for the
truncated row.

Speedup (verified, not assumed)
-------------------------------
The ``<tools>`` block lives in the system message, so it cancels out of every
per-message fixed-base delta. We pass ``tools=`` only on the leading render and
drop it from the per-message deltas. ``--verify-rows`` checks this fast path
against the exact with-tools encoder on a sample and aborts on any mismatch.

The vendored ``_normalize_chat_messages`` / ``_normalize_tool_call_payload`` /
``_coerce_tools`` are line-for-line copies of the trainer's (file:line noted) so
drift is detectable.
"""

import argparse
import json
import multiprocessing as mp
import os
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

# The Rust `tokenizers` threadpool is not fork-safe; disable it and use a spawn
# pool below so loading a fast tokenizer in the parent can't deadlock workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- vendored from skyrl/train/sft_trainer.py (keep in sync) ----------------
_NORMALIZED_KEYS = frozenset({"role", "content", "tool_calls"})


def _normalize_tool_call_payload(tc):
    # sft_trainer.py:330
    if tc is None:
        return None
    if isinstance(tc, str):
        tc = tc.strip()
        if not tc or tc == "[]":
            return None
        tc = json.loads(tc)
    if isinstance(tc, dict):
        tc = [tc]
    if not isinstance(tc, list) or not tc:
        return None
    out = []
    for call in tc:
        if not isinstance(call, dict):
            raise TypeError(f"tool call entry must be a dict, got {type(call).__name__}")
        fn = call["function"] if isinstance(call.get("function"), dict) else call
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        out.append({"type": "function", "function": {"name": fn.get("name"), "arguments": arguments}})
    return out


def _normalize_chat_messages(messages):
    # sft_trainer.py:376 -- preserves order/count; promotes string tool_calls to list form.
    out = []
    for msg in messages:
        role = msg["role"]
        new_msg = {k: v for k, v in msg.items() if k not in _NORMALIZED_KEYS}
        new_msg["role"] = role
        new_msg["content"] = msg.get("content", "") or ""
        if role == "assistant":
            tool_calls = _normalize_tool_call_payload(msg.get("tool_calls"))
            if tool_calls:
                new_msg["tool_calls"] = tool_calls
        out.append(new_msg)
    return out


def _coerce_tools(tools):
    # sft_trainer.py:313
    if tools is None:
        return None
    if isinstance(tools, str):
        if not tools.strip():
            return None
        parsed = json.loads(tools)
        return list(parsed) if parsed else None
    if isinstance(tools, list):
        return tools or None
    raise TypeError(f"Unsupported `tools` type: {type(tools).__name__}")


# ---------------------------------------------------------------------------

# Fixed base used by skyrl/train/generators/utils.py::encode_messages_subset.
_BASE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _apply_len(tok, msgs, **kw):
    return len(tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True, return_dict=False, **kw))


def _prep(messages):
    """Replicate tokenize_chat_example preamble; return (messages, first_assistant_i) or None.

    None means the trainer would drop the row (no trailing assistant turn).
    """
    messages = list(messages)
    while messages and messages[-1]["role"] == "tool":  # sft_trainer.py:483
        messages.pop()
    if not messages or messages[-1]["role"] != "assistant":  # sft_trainer.py:486
        return None
    messages = _normalize_chat_messages(messages)
    i = 0
    while i < len(messages) and messages[i]["role"] != "assistant":  # sft_trainer.py:586
        i += 1
    if i >= len(messages):
        return None
    return messages, i


def length_fast(messages, tools, tok, base_len_no_tools):
    """Trainer-exact full-trajectory length, tools dropped from per-message deltas."""
    prep = _prep(messages)
    if prep is None:
        return None
    messages, i = prep
    total = _apply_len(tok, messages[:i], **({"tools": tools} if tools else {}))
    for m in messages[i:]:
        total += _apply_len(tok, _BASE + [m]) - base_len_no_tools  # tools cancels out of the delta
    return total


def length_exact(messages, tools, tok):
    """Trainer-exact length, tools passed on every fixed-base delta (the literal trainer path)."""
    prep = _prep(messages)
    if prep is None:
        return None
    messages, i = prep
    kw = {"tools": tools} if tools else {}
    total = _apply_len(tok, messages[:i], **kw)
    base_with = _apply_len(tok, _BASE, **kw)
    for m in messages[i:]:
        total += _apply_len(tok, _BASE + [m], **kw) - base_with
    return total


def truncate_keep_n(orig_messages, tools, tok, base_len_no_tools, max_length):
    """Longest assistant-ending prefix with trainer-length < max_length.

    Returns ``(keep_n, final_len)``: keep ``orig_messages[:keep_n]`` (ends on an
    assistant turn, tokenizes to ``final_len`` < ``max_length``). ``(None, None)``
    if not even the first assistant step fits, or there is no assistant turn.
    """
    n = len(orig_messages)
    i = 0
    while i < n and orig_messages[i]["role"] != "assistant":
        i += 1
    if i >= n:
        return None, None
    norm = _normalize_chat_messages(orig_messages)  # same order/count as orig -> keep_n indexes both
    running = _apply_len(tok, norm[:i], **({"tools": tools} if tools else {}))
    keep_n, final_len = None, None
    for j in range(i, n):
        running += _apply_len(tok, _BASE + [norm[j]]) - base_len_no_tools
        if running >= max_length:
            break
        if orig_messages[j]["role"] == "assistant":
            keep_n, final_len = j + 1, running
    return keep_n, final_len


# --- worker plumbing --------------------------------------------------------
_TOK = None
_BASE_LEN_NO_TOOLS = None
_TRUNCATE = False
_MAX_LENGTH = 0


def _init(model, chat_template, truncate, max_length):
    global _TOK, _BASE_LEN_NO_TOOLS, _TRUNCATE, _MAX_LENGTH
    from transformers import AutoTokenizer

    _TOK = AutoTokenizer.from_pretrained(model)
    if chat_template is not None:
        _TOK.chat_template = chat_template
    _BASE_LEN_NO_TOOLS = _apply_len(_TOK, _BASE)
    _TRUNCATE = truncate
    _MAX_LENGTH = max_length


def _process_rg(args):
    path, rg = args
    pf = pq.ParquetFile(path)
    tbl = pf.read_row_group(rg, columns=["messages", "tools"])
    msgs = tbl.column("messages").to_pylist()
    tools_raw = tbl.column("tools").to_pylist()
    out = []
    for m, t in zip(msgs, tools_raw):
        tools = _coerce_tools(t)
        if _TRUNCATE:
            keep_n, final_len = truncate_keep_n(m, tools, _TOK, _BASE_LEN_NO_TOOLS, _MAX_LENGTH)
            out.append((keep_n, final_len, len(m)))
        else:
            out.append(length_fast(m, tools, _TOK, _BASE_LEN_NO_TOOLS))
    return rg, out


def _load_tokenizer(model, chat_template_path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    template = None
    if chat_template_path:
        with open(os.path.expanduser(chat_template_path)) as f:
            template = f.read()
        tok.chat_template = template
        print(f"chat template: from file {chat_template_path}")
    elif tok.chat_template is None:
        raise SystemExit(f"Tokenizer for {model} has no default chat_template; pass --chat-template <jinja file>.")
    else:
        print(f"chat template: tokenizer default for {model}")
    return tok, template


def _percentiles(sorted_vals, ps):
    if not sorted_vals:
        return {p: None for p in ps}
    n = len(sorted_vals)
    return {p: sorted_vals[max(0, min(n - 1, int(round(p / 100 * (n - 1)))))] for p in ps}


def _histogram(values, max_length):
    edges = [(0, 4096), (4096, 8192), (8192, 16384), (16384, max_length), (max_length, 65536), (65536, 1 << 30)]
    labels = ["<4k", "4-8k", "8-16k", f"16k-{max_length}", f"{max_length}-64k", ">=64k"]
    buckets = Counter()
    for v in values:
        for (lo, hi), lab in zip(edges, labels):
            if lo <= v < hi:
                buckets[lab] += 1
                break
    return {lab: buckets.get(lab, 0) for lab in labels}


def _write_sliced(pf, keep_ns, output_path):
    """Stream row groups, slice each row's messages to keep_ns[row], drop None rows."""
    schema = pf.schema_arrow
    msgs_t = schema.field("messages").type
    tools_t = schema.field("tools").type
    uuid_t = schema.field("uuid").type
    writer = None
    offset = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        n = tbl.num_rows
        msgs = tbl.column("messages").to_pylist()
        tools = tbl.column("tools").to_pylist()
        uuids = tbl.column("uuid").to_pylist()
        rg_keep = keep_ns[offset : offset + n]
        offset += n
        nm, nt, nu = [], [], []
        for m, t, u, kn in zip(msgs, tools, uuids, rg_keep):
            if kn is None:
                continue
            nm.append(m[:kn])
            nt.append(t)
            nu.append(u)
        if not nm:
            continue
        out = pa.table(
            {
                "messages": pa.array(nm, type=msgs_t),
                "tools": pa.array(nt, type=tools_t),
                "uuid": pa.array(nu, type=uuid_t),
            }
        ).select(schema.names)
        if writer is None:
            writer = pq.ParquetWriter(output_path, out.schema)
        writer.write_table(out)
    if writer is None:
        raise RuntimeError("No rows survived; refusing to write an empty dataset.")
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Fit SFT rows to a token budget (drop or truncate).")
    parser.add_argument("--input", default="~/data/nemotron_sft_swe_v3_openhands_sft/train.parquet")
    parser.add_argument("--output-dir", default="~/data/nemotron_sft_swe_v3_openhands_sft_max32k")
    parser.add_argument("--output-file", default="train.parquet", help="Output filename (stem becomes the HF split).")
    parser.add_argument("--max-length", type=int, default=32768, help="Token budget; rows must be STRICTLY below this.")
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate over-long trajectories to the longest assistant-ending prefix that fits, "
        "instead of dropping them.",
    )
    parser.add_argument("--model", default="willhx/Qwen3-30B-A3B_base_math_search", help="Tokenizer to load.")
    parser.add_argument(
        "--chat-template",
        default=None,
        help="Optional jinja file; default uses the tokenizer's own template (what the trainer uses).",
    )
    parser.add_argument("--num-procs", type=int, default=min(os.cpu_count() or 8, 64))
    parser.add_argument("--verify-rows", type=int, default=64, help="Rows to check fast==exact before the full run.")
    args = parser.parse_args()

    input_path = os.path.expanduser(args.input)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output_file)

    pf = pq.ParquetFile(input_path)
    n_rows = pf.metadata.num_rows
    n_rg = pf.num_row_groups
    mode = "TRUNCATE over-long rows" if args.truncate else "DROP over-long rows"
    print(f"input: {input_path}  rows={n_rows}  row_groups={n_rg}")
    print(f"mode: {mode}; budget = length STRICTLY < {args.max_length}")

    tok, template = _load_tokenizer(args.model, args.chat_template)
    base_len_no_tools = _apply_len(tok, _BASE)

    # --- verify the fast path == the exact (with-tools) encoder on a sample ---
    if args.verify_rows > 0:
        sample = pf.read_row_group(0, columns=["messages", "tools"])
        s_msgs = sample.column("messages").to_pylist()[: args.verify_rows]
        s_tools = sample.column("tools").to_pylist()[: args.verify_rows]
        max_diff = 0
        for m, t in zip(s_msgs, s_tools):
            tools = _coerce_tools(t)
            max_diff = max(
                max_diff, abs((length_fast(m, tools, tok, base_len_no_tools) or 0) - (length_exact(m, tools, tok) or 0))
            )
        if max_diff != 0:
            raise SystemExit(f"VERIFY FAILED: fast vs exact length differ (max abs diff {max_diff}).")
        print(f"verify OK: fast == exact on {len(s_msgs)} rows (max abs diff 0)")

    # --- compute over all rows (parallel by row group) ---
    n_procs = min(args.num_procs, n_rg)
    print(f"computing with {n_procs} workers ...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_procs, initializer=_init, initargs=(args.model, template, args.truncate, args.max_length)) as pool:
        results = pool.map(_process_rg, [(input_path, rg) for rg in range(n_rg)])
    rows = [item for _, rg_items in sorted(results) for item in rg_items]
    assert len(rows) == n_rows, f"row count {len(rows)} != rows {n_rows}"

    report = {
        "input": input_path,
        "output": output_path,
        "model": args.model,
        "chat_template": args.chat_template or f"tokenizer-default:{args.model}",
        "max_length": args.max_length,
        "strict_less_than": True,
        "mode": "truncate" if args.truncate else "drop",
        "rows_in": n_rows,
    }

    if args.truncate:
        keep_ns = [keep_n for (keep_n, _, _) in rows]
        rows_out = sum(1 for kn in keep_ns if kn is not None)
        dropped = sum(1 for kn in keep_ns if kn is None)
        truncated = sum(1 for (kn, _, norig) in rows if kn is not None and kn < norig)
        kept_whole = sum(1 for (kn, _, norig) in rows if kn is not None and kn == norig)
        msgs_removed = sum((norig - kn) for (kn, _, norig) in rows if kn is not None)
        final_lens = sorted(fl for (kn, fl, _) in rows if kn is not None)
        pcts = _percentiles(final_lens, [50, 90, 95, 99, 100])
        _write_sliced(pf, keep_ns, output_path)
        report.update(
            {
                "rows_out": rows_out,
                "rows_dropped_first_step_over_budget": dropped,
                "rows_truncated": truncated,
                "rows_kept_whole": kept_whole,
                "messages_removed_total": msgs_removed,
                "final_length_percentiles": {f"p{p}": pcts[p] for p in [50, 90, 95, 99]}
                | {"max": pcts[100], "min": final_lens[0] if final_lens else None},
                "final_length_histogram": _histogram(final_lens, args.max_length),
            }
        )
        print("\n=== filter_by_length report (truncate) ===")
        print(f"rows_in:        {n_rows}")
        print(f"rows_out:       {rows_out}  ({100 * rows_out / n_rows:.1f}%)")
        print(f"  kept whole:   {kept_whole}")
        print(f"  truncated:    {truncated}")
        print(f"DROPPED (first step over budget): {dropped}")
        print(f"messages removed (total): {msgs_removed}")
        print(f"final length percentiles: {report['final_length_percentiles']}")
        print(f"final length histogram:   {report['final_length_histogram']}")
    else:
        lengths = rows
        mask = [length is not None and length < args.max_length for length in lengths]
        rows_out = 0
        too_long = sum(1 for length in lengths if length is not None and length >= args.max_length)
        no_assistant = sum(1 for length in lengths if length is None)
        # write via mask filter (exact passthrough, no round-trip)
        writer = None
        offset = 0
        for rg in range(n_rg):
            tbl = pf.read_row_group(rg)
            n = tbl.num_rows
            kept = tbl.filter(pa.array(mask[offset : offset + n], type=pa.bool_()))
            offset += n
            rows_out += kept.num_rows
            if kept.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(output_path, kept.schema)
                writer.write_table(kept)
        if writer is None:
            raise RuntimeError("No rows passed the length filter; refusing to write an empty dataset.")
        writer.close()
        present = sorted(length for length in lengths if length is not None)
        pcts = _percentiles(present, [50, 90, 95, 99, 100])
        report.update(
            {
                "rows_out": rows_out,
                "dropped_too_long": too_long,
                "dropped_no_trailing_assistant": no_assistant,
                "length_percentiles": {f"p{p}": pcts[p] for p in [50, 90, 95, 99]}
                | {"max": pcts[100], "min": present[0] if present else None},
                "length_histogram": _histogram(present, args.max_length),
            }
        )
        print("\n=== filter_by_length report (drop) ===")
        print(f"rows_in:  {n_rows}")
        print(f"rows_out: {rows_out}  ({100 * rows_out / n_rows:.1f}%)")
        print(f"DROPPED too long (>= {args.max_length}): {too_long}")
        print(f"DROPPED no trailing assistant: {no_assistant}")
        print(f"length percentiles: {report['length_percentiles']}")
        print(f"length histogram:   {report['length_histogram']}")

    with open(os.path.join(output_dir, "filter_by_length_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
