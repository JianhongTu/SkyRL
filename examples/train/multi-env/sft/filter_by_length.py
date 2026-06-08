#!/usr/bin/env python3
"""Filter the OpenHands SFT dataset to rows that fit within a token budget.

Third step of the multi-env SFT pipeline::

    prepare_openhands.py  ->  openhands.parquet
    generate_sft.py       ->  train.parquet          (messages + tools)
    filter_by_length.py   ->  train.parquet          (rows strictly under --max-length tokens)

Keeps only rows whose tokenized length is **strictly below** ``--max-length``
(default 32768), measured **exactly the way the SFT trainer tokenizes**.

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
whereas a single full render injects it at most once. A naive single render would
therefore *under*-count and let too-long rows slip past the filter only to be
truncated at train time. So we replicate the trainer's exact measure:

    length = len(apply_chat_template(messages[:first_assistant], tools=tools))
           + sum_over_later_messages( fixed_base_token_delta(message) )

Speedup (verified, not assumed)
-------------------------------
The ``<tools>`` block lives in the system message, so it cancels out of every
per-message fixed-base delta. We therefore pass ``tools=`` only on the leading
render and drop it from the per-message deltas (avoids re-tokenizing the ~2k-token
schema once per message). ``--verify-rows`` checks this fast path against the
exact with-tools encoder on a sample and aborts on any mismatch before trusting
it on the full dataset.

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
    # sft_trainer.py:376
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
    """Trainer-exact length, with the tools block dropped from per-message deltas."""
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


# --- worker plumbing --------------------------------------------------------
_TOK = None
_BASE_LEN_NO_TOOLS = None


def _init(model, chat_template):
    global _TOK, _BASE_LEN_NO_TOOLS
    from transformers import AutoTokenizer

    _TOK = AutoTokenizer.from_pretrained(model)
    if chat_template is not None:
        _TOK.chat_template = chat_template
    _BASE_LEN_NO_TOOLS = _apply_len(_TOK, _BASE)


def _process_rg(args):
    path, rg = args
    pf = pq.ParquetFile(path)
    tbl = pf.read_row_group(rg, columns=["messages", "tools"])
    msgs = tbl.column("messages").to_pylist()
    tools_raw = tbl.column("tools").to_pylist()
    lengths = [length_fast(m, _coerce_tools(t), _TOK, _BASE_LEN_NO_TOOLS) for m, t in zip(msgs, tools_raw)]
    return rg, lengths


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


def main():
    parser = argparse.ArgumentParser(description="Filter SFT rows to those strictly under a token budget.")
    parser.add_argument("--input", default="~/data/nemotron_sft_swe_v3_openhands_sft/train.parquet")
    parser.add_argument("--output-dir", default="~/data/nemotron_sft_swe_v3_openhands_sft_max32k")
    parser.add_argument("--output-file", default="train.parquet", help="Output filename (stem becomes the HF split).")
    parser.add_argument("--max-length", type=int, default=32768, help="Keep rows with length STRICTLY below this.")
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
    print(f"input: {input_path}  rows={n_rows}  row_groups={n_rg}")
    print(f"keep rows with tokenized length < {args.max_length} (strict)")

    tok, template = _load_tokenizer(args.model, args.chat_template)
    base_len_no_tools = _apply_len(tok, _BASE)

    # --- verify the fast path == the exact (with-tools) encoder on a sample ---
    if args.verify_rows > 0:
        sample = pf.read_row_group(0, columns=["messages", "tools"])
        msgs = sample.column("messages").to_pylist()[: args.verify_rows]
        tools_raw = sample.column("tools").to_pylist()[: args.verify_rows]
        max_diff = 0
        for m, t in zip(msgs, tools_raw):
            tools = _coerce_tools(t)
            lf = length_fast(m, tools, tok, base_len_no_tools)
            le = length_exact(m, tools, tok)
            if lf != le:
                max_diff = max(max_diff, abs((lf or 0) - (le or 0)))
        if max_diff != 0:
            raise SystemExit(f"VERIFY FAILED: fast vs exact length differ (max abs diff {max_diff}).")
        print(f"verify OK: fast == exact on {len(msgs)} rows (max abs diff 0)")

    # --- compute lengths over all rows (parallel by row group) ---
    n_procs = min(args.num_procs, n_rg)
    print(f"computing lengths with {n_procs} workers ...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_procs, initializer=_init, initargs=(args.model, template)) as pool:
        results = pool.map(_process_rg, [(input_path, rg) for rg in range(n_rg)])
    lengths = [length for _, rg_lengths in sorted(results) for length in rg_lengths]
    assert len(lengths) == n_rows, f"length count {len(lengths)} != rows {n_rows}"

    # --- build keep mask (strict) ---
    mask = [length is not None and length < args.max_length for length in lengths]
    rows_out = sum(mask)
    dropped_too_long = sum(1 for length in lengths if length is not None and length >= args.max_length)
    dropped_no_trailing_assistant = sum(1 for length in lengths if length is None)

    # --- write kept rows (streamed, schema preserved) ---
    writer = None
    offset = 0
    for rg in range(n_rg):
        tbl = pf.read_row_group(rg)
        n = tbl.num_rows
        kept = tbl.filter(pa.array(mask[offset : offset + n], type=pa.bool_()))
        offset += n
        if kept.num_rows:
            if writer is None:
                writer = pq.ParquetWriter(output_path, kept.schema)
            writer.write_table(kept)
    if writer is None:
        raise RuntimeError("No rows passed the length filter; refusing to write an empty dataset.")
    writer.close()

    # --- report ---
    present = sorted(length for length in lengths if length is not None)
    pcts = _percentiles(present, [50, 90, 95, 99, 100])
    buckets = Counter()
    edges = [
        (0, 4096),
        (4096, 8192),
        (8192, 16384),
        (16384, args.max_length),
        (args.max_length, 65536),
        (65536, 1 << 30),
    ]
    labels = ["<4k", "4-8k", "8-16k", f"16k-{args.max_length}", f"{args.max_length}-64k", ">=64k"]
    for length in present:
        for (lo, hi), lab in zip(edges, labels):
            if lo <= length < hi:
                buckets[lab] += 1
                break

    report = {
        "input": input_path,
        "output": output_path,
        "model": args.model,
        "chat_template": args.chat_template or f"tokenizer-default:{args.model}",
        "max_length": args.max_length,
        "strict_less_than": True,
        "rows_in": n_rows,
        "rows_out": rows_out,
        "dropped_too_long": dropped_too_long,
        "dropped_no_trailing_assistant": dropped_no_trailing_assistant,
        "length_percentiles": {f"p{p}": pcts[p] for p in [50, 90, 95, 99]}
        | {"max": pcts[100], "min": present[0] if present else None},
        "length_histogram": {lab: buckets.get(lab, 0) for lab in labels},
    }
    with open(os.path.join(output_dir, "filter_by_length_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== filter_by_length report ===")
    print(f"rows_in:  {report['rows_in']}")
    print(f"rows_out: {report['rows_out']}  ({100 * rows_out / n_rows:.1f}%)")
    print(f"DROPPED too long (>= {args.max_length}): {dropped_too_long}")
    print(f"DROPPED no trailing assistant (trainer would skip): {dropped_no_trailing_assistant}")
    print(f"length percentiles: {report['length_percentiles']}")
    print(f"length histogram:   {report['length_histogram']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
