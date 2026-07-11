#!/usr/bin/env python3
"""Build the RL train/validation parquets for the multi-env SWE phase.

Self-contained adaptation of skyrl-agent/data/swe_data.py (the canonical prep for
the SWEBench task), pinned to OUR conventions:
  - output under /home/tovi/data/r2e-all by default (the launcher's DATA_DIR)
  - a --val-size subsample knob, because eval runs the WHOLE val split every
    `eval_interval` steps -> 500 SWE-bench-Verified rollouts/eval is expensive.
  - a JSON report sidecar (same convention as the sft/ data scripts).

Schema the SWEBench task consumes (see skyrl_agent/tasks/swebench/utils.py and
agents/base.py::_get_data): each row needs a nested `instance` struct + a
`data_source` string. `instance.instance_id` is indexed as the docker image for
r2e-gym rows; for swe-bench rows the image is derived as sweb.eval.x86_64.<id>.

  TRAIN: R2E-Gym/R2E-Gym-Subset  -> data_source="r2e-gym"
         (renames docker_image->instance_id, repo_name->repo, commit_hash->base_commit;
          for r2e, instance_id IS the docker image ref, and reward comes from
          instance.expected_output_json + /testbed run_tests.sh)
  VAL:   princeton-nlp/SWE-bench_Verified[test] -> data_source="swe-bench"
         (graded via swebench make_test_spec; needs the standard test-spec fields)

Usage (inside the container / where `datasets` is available):
  uv run --isolated --with datasets --with tqdm \
      examples/train/multi-env/rl/prepare_rl_data.py
  # smoke test:  --train-size 64 --val-size 16
  # cheap eval:  --val-size 100
"""
import argparse
import json
import os
import random
from collections import defaultdict

from datasets import Dataset, load_dataset
from tqdm import tqdm

# r2e HF keys -> SWE-bench-style keys the task expects on `instance`.
_R2E_RENAME = {"docker_image": "instance_id", "repo_name": "repo", "commit_hash": "base_commit"}


def build_train(dataset_name, limit):
    """R2E-Gym subset -> list of task rows (data_source='r2e-gym')."""
    ds = load_dataset(dataset_name)["train"]
    rows, repo_count, skipped_empty = [], defaultdict(int), 0
    for entry in tqdm(ds, desc="train (r2e-gym)"):
        entry = dict(entry)
        for old, new in _R2E_RENAME.items():
            if old in entry:
                entry[new] = entry.pop(old)
        if not entry.get("problem_statement"):
            skipped_empty += 1
            continue
        rows.append(
            {
                "prompt": entry["problem_statement"],
                "data_source": "r2e-gym",
                "ability": "coding",
                "instance": entry,
            }
        )
        repo_count[entry.get("repo", "?")] += 1
        if limit and len(rows) >= limit:
            break
    return rows, {"source_rows": len(ds), "skipped_empty": skipped_empty, "repos": len(repo_count)}


def build_val(dataset_name, size, seed):
    """SWE-bench Verified test split -> list of task rows (data_source='swe-bench')."""
    ds = load_dataset(dataset_name)["test"]
    idxs = list(range(len(ds)))
    if size and size < len(ds):
        # Deterministic subsample so eval is comparable across runs.
        rng = random.Random(seed)
        idxs = sorted(rng.sample(idxs, size))
    rows = []
    for i in tqdm(idxs, desc="val (swe-bench)"):
        entry = dict(ds[i])
        rows.append(
            {
                "prompt": entry["problem_statement"],
                "data_source": "swe-bench",
                "ability": "coding",
                "instance": entry,
            }
        )
    return rows, {"source_rows": len(ds), "sampled": len(rows), "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dataset", default="R2E-Gym/R2E-Gym-Subset")
    ap.add_argument("--val-dataset", default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--output-dir", default="/home/tovi/data/r2e-all")
    ap.add_argument("--train-size", type=int, default=None, help="cap train rows (smoke test)")
    ap.add_argument(
        "--val-size",
        type=int,
        default=100,
        help="subsample the val split (whole split is evaluated every eval_interval; "
        "500 is expensive). Use 0 for the full split.",
    )
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_rows, train_stats = build_train(args.train_dataset, args.train_size)
    Dataset.from_list(train_rows).to_parquet(os.path.join(args.output_dir, "train.parquet"))
    print(f"[train] wrote {len(train_rows)} rows  ({train_stats})")

    val_rows, val_stats = build_val(args.val_dataset, args.val_size or None, args.seed)
    Dataset.from_list(val_rows).to_parquet(os.path.join(args.output_dir, "validation.parquet"))
    print(f"[val]   wrote {len(val_rows)} rows  ({val_stats})")

    report = {
        "train_dataset": args.train_dataset,
        "val_dataset": args.val_dataset,
        "output_dir": args.output_dir,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_stats": train_stats,
        "val_stats": val_stats,
    }
    report_path = os.path.join(args.output_dir, "prepare_rl_data_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] {report_path}")


if __name__ == "__main__":
    main()
