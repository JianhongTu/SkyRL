#!/usr/bin/env python3
"""Prepare SkyRL-Agent parquet data for the custom R2E baseline.

This follows the original SkyRL-Agent recipe: R2E-Gym is used for training and
SWE-bench Verified is used for validation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from tqdm import tqdm


FIELD_MAPPING = {
    "docker_image": "instance_id",
    "repo_name": "repo",
    "commit_hash": "base_commit",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="R2E-Gym/R2E-Gym-Subset", help="Hugging Face dataset id.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face dataset revision.")
    parser.add_argument("--split", default="train", help="Source split to read from the R2E training dataset.")
    parser.add_argument("--output", default="dataset/r2e-skyrl", help="Output directory for parquet files.")
    parser.add_argument("--seed", type=int, default=20260517, help="Deterministic train sampling seed.")
    parser.add_argument("--train-size", type=int, default=None, help="Optional number of train examples.")
    parser.add_argument(
        "--validation-dataset",
        default="princeton-nlp/SWE-bench_Verified",
        help="Validation dataset id, matching the original SkyRL-Agent recipe.",
    )
    parser.add_argument("--validation-split", default="test", help="Validation source split.")
    parser.add_argument(
        "--validation-size",
        type=int,
        default=None,
        help="Optional number of validation examples from SWE-bench Verified.",
    )
    return parser.parse_args()


def normalize_instance(row: dict[str, Any]) -> dict[str, Any]:
    instance = dict(row)
    for old_key, new_key in FIELD_MAPPING.items():
        if old_key in instance:
            instance[new_key] = instance.pop(old_key)
    return instance


def get_task_id(instance: dict[str, Any], fallback_idx: int) -> str:
    for key in ("instance_id", "task_id", "id", "repo"):
        value = instance.get(key)
        if value:
            return str(value)
    return f"r2e-{fallback_idx}"


def to_skyrl_row(instance: dict[str, Any], data_source: str) -> dict[str, Any] | None:
    problem_statement = instance.get("problem_statement")
    if not problem_statement:
        return None
    return {
        "prompt": problem_statement,
        "data_source": data_source,
        "ability": "coding",
        "instance": instance,
    }


def write_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output_path))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()

    dataset_kwargs = {"trust_remote_code": True}
    if args.revision:
        dataset_kwargs["revision"] = args.revision
    train_source = load_dataset(args.dataset, **dataset_kwargs)
    train_source_split = train_source[args.split]

    rows_with_ids: list[tuple[str, dict[str, Any]]] = []
    for idx, row in enumerate(tqdm(train_source_split, desc=f"Reading {args.dataset}:{args.split}")):
        instance = normalize_instance(row)
        skyrl_row = to_skyrl_row(instance, data_source="r2e-gym")
        if skyrl_row is None:
            continue
        rows_with_ids.append((get_task_id(instance, idx), skyrl_row))

    if not rows_with_ids:
        raise ValueError(f"No usable R2E rows found in {args.dataset}:{args.split}")

    rng = random.Random(args.seed)
    rng.shuffle(rows_with_ids)

    train = rows_with_ids
    if args.train_size is not None:
        train = train[: args.train_size]

    if not train:
        raise ValueError("Train split is empty. Increase --train-size or check the source dataset.")

    train_ids = [task_id for task_id, _ in train]

    validation_source = load_dataset(args.validation_dataset)
    validation_source_split = validation_source[args.validation_split]
    validation_rows: list[tuple[str, dict[str, Any]]] = []
    for idx, row in enumerate(
        tqdm(validation_source_split, desc=f"Reading {args.validation_dataset}:{args.validation_split}")
    ):
        instance = dict(row)
        skyrl_row = to_skyrl_row(instance, data_source="swe-bench")
        if skyrl_row is None:
            continue
        validation_rows.append((get_task_id(instance, idx), skyrl_row))

    if args.validation_size is not None:
        validation_rows = validation_rows[: args.validation_size]

    if not validation_rows:
        raise ValueError("Validation split is empty. Check --validation-dataset and --validation-split.")

    validation_ids = [task_id for task_id, _ in validation_rows]

    write_parquet([row for _, row in train], output_dir / "train.parquet")
    write_parquet([row for _, row in validation_rows], output_dir / "validation.parquet")

    manifest = {
        "train_dataset": args.dataset,
        "train_revision": args.revision,
        "train_source_split": args.split,
        "validation_dataset": args.validation_dataset,
        "validation_source_split": args.validation_split,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "train_count": len(train),
        "validation_count": len(validation_rows),
        "train_ids": train_ids,
        "validation_ids": validation_ids,
    }
    with open(output_dir / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write(os.linesep)

    print(f"Wrote {len(train)} R2E train rows and {len(validation_rows)} SWE-bench validation rows to {output_dir}")


if __name__ == "__main__":
    main()
