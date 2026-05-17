#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="List R2E Docker images from SkyRL parquet data.")
    parser.add_argument("--data", default="/mnt/swe/data/r2e-skyrl/train.parquet")
    args = parser.parse_args()

    data_path = Path(args.data)
    df = pd.read_parquet(data_path)
    images: list[str] = []
    for instance in df["instance"]:
        if isinstance(instance, str):
            instance = json.loads(instance)
        image = instance["instance_id"]
        if image not in images:
            images.append(image)

    for image in images:
        print(image)


if __name__ == "__main__":
    main()
