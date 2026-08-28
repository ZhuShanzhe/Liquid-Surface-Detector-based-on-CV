#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


PATH_COLUMNS = (
    "rgb_path",
    "raw_depth_path",
    "target_depth_path",
    "mask_path",
    "normal_path",
    "uncertainty_path",
    "layer_depths_path",
    "layer_valid_path",
    "metadata_path",
)
REQUIRED = (
    "rgb_path",
    "raw_depth_path",
    "target_depth_path",
    "mask_path",
    "normal_path",
    "split",
    "sequence_id",
    "difficulty_tags",
    "depth_scale_to_m",
)


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def canonical_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    source_name = manifest.stem
    for row in rows:
        row.setdefault("normal_path", "")
        row.setdefault("depth_scale_to_m", "")
        row.setdefault("difficulty_tags", "ordinary")
        row.setdefault("sequence_id", row.get("frame_id", source_name))
        row.setdefault("dataset", source_name)
        row.setdefault("scenario", row.get("dataset", source_name))
        row["source_manifest"] = manifest.as_posix()
        for column in PATH_COLUMNS:
            value = row.get(column, "").strip()
            if value and not Path(value).is_absolute():
                row[column] = (manifest.parent / value).resolve().as_posix()
        missing = [name for name in REQUIRED if name not in row]
        if missing:
            raise ValueError(f"{manifest} is missing {', '.join(missing)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic mixed synthetic/research manifest")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-per-dataset", type=int, default=2000)
    parser.add_argument("--max-val-per-dataset", type=int, default=500)
    parser.add_argument("--max-test-per-dataset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    grouped: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for manifest in args.input:
        for row in canonical_rows(manifest.resolve()):
            grouped[(row.get("dataset", manifest.stem), row["split"])].append(row)
    limits = {
        "train": args.max_train_per_dataset,
        "val": args.max_val_per_dataset,
        "test": args.max_test_per_dataset,
    }
    selected: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for (dataset, split), rows in sorted(grouped.items()):
        limit = limits.get(split, 0)
        if limit <= 0:
            continue
        rng = random.Random(stable_seed(f"{dataset}:{split}", args.seed))
        rows = list(rows)
        rng.shuffle(rows)
        kept = rows[:limit]
        selected.extend(kept)
        counts[f"{dataset}/{split}"] = len(kept)
    fieldnames = list(REQUIRED)
    for row in selected:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    print(
        json.dumps(
            {
                "output": args.output.resolve().as_posix(),
                "rows": len(selected),
                "counts": counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
