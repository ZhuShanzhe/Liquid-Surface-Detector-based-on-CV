#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from liquid_depth.specialist_training import specialize_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assign deterministic scenario augmentations for specialist training"
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profiles",
        default="standard,glare,depth_failure,low_light",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    profiles = tuple(
        value.strip()
        for value in args.profiles.split(",")
        if value.strip()
    )

    with args.input.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    if "augmentation_profile" not in fieldnames:
        fieldnames.append("augmentation_profile")
    output = specialize_rows(rows, profiles, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)
    counts = Counter(
        (row.get("split", ""), row["augmentation_profile"])
        for row in output
    )
    print(
        json.dumps(
            {
                "output": args.output.resolve().as_posix(),
                "rows": len(output),
                "counts": {
                    f"{split}/{profile}": count
                    for (split, profile), count in sorted(counts.items())
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
