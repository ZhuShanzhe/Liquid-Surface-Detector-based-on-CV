#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a common manifest from ClearGrasp real-test data")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be at least one")

    rows = []
    for sensor_root in sorted(path for path in args.root.iterdir() if path.is_dir()):
        rgb_files = sorted(sensor_root.glob("*-transparent-rgb-img.jpg"))[:: args.stride]
        for rgb_path in rgb_files:
            frame_id = rgb_path.name.removesuffix("-transparent-rgb-img.jpg")
            raw_depth = sensor_root / f"{frame_id}-transparent-depth-img.exr"
            target_depth = sensor_root / f"{frame_id}-opaque-depth-img.exr"
            mask = sensor_root / f"{frame_id}-mask.png"
            if not all(path.is_file() for path in (raw_depth, target_depth, mask)):
                continue
            rows.append(
                {
                    "frame_id": f"cleargrasp_{sensor_root.name}_{frame_id}",
                    "sequence_id": sensor_root.name,
                    "rgb_path": str(rgb_path),
                    "raw_depth_path": str(raw_depth),
                    "target_depth_path": str(target_depth),
                    "mask_path": str(mask),
                    "scenario": f"cleargrasp_real_test_{sensor_root.name}",
                    "difficulty_tags": "transparent;boundary;non_lambertian",
                    "depth_scale_to_m": "",
                }
            )
    if not rows:
        raise SystemExit(f"No complete ClearGrasp real-test frames found under {args.root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} ClearGrasp real-test rows to {args.output}")


if __name__ == "__main__":
    main()
