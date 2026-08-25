#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the common depth benchmark manifest from extracted DREDS STD data"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="test",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth frame within each physical sequence",
    )
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be at least one")

    rows = []
    mask_root = args.output.parent / f"{args.output.stem}_masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    sequence_dirs = sorted(path for path in args.root.iterdir() if path.is_dir())
    for sequence in sequence_dirs:
        split = "test" if sequence.name.startswith("test") else "train"
        if args.split != "all" and split != args.split:
            continue
        colors = sorted(sequence.glob("*_color.png"))[:: args.stride]
        for color in colors:
            frame_id = color.name.removesuffix("_color.png")
            raw_depth = sequence / f"{frame_id}_depth_415.exr"
            target_depth = sequence / f"{frame_id}_gt_depth.exr"
            source_mask = sequence / f"{frame_id}_mask.png"
            required = (raw_depth, target_depth, source_mask)
            if not all(path.is_file() for path in required):
                continue

            mask_image = cv2.imread(str(source_mask), cv2.IMREAD_UNCHANGED)
            if mask_image is None:
                raise FileNotFoundError(source_mask)
            binary_mask_path = mask_root / f"{sequence.name}_{frame_id}.npy"
            np.save(
                binary_mask_path,
                ((mask_image != 255).astype(np.uint8) * 255),
                allow_pickle=False,
            )
            rows.append(
                {
                    "frame_id": f"{sequence.name}_{frame_id}",
                    "sequence_id": sequence.name,
                    "rgb_path": str(color),
                    "raw_depth_path": str(raw_depth),
                    "target_depth_path": str(target_depth),
                    "mask_path": str(binary_mask_path),
                    "scenario": f"dreds_std_{split}",
                    "difficulty_tags": "transparent;specular;diffuse",
                    "depth_scale_to_m": "",
                }
            )

    if not rows:
        raise SystemExit(f"No complete DREDS frames found under {args.root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows from {len(sequence_dirs)} sequences to {args.output}")


if __name__ == "__main__":
    main()
