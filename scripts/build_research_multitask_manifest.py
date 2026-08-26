#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

FIELDS = (
    "frame_id",
    "dataset",
    "rgb_path",
    "raw_depth_path",
    "target_depth_path",
    "mask_path",
    "normal_path",
    "normal_channel_order",
    "split",
    "sequence_id",
    "difficulty_tags",
    "depth_scale_to_m",
    "corrupt_depth_in_mask",
)


def complete(paths: tuple[Path, ...]) -> bool:
    return all(path.is_file() for path in paths)


def add_cleargrasp_synthetic(rows: list[dict[str, str]], root: Path, split: str, stride: int) -> None:
    for sequence in sorted(path for path in root.iterdir() if path.is_dir()):
        for rgb in sorted((sequence / "rgb-imgs").glob("*-rgb.jpg"))[::stride]:
            frame = rgb.name.removesuffix("-rgb.jpg")
            depth = sequence / "depth-imgs-rectified" / f"{frame}-depth-rectified.exr"
            mask = sequence / "segmentation-masks" / f"{frame}-segmentation-mask.png"
            normal = sequence / "camera-normals" / f"{frame}-cameraNormals.exr"
            if not complete((rgb, depth, mask, normal)):
                continue
            rows.append(
                {
                    "frame_id": f"cleargrasp_{split}_{sequence.name}_{frame}",
                    "dataset": "cleargrasp_synthetic",
                    "rgb_path": str(rgb),
                    "raw_depth_path": str(depth),
                    "target_depth_path": str(depth),
                    "mask_path": str(mask),
                    "normal_path": str(normal),
                    "normal_channel_order": "bgr",
                    "split": split,
                    "sequence_id": f"cleargrasp_{split}_{sequence.name}",
                    "difficulty_tags": "transparent;container_edge;non_lambertian",
                    "depth_scale_to_m": "",
                    "corrupt_depth_in_mask": "1",
                }
            )


def add_cleargrasp_real(rows: list[dict[str, str]], root: Path, stride: int) -> None:
    for sensor in sorted(path for path in root.iterdir() if path.is_dir()):
        for rgb in sorted(sensor.glob("*-transparent-rgb-img.jpg"))[::stride]:
            frame = rgb.name.removesuffix("-transparent-rgb-img.jpg")
            raw = sensor / f"{frame}-transparent-depth-img.exr"
            target = sensor / f"{frame}-opaque-depth-img.exr"
            mask = sensor / f"{frame}-mask.png"
            if not complete((rgb, raw, target, mask)):
                continue
            rows.append(
                {
                    "frame_id": f"cleargrasp_real_{sensor.name}_{frame}",
                    "dataset": "cleargrasp_real",
                    "rgb_path": str(rgb),
                    "raw_depth_path": str(raw),
                    "target_depth_path": str(target),
                    "mask_path": str(mask),
                    "normal_path": "",
                    "normal_channel_order": "",
                    "split": "test",
                    "sequence_id": f"cleargrasp_real_{sensor.name}",
                    "difficulty_tags": "transparent;glare;container_edge;non_lambertian",
                    "depth_scale_to_m": "",
                    "corrupt_depth_in_mask": "0",
                }
            )


def add_transcg(rows: list[dict[str, str]], root: Path, stride: int) -> None:
    metadata = json.loads((root / "metadata.json").read_text())
    train_scenes = set(map(int, metadata["train"]))
    test_scenes = set(map(int, metadata["test"]))
    for scene in sorted(root.glob("scene*"), key=lambda path: int(path.name.removeprefix("scene"))):
        scene_number = int(scene.name.removeprefix("scene"))
        if scene_number in train_scenes:
            split = "train"
        elif scene_number in test_scenes:
            split = "val" if scene_number % 2 else "test"
        else:
            continue
        frames = sorted(
            (path for path in scene.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        )[::stride]
        for frame in frames:
            for camera in ("1", "2"):
                rgb = frame / f"rgb{camera}.png"
                raw = frame / f"depth{camera}.png"
                target = frame / f"depth{camera}-gt.png"
                mask = frame / f"depth{camera}-gt-mask.png"
                normal = frame / f"depth{camera}-gt-sn.png"
                if not complete((rgb, raw, target, mask, normal)):
                    continue
                rows.append(
                    {
                        "frame_id": f"transcg_{scene.name}_{frame.name}_c{camera}",
                        "dataset": "transcg",
                        "rgb_path": str(rgb),
                        "raw_depth_path": str(raw),
                        "target_depth_path": str(target),
                        "mask_path": str(mask),
                        "normal_path": str(normal),
                        "normal_channel_order": "bgr",
                        "split": split,
                        "sequence_id": f"transcg_{scene.name}",
                        "difficulty_tags": "transparent;container_edge;non_lambertian",
                        "depth_scale_to_m": "0.001",
                        "corrupt_depth_in_mask": "0",
                    }
                )


def add_todd(rows: list[dict[str, str]], root: Path, stride: int) -> None:
    for split in ("train", "val", "test"):
        split_root = root / split / split
        samples = sorted(path for path in split_root.iterdir() if path.is_dir())[::stride]
        for sample in samples:
            rgb = sample / "image.jpg"
            raw = sample / "depth.exr"
            target = sample / "detph_GroundTruth.exr"
            mask = sample / "instance_segment.png"
            if not complete((rgb, raw, target, mask)):
                continue
            rows.append(
                {
                    "frame_id": f"todd_{split}_{sample.name}",
                    "dataset": "todd",
                    "rgb_path": str(rgb),
                    "raw_depth_path": str(raw),
                    "target_depth_path": str(target),
                    "mask_path": str(mask),
                    "normal_path": "",
                    "normal_channel_order": "",
                    "split": split,
                    "sequence_id": f"todd_{split}",
                    "difficulty_tags": "transparent;translucent;container_edge;non_lambertian",
                    "depth_scale_to_m": "",
                    "corrupt_depth_in_mask": "0",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical research multi-task manifest")
    parser.add_argument(
        "--research-root",
        type=Path,
        default=Path("/root/autodl-tmp/liquid-depth-data/research"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transcg-stride", type=int, default=1)
    parser.add_argument("--cleargrasp-stride", type=int, default=1)
    parser.add_argument("--todd-stride", type=int, default=1)
    args = parser.parse_args()
    if min(args.transcg_stride, args.cleargrasp_stride, args.todd_stride) < 1:
        raise ValueError("Dataset strides must be positive")

    root = args.research_root.resolve()
    clear = root / "cleargrasp" / "extracted"
    rows: list[dict[str, str]] = []
    add_transcg(rows, root / "transcg" / "extracted_full" / "transcg", args.transcg_stride)
    add_cleargrasp_synthetic(
        rows,
        clear / "cleargrasp-dataset-train" / "cleargrasp-dataset-train",
        "train",
        args.cleargrasp_stride,
    )
    clear_eval = clear / "cleargrasp-dataset-test-val" / "cleargrasp-dataset-test-val"
    add_cleargrasp_synthetic(rows, clear_eval / "synthetic-val", "val", args.cleargrasp_stride)
    add_cleargrasp_synthetic(rows, clear_eval / "synthetic-test", "test", args.cleargrasp_stride)
    add_cleargrasp_real(rows, clear_eval / "real-test", args.cleargrasp_stride)
    add_todd(rows, root / "todd" / "extracted", args.todd_stride)

    if not rows:
        raise SystemExit(f"No complete samples found under {root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter((row["dataset"], row["split"]) for row in rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    for (dataset, split), count in sorted(counts.items()):
        print(f"  {dataset:24s} {split:5s} {count:7d}")


if __name__ == "__main__":
    main()
