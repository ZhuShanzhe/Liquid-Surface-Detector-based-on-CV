#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

FIELDS = (
    "frame_id",
    "split",
    "scene_id",
    "rgb_path",
    "raw_depth_path",
    "camera_path",
    "camera_pose_path",
    "object_pose_path",
    "objects_json",
    "partial_fill_count",
    "difficulty_tags",
    "luma_p50",
    "dark_pixel_ratio",
    "saturated_pixel_ratio",
    "dynamic_range",
)


def scene_split(
    scene_id: str,
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> str:
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions cannot be negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")
    digest = hashlib.sha256(f"{seed}:{scene_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "val"
    return "train"


def load_objects(path: Path) -> list[dict[str, object]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(payload, list):
        raise TypeError(f"Expected a list in {path}")
    objects = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        fill = item.get("fill")
        objects.append(
            {
                "id": str(item.get("id", "unknown")),
                "fill": float(fill) if fill is not None else None,
            }
        )
    return objects


def exposure_metrics(path: Path) -> dict[str, str]:
    import cv2

    image = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    height, width = image.shape[:2]
    scale = min(1.0, 320.0 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    p10, p50, p90 = np.percentile(luma, (10, 50, 90))
    return {
        "luma_p50": f"{float(p50):.8f}",
        "dark_pixel_ratio": f"{float((luma <= 20.0 / 255.0).mean()):.8f}",
        "saturated_pixel_ratio": f"{float((luma >= 250.0 / 255.0).mean()):.8f}",
        "dynamic_range": f"{float(p90 - p10):.8f}",
    }


def build_rows(
    root: Path,
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    measure_exposure: bool,
) -> list[dict[str, str]]:
    scenes_root = root / "scenes"
    if not scenes_root.is_dir():
        raise FileNotFoundError(f"TRADE scenes directory is missing: {scenes_root}")
    rows = []
    for scene in sorted(path for path in scenes_root.iterdir() if path.is_dir()):
        camera_path = scene / "camera_d435.yaml"
        camera_pose_path = scene / "groundtruth_handeye.txt"
        object_pose_path = scene / "poses.yaml"
        for required in (camera_path, camera_pose_path, object_pose_path):
            if not required.is_file():
                raise FileNotFoundError(f"Missing TRADE metadata: {required}")
        objects = load_objects(object_pose_path)
        partial_fill_count = sum(
            item["fill"] is not None and 0.0 < float(item["fill"]) < 0.999
            for item in objects
        )
        tags = ["transparent", "non_lambertian", "container_edge"]
        if partial_fill_count:
            tags.append("liquid_fill_level")
        split = scene_split(
            scene.name,
            seed=seed,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        rgb_paths = sorted((scene / "rgb").glob("*.png"))
        if not rgb_paths:
            raise ValueError(f"No RGB frames in {scene}")
        for rgb_path in rgb_paths:
            depth_path = scene / "depth" / rgb_path.name
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing paired depth frame: {depth_path}")
            row = {
                "frame_id": f"trade_{scene.name}_{rgb_path.stem}",
                "split": split,
                "scene_id": scene.name,
                "rgb_path": rgb_path.resolve().as_posix(),
                "raw_depth_path": depth_path.resolve().as_posix(),
                "camera_path": camera_path.resolve().as_posix(),
                "camera_pose_path": camera_pose_path.resolve().as_posix(),
                "object_pose_path": object_pose_path.resolve().as_posix(),
                "objects_json": json.dumps(objects, separators=(",", ":")),
                "partial_fill_count": str(partial_fill_count),
                "difficulty_tags": ";".join(tags),
                "luma_p50": "",
                "dark_pixel_ratio": "",
                "saturated_pixel_ratio": "",
                "dynamic_range": "",
            }
            if measure_exposure:
                row.update(exposure_metrics(rgb_path))
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a scene-disjoint TRADE RGB-D stress-test manifest"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--measure-exposure", action="store_true")
    args = parser.parse_args()
    rows = build_rows(
        args.root.resolve(),
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        measure_exposure=args.measure_exposure,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "val", "test")
    }
    scene_counts = {
        split: len({row["scene_id"] for row in rows if row["split"] == split})
        for split in ("train", "val", "test")
    }
    print(
        json.dumps(
            {
                "output": args.output.resolve().as_posix(),
                "frames": len(rows),
                "frame_split_counts": split_counts,
                "scene_split_counts": scene_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
