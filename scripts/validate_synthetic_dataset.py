#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the project synthetic liquid dataset contract")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    root = manifest.parent
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    errors: list[str] = []
    scenario_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    range_counts: Counter[str] = Counter()
    coverage: defaultdict[str, list[float]] = defaultdict(list)
    target_depths: list[float] = []
    for row_index, row in enumerate(rows):
        try:
            rgb_path = resolve(root, row["rgb_path"])
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise ValueError(f"unreadable RGB: {rgb_path}")
            target = np.load(resolve(root, row["target_depth_path"]))
            raw = np.load(resolve(root, row["raw_depth_path"]))
            mask = np.load(resolve(root, row["mask_path"]))
            normal = np.load(resolve(root, row["normal_path"]))
            layers = np.load(resolve(root, row["layer_depths_path"]))
            layer_valid = np.load(resolve(root, row["layer_valid_path"]))
            height, width = target.shape
            if rgb.shape[:2] != (height, width):
                raise ValueError(f"RGB/label shape mismatch: {rgb.shape[:2]} vs {(height, width)}")
            if raw.shape != target.shape or mask.shape != target.shape:
                raise ValueError("depth/mask shape mismatch")
            if normal.shape != (*target.shape, 3):
                raise ValueError(f"normal shape {normal.shape}")
            if layers.shape != layer_valid.shape or layers.shape[1:] != target.shape:
                raise ValueError("layer depth/valid shape mismatch")
            inside = mask > 0
            if int(inside.sum()) < max(64, int(0.003 * mask.size)):
                raise ValueError("liquid surface mask is too small")
            valid_target = target[inside]
            if not np.all(np.isfinite(valid_target)) or np.any(valid_target <= 0):
                raise ValueError("invalid metric target depth")
            norm = np.linalg.norm(normal[inside], axis=1)
            if not np.allclose(norm, 1.0, atol=2e-3):
                raise ValueError("surface normals are not unit length")
            adjacent = (layer_valid[1:] > 0) & (layer_valid[:-1] > 0)
            if np.any((layers[1:] - layers[:-1])[adjacent] < -1e-5):
                raise ValueError("multi-layer depths are not ordered")
            scenario = row.get("scenario", "unknown")
            scenario_counts[scenario] += 1
            split_counts[row["split"]] += 1
            valid_raw = (raw > 0) & inside
            coverage[scenario].append(float(valid_raw.sum() / inside.sum()))
            median_depth = float(np.median(valid_target))
            target_depths.append(median_depth)
            if median_depth < 0.3:
                range_counts["0.1-0.3m"] += 1
            elif median_depth < 1.0:
                range_counts["0.3-1m"] += 1
            elif median_depth < 3.0:
                range_counts["1-3m"] += 1
            else:
                range_counts["3-10m"] += 1
        except Exception as exc:
            errors.append(f"row {row_index}: {exc}")
            if len(errors) >= args.max_errors:
                break
    if len(rows) < args.min_samples:
        errors.append(f"expected at least {args.min_samples} samples, found {len(rows)}")
    report = {
        "manifest": manifest.as_posix(),
        "samples": len(rows),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "range_counts": dict(range_counts),
        "median_surface_depth_m": float(np.median(target_depths)) if target_depths else None,
        "scenario_raw_coverage": {
            name: {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in sorted(coverage.items())
        },
        "errors": errors,
        "valid": not errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
