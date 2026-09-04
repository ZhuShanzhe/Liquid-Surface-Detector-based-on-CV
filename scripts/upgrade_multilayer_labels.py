#!/usr/bin/env python3
"""Upgrade existing simulator manifests with physically complete ray layers."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

import numpy as np

from liquid_depth.simulation import SyntheticScene, render_geometric_labels

DEFAULT_SCENARIOS = "transparent,translucent,multilayer,compound"


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def build_scene(metadata: dict[str, object]) -> SyntheticScene:
    names = {field.name for field in fields(SyntheticScene)}
    values = {name: metadata[name] for name in names}
    for name in (
        "camera_position_m",
        "camera_target_m",
        "liquid_color_rgb",
    ):
        values[name] = tuple(values[name])
    return SyntheticScene(**values)


def upgrade_one(
    task: tuple[int, str, str, bool],
) -> tuple[int, str, str, np.ndarray]:
    row_index, raw_metadata_path, raw_output_root, overwrite = task
    metadata_path = Path(raw_metadata_path)
    output_root = Path(raw_output_root)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scene = build_scene(metadata)
    labels = render_geometric_labels(scene)
    sample_name = metadata_path.parent.name
    shard = sample_name[:2] if len(sample_name) > 2 else "00"
    sample_output = output_root / shard / sample_name
    depth_path = sample_output / "layer_depths_v4_m.npz"
    valid_path = sample_output / "layer_valid_v4.npz"
    if overwrite or not (depth_path.exists() and valid_path.exists()):
        sample_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            depth_path,
            layer_depths_m=labels["layer_depths_m"].astype(np.float16),
        )
        np.savez_compressed(
            valid_path,
            layer_valid=labels["layer_valid"].astype(np.uint8),
        )
    counts = labels["layer_valid"].sum(axis=0).clip(0, 4).astype(np.int64)
    histogram = np.bincount(counts.ravel(), minlength=5)
    return row_index, str(depth_path), str(valid_path), histogram


def main() -> None:
    parser = argparse.ArgumentParser(description="Add liquid-surface plus container-bottom ray labels")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    manifest_root = manifest.parent
    output_manifest = args.output_manifest.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    scenarios = {value.strip().lower() for value in args.scenarios.split(",") if value.strip()}

    with manifest.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    required = {
        "metadata_path",
        "layer_depths_path",
        "layer_valid_path",
        "scenario",
    }
    missing = required - set(fieldnames)
    if missing:
        raise SystemExit(f"Manifest missing: {sorted(missing)}")

    selected_indices = [
        index for index, row in enumerate(rows) if row["scenario"].strip().lower() in scenarios
    ]
    if args.limit is not None:
        selected_indices = selected_indices[: args.limit]
    tasks = [
        (
            index,
            str(resolve(manifest_root, rows[index]["metadata_path"])),
            str(output_root),
            args.overwrite,
        )
        for index in selected_indices
    ]

    layer_histogram = np.zeros(5, dtype=np.int64)
    with ProcessPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        for upgraded, result in enumerate(
            executor.map(upgrade_one, tasks, chunksize=1),
            start=1,
        ):
            row_index, depth_path, valid_path, histogram = result
            rows[row_index]["layer_depths_path"] = os.path.relpath(
                depth_path,
                output_manifest.parent,
            )
            rows[row_index]["layer_valid_path"] = os.path.relpath(
                valid_path,
                output_manifest.parent,
            )
            layer_histogram += histogram
            if upgraded % 100 == 0:
                print(
                    json.dumps(
                        {
                            "upgraded": upgraded,
                            "selected": len(tasks),
                            "source_row": row_index + 1,
                        }
                    ),
                    flush=True,
                )

    temporary = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_manifest)
    total = int(layer_histogram.sum())
    valid_rays = int(layer_histogram[1:].sum())
    summary = {
        "source_manifest": str(manifest),
        "output_manifest": str(output_manifest),
        "output_root": str(output_root),
        "scenarios": sorted(scenarios),
        "upgraded_samples": len(tasks),
        "workers": args.workers,
        "layer_count_pixels": {str(index): int(value) for index, value in enumerate(layer_histogram)},
        "multilayer_pixel_ratio_all_pixels": (float(layer_histogram[2:].sum() / total) if total else 0.0),
        "multilayer_ratio_among_valid_rays": (float(layer_histogram[2:].sum() / max(valid_rays, 1))),
        "generator_version": "liquid_sim_v4_multilayer",
    }
    summary_path = output_manifest.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
