#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from liquid_depth.evaluation_manifest import (
    EvaluationThresholds,
    cap_records_per_source_bucket,
    difficulty_buckets,
    stable_fraction,
)
from liquid_depth.scenario_policy import SceneSignals, measure_scene_signals

FIELDS = (
    "record_id",
    "dataset",
    "split",
    "task",
    "bucket",
    "scene_id",
    "frame_id",
    "rgb_path",
    "raw_depth_path",
    "target_depth_path",
    "mask_path",
    "parquet_path",
    "parquet_row_index",
    "difficulty_tags",
    "luma_p50",
    "dark_pixel_ratio",
    "saturated_pixel_ratio",
    "dynamic_range",
    "raw_depth_valid_ratio",
    "depth_scale_to_m",
    "metric_supervision",
    "relative_layer_supervision",
    "source_manifest",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _depth(path: Path) -> np.ndarray:
    value = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise FileNotFoundError(path)
    if value.ndim == 3:
        counts = [
            int((np.isfinite(value[..., index]) & (value[..., index] > 0)).sum())
            for index in range(value.shape[2])
        ]
        value = value[..., int(np.argmax(counts))]
    return value


def _mask_roi(
    mask_path: Path | None,
) -> tuple[int, int, int, int] | None:
    if mask_path is None:
        return None
    mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(mask_path)
    active = (
        np.any(mask[..., :3] != 0, axis=2)
        if mask.ndim == 3
        else mask != 0
    )
    rows, columns = np.nonzero(active)
    if not len(rows):
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _signals(
    rgb_path: Path,
    depth_path: Path,
    scale: float | None,
    mask_path: Path | None = None,
) -> SceneSignals:
    rgb = cv2.imread(rgb_path.as_posix(), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    return measure_scene_signals(
        rgb,
        _depth(depth_path),
        roi=_mask_roi(mask_path),
        depth_scale_to_m=scale,
        max_depth_m=10.0,
    )


def _signal_fields(signals: SceneSignals) -> dict[str, str]:
    return {
        name: f"{value:.8f}"
        for name, value in signals.to_dict().items()
    }


def _base_record(**values: object) -> dict[str, str]:
    record = {field: "" for field in FIELDS}
    record.update({key: str(value) for key, value in values.items()})
    return record


def _candidate(
    row: dict[str, str],
    candidate_stride: int,
    seed: int,
) -> bool:
    key = row.get("frame_id", "")
    return int(stable_fraction(key, seed) * candidate_stride) == 0


def add_multitask(
    records: list[dict[str, str]],
    manifest: Path,
    *,
    candidate_stride: int,
    seed: int,
    thresholds: EvaluationThresholds,
) -> None:
    candidates = [
        row
        for row in _read_csv(manifest)
        if row.get("split") in {"val", "test"}
        and _candidate(row, candidate_stride, seed)
    ]
    for row in candidates:
        scale_text = row.get("depth_scale_to_m", "").strip()
        signals = _signals(
            Path(row["rgb_path"]),
            Path(row["raw_depth_path"]),
            float(scale_text) if scale_text else None,
            Path(row["mask_path"]),
        )
        buckets = difficulty_buckets(
            signals,
            row.get("difficulty_tags", ""),
            thresholds=thresholds,
        )
        for bucket in buckets:
            frame_id = row.get("frame_id") or Path(row["rgb_path"]).stem
            record = _base_record(
                record_id=(
                    f"{row.get('dataset', 'unknown')}:{bucket}:{frame_id}"
                ),
                dataset=row.get("dataset", "unknown"),
                split=row["split"],
                task="metric_depth_restoration",
                bucket=bucket,
                scene_id=row.get("sequence_id", ""),
                frame_id=frame_id,
                rgb_path=row["rgb_path"],
                raw_depth_path=row["raw_depth_path"],
                target_depth_path=row["target_depth_path"],
                mask_path=row["mask_path"],
                difficulty_tags=row.get("difficulty_tags", ""),
                depth_scale_to_m=scale_text,
                metric_supervision="1",
                relative_layer_supervision="0",
                source_manifest=manifest.resolve().as_posix(),
            )
            record.update(_signal_fields(signals))
            records.append(record)


def add_trade(
    records: list[dict[str, str]],
    manifest: Path,
    *,
    candidate_stride: int,
    seed: int,
    thresholds: EvaluationThresholds,
) -> None:
    candidates = [
        row
        for row in _read_csv(manifest)
        if row.get("split") in {"val", "test"}
        and _candidate(row, candidate_stride, seed)
    ]
    for row in candidates:
        if row.get("luma_p50", "").strip():
            raw = _depth(Path(row["raw_depth_path"]))
            valid = np.isfinite(raw) & (raw > 0)
            signals = SceneSignals(
                raw_depth_valid_ratio=float(valid.mean()),
                luma_p50=float(row["luma_p50"]),
                dark_pixel_ratio=float(row["dark_pixel_ratio"]),
                saturated_pixel_ratio=float(row["saturated_pixel_ratio"]),
                dynamic_range=float(row["dynamic_range"]),
            )
        else:
            signals = _signals(
                Path(row["rgb_path"]),
                Path(row["raw_depth_path"]),
                0.001,
            )
        buckets = difficulty_buckets(
            signals,
            row.get("difficulty_tags", ""),
            thresholds=thresholds,
        )
        for bucket in buckets:
            record = _base_record(
                record_id=f"trade:{bucket}:{row['frame_id']}",
                dataset="trade_real",
                split=row["split"],
                task="real_rgbd_stress",
                bucket=bucket,
                scene_id=row.get("scene_id", ""),
                frame_id=row["frame_id"],
                rgb_path=row["rgb_path"],
                raw_depth_path=row["raw_depth_path"],
                difficulty_tags=row.get("difficulty_tags", ""),
                metric_supervision="0",
                relative_layer_supervision="0",
                source_manifest=manifest.resolve().as_posix(),
            )
            record.update(_signal_fields(signals))
            records.append(record)


def add_layereddepth(
    records: list[dict[str, str]],
    root: Path,
) -> None:
    import pyarrow.parquet as pq

    for shard in sorted((root / "data").glob("*.parquet")):
        count = pq.ParquetFile(shard).metadata.num_rows
        for row_index in range(count):
            frame_id = f"{shard.stem}:{row_index}"
            signals = SceneSignals(1.0, 0.5, 0.0, 0.0, 0.5)
            record = _base_record(
                record_id=(
                    "layereddepth:transparent_multilayer:"
                    f"{frame_id}"
                ),
                dataset="layereddepth_real",
                split="test",
                task="relative_multilayer_depth",
                bucket="transparent_multilayer",
                scene_id=shard.stem,
                frame_id=frame_id,
                parquet_path=shard.resolve().as_posix(),
                parquet_row_index=row_index,
                difficulty_tags=(
                    "transparent;multi_layer;non_lambertian"
                ),
                metric_supervision="0",
                relative_layer_supervision="1",
                source_manifest=root.resolve().as_posix(),
            )
            record.update(_signal_fields(signals))
            records.append(record)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the scene-stratified robustness evaluation manifest"
        )
    )
    parser.add_argument(
        "--multitask-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--trade-manifest", type=Path)
    parser.add_argument("--layereddepth-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-per-source-bucket",
        type=int,
        default=256,
    )
    parser.add_argument("--candidate-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.candidate_stride < 1:
        raise ValueError("candidate stride must be positive")

    thresholds = EvaluationThresholds()
    records: list[dict[str, str]] = []
    add_multitask(
        records,
        args.multitask_manifest.resolve(),
        candidate_stride=args.candidate_stride,
        seed=args.seed,
        thresholds=thresholds,
    )
    if args.trade_manifest:
        add_trade(
            records,
            args.trade_manifest.resolve(),
            candidate_stride=args.candidate_stride,
            seed=args.seed,
            thresholds=thresholds,
        )
    if args.layereddepth_root:
        add_layereddepth(records, args.layereddepth_root.resolve())
    records = cap_records_per_source_bucket(
        records,
        args.max_per_source_bucket,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    counts = Counter(
        (row["dataset"], row["bucket"])
        for row in records
    )
    summary = {
        "output": args.output.resolve().as_posix(),
        "records": len(records),
        "counts": {
            f"{dataset}/{bucket}": count
            for (dataset, bucket), count in sorted(counts.items())
        },
        "thresholds": thresholds.__dict__,
        "contracts": {
            "metric_depth_restoration": (
                "metric depth error may be reported"
            ),
            "real_rgbd_stress": (
                "robustness/coverage only; no metric target"
            ),
            "relative_multilayer_depth": (
                "relative layer ordering only; never metric liquid height"
            ),
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
