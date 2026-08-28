#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from liquid_depth.config import load_config
from liquid_depth.depth_evaluation import evaluate_depth_manifest, select_depth_channel
from liquid_depth.refinement import make_depth_refiner


def read_array(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    value = cv2.imread(str(path), flags)
    if value is None:
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".exr":
        value = select_depth_channel(value)
    return value


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "frame"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one configured depth-restoration backend under the common metric contract"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument(
        "--backend",
        choices=("identity", "transcg_dfnet", "dreds_swindrnet", "cleargrasp", "torchscript"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--preserve-valid-raw", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    root = manifest.parent
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"rgb_path", "raw_depth_path", "target_depth_path", "mask_path"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"Manifest requires columns: {', '.join(sorted(required))}")
    source_row_count = len(rows)
    rows = [
        row
        for row in rows
        if all(row.get(key, "").strip() for key in required)
    ]
    if not rows:
        raise ValueError("Manifest contains no metric-supervised depth rows")
    metric_row_count = len(rows)
    if args.limit is not None:
        rows = rows[: args.limit]

    config = load_config(args.config)
    config["depth_refinement"]["backend"] = args.backend
    if args.preserve_valid_raw:
        config["depth_refinement"]["torchscript"][
            "preserve_valid_raw"
        ] = True
    if args.model_path:
        config["depth_refinement"]["torchscript"]["model_path"] = (
            args.model_path.resolve().as_posix()
        )
    load_start = time.perf_counter()
    refiner = make_depth_refiner(config)
    model_load_seconds = time.perf_counter() - load_start
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    latencies_ms = []
    failures = []
    for index, row in enumerate(rows):
        rgb_path = resolve(root, row["rgb_path"])
        raw_depth_path = resolve(root, row["raw_depth_path"])
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(rgb_path)
        raw_depth = read_array(raw_depth_path)
        target_shape = read_array(
            resolve(root, row["target_depth_path"])
        ).shape[:2]
        start = time.perf_counter()
        failure = None
        try:
            prediction = refiner.predict(rgb, raw_depth)
            restored = prediction.depth_m
            confidence = prediction.confidence
            backend_name = prediction.backend
        except (RuntimeError, ValueError) as error:
            restored = np.zeros(target_shape, dtype=np.float32)
            confidence = np.zeros(target_shape, dtype=np.float32)
            backend_name = f"{args.backend}:failed"
            failure = f"{type(error).__name__}: {error}"
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        if restored.shape != target_shape:
            target_size = (target_shape[1], target_shape[0])
            restored = cv2.resize(restored, target_size, interpolation=cv2.INTER_NEAREST)
            confidence = cv2.resize(confidence, target_size, interpolation=cv2.INTER_NEAREST)
        frame_id = safe_id(
            row.get("record_id")
            or row.get("frame_id")
            or f"{row.get('sequence_id', 'sequence')}_{index:06d}"
        )
        prediction_path = args.output_dir / f"{frame_id}_depth_m.npy"
        confidence_path = args.output_dir / f"{frame_id}_confidence.npy"
        np.save(prediction_path, restored, allow_pickle=False)
        np.save(confidence_path, confidence, allow_pickle=False)
        output_rows.append(
            {
                "target_depth_path": str(resolve(root, row["target_depth_path"])),
                "prediction_path": str(prediction_path),
                "mask_path": str(resolve(root, row["mask_path"])),
                "confidence_path": str(confidence_path),
                "scenario": (
                    row.get("bucket")
                    or row.get("scenario", "unspecified")
                ),
                "difficulty_tags": row.get("difficulty_tags", "ordinary"),
                "dataset": row.get("dataset", "unspecified"),
                "target_depth_scale_to_m": row.get("depth_scale_to_m", ""),
                "prediction_depth_scale_to_m": "1.0",
            }
        )
        if failure:
            failures.append({"frame_id": frame_id, "error": failure})
        print(
            f"{index + 1}/{len(rows)} {frame_id}: "
            f"{latencies_ms[-1]:.1f} ms backend={backend_name}"
        )

    evaluation_manifest = args.output_dir / "evaluation_manifest.csv"
    with evaluation_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    metrics = evaluate_depth_manifest(evaluation_manifest)
    summary = {
        "backend": args.backend,
        "frames": len(rows),
        "skipped_non_metric_rows": source_row_count - metric_row_count,
        "failed_frames": len(failures),
        "failures": failures,
        "model_load_seconds": model_load_seconds,
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "median_latency_ms": float(np.median(latencies_ms)),
        "metrics": metrics,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
