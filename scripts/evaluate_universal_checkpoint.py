#!/usr/bin/env python3
# ruff: noqa: B023
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path


def empty_metrics() -> dict[str, float]:
    return {"absolute": 0.0, "squared": 0.0, "relative": 0.0, "within": 0.0, "pixels": 0.0}


def summarize(values: dict[str, float]) -> dict[str, float]:
    count = max(values["pixels"], 1.0)
    return {
        "pixels": int(values["pixels"]),
        "mae_m": values["absolute"] / count,
        "rmse_m": (values["squared"] / count) ** 0.5,
        "abs_rel": values["relative"] / count,
        "within_tolerance_rate": values["within"] / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a universal checkpoint by scenario and range")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from liquid_depth.models.universal import UniversalLiquidSurfaceNet
    from liquid_depth.training.universal_dataset import UniversalMultiTaskDataset

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = tuple(map(int, checkpoint["image_size"]))
    minimum = float(checkpoint.get("min_depth_m", 0.1))
    maximum = float(checkpoint.get("max_depth_m", 10.0))
    dataset = UniversalMultiTaskDataset(
        args.manifest,
        args.split,
        image_size,
        augment=False,
        min_depth_m=minimum,
        max_depth_m=maximum,
    )
    loader = DataLoader(dataset, args.batch_size, num_workers=args.workers, pin_memory=True)
    model = UniversalLiquidSurfaceNet(
        int(checkpoint["base_channels"]),
        minimum,
        maximum,
        rgb_prior_enabled=bool(checkpoint.get("rgb_prior_enabled", False)),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    global_metrics = empty_metrics()
    scenarios: defaultdict[str, dict[str, float]] = defaultdict(empty_metrics)
    ranges: defaultdict[str, dict[str, float]] = defaultdict(empty_metrics)
    confidence_thresholds = (0.25, 0.50, 0.75, 0.90)
    selective = {threshold: empty_metrics() for threshold in confidence_thresholds}
    calibration_squared = 0.0
    calibration_pixels = 0
    surface_level = {
        threshold: {
            "absolute": 0.0,
            "relative": 0.0,
            "within": 0.0,
            "samples": 0.0,
        }
        for threshold in confidence_thresholds
    }
    range_defs = (("0.1-0.3m", 0.1, 0.3), ("0.3-1m", 0.3, 1.0), ("1-3m", 1.0, 3.0), ("3-10m", 3.0, 10.0001))
    latencies: list[float] = []
    offset = 0
    with torch.inference_mode():
        for inputs, target in loader:
            inputs = inputs.to(device, non_blocking=True)
            target = {key: value.to(device, non_blocking=True) for key, value in target.items()}
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0 / inputs.shape[0])
            valid = target["valid"] > 0
            truth = target["depth_m"]
            absolute = (prediction["depth_m"] - truth).abs()
            tolerance = torch.maximum(truth * 0.01, torch.full_like(truth, 0.003))
            confidence = prediction["confidence"]
            reliable = (absolute <= tolerance).float()
            calibration_squared += float(((confidence - reliable) ** 2)[valid].sum())
            calibration_pixels += int(valid.sum())

            def accumulate(store, selected):
                count = int(selected.sum())
                if count == 0:
                    return
                error = absolute[selected]
                store["absolute"] += float(error.sum())
                store["squared"] += float((error * error).sum())
                store["relative"] += float((error / truth[selected].clamp_min(1e-6)).sum())
                store["within"] += int((error <= tolerance[selected]).sum())
                store["pixels"] += count

            accumulate(global_metrics, valid)
            for threshold in confidence_thresholds:
                accumulate(selective[threshold], valid & (confidence >= threshold))
            for name, low, high in range_defs:
                accumulate(ranges[name], valid & (truth >= low) & (truth < high))
            predicted_surface = prediction["mask_logits"].sigmoid() >= 0.5
            for batch_index in range(inputs.shape[0]):
                scenario = dataset.rows[offset + batch_index].get("scenario", "unknown")
                selected = valid[batch_index : batch_index + 1]
                batch_truth = truth[batch_index : batch_index + 1]
                batch_abs = absolute[batch_index : batch_index + 1]
                batch_tol = tolerance[batch_index : batch_index + 1]
                count = int(selected.sum())
                if count:
                    store = scenarios[scenario]
                    error = batch_abs[selected]
                    store["absolute"] += float(error.sum())
                    store["squared"] += float((error * error).sum())
                    store["relative"] += float((error / batch_truth[selected].clamp_min(1e-6)).sum())
                    store["within"] += int((error <= batch_tol[selected]).sum())
                    store["pixels"] += count
                valid_pixels = int(selected.sum())
                for threshold in confidence_thresholds:
                    level_selected = (
                        selected
                        & predicted_surface[batch_index : batch_index + 1]
                        & (confidence[batch_index : batch_index + 1] >= threshold)
                    )
                    minimum_points = max(64, int(0.01 * valid_pixels))
                    if int(level_selected.sum()) < minimum_points:
                        continue
                    signed_error = (prediction["depth_m"][batch_index : batch_index + 1] - batch_truth)[
                        level_selected
                    ]
                    level_error = float(signed_error.median().abs())
                    reference = float(batch_truth[level_selected].median())
                    level_tolerance = max(0.003, 0.01 * reference)
                    level_store = surface_level[threshold]
                    level_store["absolute"] += level_error
                    level_store["relative"] += level_error / max(reference, 1e-6)
                    level_store["within"] += float(level_error <= level_tolerance)
                    level_store["samples"] += 1.0
            offset += inputs.shape[0]
    report = {
        "checkpoint": args.checkpoint.resolve().as_posix(),
        "manifest": Path(args.manifest).resolve().as_posix(),
        "split": args.split,
        "samples": len(dataset),
        "global": summarize(global_metrics),
        "by_scenario": {name: summarize(values) for name, values in sorted(scenarios.items())},
        "by_range": {name: summarize(values) for name, values in ranges.items()},
        "selective_rejection": {
            f"confidence>={threshold:.2f}": {
                **summarize(values),
                "coverage": values["pixels"] / max(global_metrics["pixels"], 1.0),
            }
            for threshold, values in selective.items()
        },
        "confidence_brier_score": calibration_squared / max(calibration_pixels, 1),
        "surface_level_selective": {
            f"confidence>={threshold:.2f}": {
                "accepted_frames": int(values["samples"]),
                "frame_coverage": values["samples"] / max(len(dataset), 1),
                "mae_m": values["absolute"] / max(values["samples"], 1.0),
                "abs_rel": values["relative"] / max(values["samples"], 1.0),
                "within_tolerance_rate": values["within"] / max(values["samples"], 1.0),
            }
            for threshold, values in surface_level.items()
        },
        "latency_ms_per_frame": {
            "mean": sum(latencies) / max(len(latencies), 1),
            "max_batch_mean": max(latencies, default=0.0),
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
