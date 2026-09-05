#!/usr/bin/env python3
"""Evaluate full-range behavior, RGB-only echo checks and post-outage reacquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from liquid_depth.rgb_witness import RGBContourWitness
from liquid_depth.surface_memory import MetricSurfaceMemory, robust_plane, world_points
from liquid_depth.surface_video_runtime import SequencePredictor
from liquid_depth.verified_tracking import VerifiedSurfaceTracker


def metrics(rows, method):
    accepted = [r for r in rows if r[method]["accepted"]]
    error = np.array([r[method]["level_m"] - r["truth_m"] for r in accepted])
    truth = np.array([r["truth_m"] for r in accepted])
    bad = abs(error) > np.maximum(0.005, 0.02 * truth)
    return {
        "frames": len(rows),
        "outputs": len(accepted),
        "coverage": len(accepted) / max(len(rows), 1),
        "mae_mm": float(np.mean(abs(error)) * 1000) if len(error) else None,
        "p95_mm": float(np.percentile(abs(error), 95) * 1000) if len(error) else None,
        "abs_rel": float(np.mean(abs(error) / truth)) if len(error) else None,
        "pass_rate": float(np.mean(~bad)) if len(error) else None,
        "false_accept_given_output": float(np.mean(bad)) if len(error) else None,
        "false_accept_all_frames": float(bad.sum() / max(len(rows), 1)),
        "latency_p95_ms": float(np.percentile([r[method]["latency_ms"] for r in rows], 95)) if rows else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    predictor = SequencePredictor(args.checkpoint)
    rows = json.loads((args.data / "sequences.json").read_text())
    groups = defaultdict(list)
    for row in rows:
        groups[row["sequence"]].append(row)
    results = []
    for name, frames in groups.items():
        variants = (
            ["recovery", "recovery_bad_echo"]
            if "recovery" in name
            else ["normal", "drop75", "drop90", "slow_echo"]
        )
        for variant in variants:
            memory = MetricSurfaceMemory()
            verified = VerifiedSurfaceTracker()
            witness = RGBContourWitness()
            calibration_error = None
            for row in frames:
                i = row["index"]
                arrays = np.load(Path(row["path"]) / "frame.npz")
                rgb = cv2.imread(str(Path(row["path"]) / "rgb.png"))
                raw = arrays["depth"].copy()
                truth_mask = arrays["truth_mask"].astype(bool)
                k = arrays["intrinsics"]
                pose = arrays["camera_to_world"].copy()
                pose[:3, :3] = pose[:3, :3] @ np.diag([1.0, -1.0, -1.0])
                if i == 0:
                    try:
                        # One annotated initialization image is allowed;
                        # no evaluation-frame truth enters the witness.
                        witness.calibrate(
                            rgb,
                            truth_mask,
                            row["truth_m"],
                            k,
                            pose,
                            row["bottom_world_m"],
                            row["radius_x_m"],
                            row["radius_y_m"],
                        )
                    except ValueError as exc:
                        calibration_error = str(exc)
                if variant.startswith("drop") and i >= 5:
                    yy, xx = np.nonzero(truth_mask)
                    phase = (xx + i * 2) % raw.shape[1]
                    cutoff = np.quantile(phase, int(variant[4:]) / 100.0)
                    raw[yy[phase <= cutoff], xx[phase <= cutoff]] = 0.0
                if variant == "slow_echo" and i >= 5:
                    raw[truth_mask & (raw > 0)] += 0.003 * max(1.0, row["standoff_m"]) * (i - 4)
                if variant.startswith("recovery") and 8 <= i < 68:
                    raw[:] = 0.0
                if variant == "recovery_bad_echo" and i >= 68:
                    raw[truth_mask & (raw > 0)] += 0.06
                start = perf_counter()
                prediction = predictor.predict(rgb, raw)
                model_ms = (perf_counter() - start) * 1000
                selected = prediction["mask"] & (prediction["confidence"] >= 0.3)
                yy, xx = np.nonzero(selected)
                plane = (
                    robust_plane(
                        world_points(prediction["depth_m"][selected], np.column_stack((xx, yy)), k, pose)
                    )
                    if len(xx) >= 64
                    else None
                )
                network = {
                    "accepted": plane is not None,
                    "level_m": plane["level_world_m"] - row["bottom_world_m"] if plane else None,
                    "latency_ms": (perf_counter() - start) * 1000,
                }
                start = perf_counter()
                baseline = memory.estimate(rgb, raw, prediction, k, pose, row["bottom_world_m"])
                baseline["latency_ms"] = model_ms + (perf_counter() - start) * 1000
                start = perf_counter()
                independent = witness.estimate(rgb, k, pose)
                result = verified.process(
                    rgb, raw, prediction, k, pose, row["bottom_world_m"], witness=independent
                )
                result["latency_ms"] = model_ms + (perf_counter() - start) * 1000
                phase = (
                    "calibration"
                    if i < 5
                    else "outage"
                    if variant.startswith("recovery") and 8 <= i < 68
                    else "reacquisition"
                    if variant.startswith("recovery") and i >= 68
                    else "evaluation"
                )
                results.append(
                    {
                        **row,
                        "variant": variant,
                        "phase": phase,
                        "raw_valid_ratio": float(((raw > 0) & truth_mask).sum() / max(truth_mask.sum(), 1)),
                        "rgb_calibration_error": calibration_error,
                        "network": network,
                        "baseline": baseline,
                        "verified": result,
                    }
                )
            print(
                json.dumps(
                    {
                        "sequence": name,
                        "variant": variant,
                        "outputs": sum(r["verified"]["accepted"] for r in results[-len(frames) :]),
                    }
                ),
                flush=True,
            )
    summaries = {}
    buckets = defaultdict(list)
    for row in results:
        if row["phase"] == "calibration":
            continue
        key = f"{'heldout' if row['seed'] == 10709 else 'development'}/{row['motion']}/d{row['standoff_m']:g}_h{row['truth_m']:g}/{row['sensor']}/{row['variant']}/{row['phase']}"
        buckets[key].append(row)
    for key, bucket in buckets.items():
        summaries[key] = {method: metrics(bucket, method) for method in ("network", "baseline", "verified")}
    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "rendered_frames": len(rows),
        "evaluated_frames": len(results),
        "severe_failure_policy": "reject_only_no_recovery_optimization",
        "calibration": "first RGB frame: annotated surface ROI and known metric level; known ellipse CAD and pose",
        "sensor_scope": "existing simulator proxy, not a claim of sensor qualification at 10m",
        "summaries": summaries,
        "frames": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
