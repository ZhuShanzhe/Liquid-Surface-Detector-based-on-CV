#!/usr/bin/env python3
"""Frozen-profile ablation on seed 10709; no test-time calibration fitting."""

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
from evaluate_range_reacquisition import metrics

from liquid_depth.range_calibration import RangeNoiseCalibration
from liquid_depth.rgb_witness import RGBContourWitness
from liquid_depth.surface_memory import MetricSurfaceMemory
from liquid_depth.surface_video_runtime import SequencePredictor
from liquid_depth.verified_tracking import VerifiedSurfaceTracker

METHODS = ("v3", "noise", "calibrated", "strict", "geometry")


def calibration_metrics(items):
    if not items:
        return {}
    values = np.concatenate(items)
    result = {"sampled_pixels": len(values)}
    for j, name in enumerate(("original_score", "range_calibrated")):
        prob, target = values[:, j], values[:, 2]
        bins = np.minimum((prob * 10).astype(int), 9)
        ece = sum(
            np.mean(bins == i) * abs(np.mean(prob[bins == i]) - np.mean(target[bins == i]))
            for i in range(10)
            if np.any(bins == i)
        )
        result[name] = {"brier": float(np.mean((prob - target) ** 2)), "ece": float(ece)}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    payload = json.loads(args.profile.read_text())
    assert payload["development_seed"] == 10607
    predictor = SequencePredictor(args.checkpoint)
    groups = defaultdict(list)
    for row in json.loads((args.data / "sequences.json").read_text()):
        if row["seed"] == 10709:
            groups[row["sequence"]].append(row)
    results, confidence = [], defaultdict(list)
    for name, frames in groups.items():
        variants = (
            ["recovery", "recovery_bad_echo"]
            if "recovery" in name
            else ["normal", "drop75", "drop90", "slow_echo"]
        )
        if "recovery" not in name and frames[0]["sensor"] == "tof":
            variants += ["rgb_dim", "rgb_occlusion", "rgb_shift"]
        sensor = frames[0]["sensor"]
        noise = RangeNoiseCalibration(payload, sensor, calibrate_confidence=False)
        calibrated = RangeNoiseCalibration(payload, sensor)
        for variant in variants:
            trackers = {
                "v3": VerifiedSurfaceTracker(),
                "noise": VerifiedSurfaceTracker(memory_options={"range_calibration": noise}),
                "calibrated": VerifiedSurfaceTracker(memory_options={"range_calibration": calibrated}),
                "strict": VerifiedSurfaceTracker(
                    memory_options={"range_calibration": calibrated}, strict_rgb=True
                ),
                "geometry": MetricSurfaceMemory(range_calibration=calibrated),
            }
            witness = RGBContourWitness()
            calibration_error = None
            for row in frames:
                i = row["index"]
                path = Path(row["path"])
                a = np.load(path / "frame.npz")
                rgb = cv2.imread(str(path / "rgb.png"))
                raw, truth_mask = a["depth"].copy(), a["truth_mask"].astype(bool)
                k, pose = a["intrinsics"], a["camera_to_world"].copy()
                pose[:3, :3] = pose[:3, :3] @ np.diag([1.0, -1.0, -1.0])
                if i == 0:
                    try:
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
                    raw[
                        yy[phase <= np.quantile(phase, int(variant[4:]) / 100)],
                        xx[phase <= np.quantile(phase, int(variant[4:]) / 100)],
                    ] = 0
                if variant == "slow_echo" and i >= 5:
                    raw[truth_mask & (raw > 0)] += 0.003 * max(1.0, row["standoff_m"]) * (i - 4)
                if variant.startswith("recovery") and 8 <= i < 68:
                    raw[:] = 0
                if variant == "recovery_bad_echo" and i >= 68:
                    raw[truth_mask & (raw > 0)] += 0.06
                if i >= 5:
                    if variant == "rgb_dim":
                        rgb = (rgb.astype(float) * 0.35).astype(np.uint8)
                    elif variant == "rgb_occlusion":
                        h, w = raw.shape
                        rgb[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 0
                    elif variant == "rgb_shift":
                        rgb = cv2.warpAffine(
                            rgb, np.float32([[1, 0, 3], [0, 1, 0]]), (rgb.shape[1], rgb.shape[0])
                        )
                start = perf_counter()
                pred = predictor.predict(rgb, raw)
                model_ms = (perf_counter() - start) * 1000
                start = perf_counter()
                cue = witness.estimate(rgb, k, pose)
                cue_ms = (perf_counter() - start) * 1000
                start = perf_counter()
                strict_cue = witness.estimate(rgb, k, pose, resolution_checks=True)
                strict_ms = (perf_counter() - start) * 1000
                outputs = {}
                for method, tracker in trackers.items():
                    start = perf_counter()
                    if method == "geometry":
                        out = tracker.estimate(rgb, raw, pred, k, pose, row["bottom_world_m"])
                    else:
                        out = tracker.process(
                            rgb,
                            raw,
                            pred,
                            k,
                            pose,
                            row["bottom_world_m"],
                            witness=strict_cue if method == "strict" else cue,
                        )
                    out["latency_ms"] = (
                        model_ms
                        + (perf_counter() - start) * 1000
                        + (0 if method == "geometry" else strict_ms if method == "strict" else cue_ms)
                    )
                    outputs[method] = out
                if variant == "normal" and i >= 5:
                    interior = cv2.erode(pred["mask"].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(
                        bool
                    )
                    support = interior & (raw > 0) & np.isfinite(raw)
                    if support.any():
                        distance = float(np.median(raw[support]))
                        prob, _, _ = calibrated.reliability(pred["confidence"], distance)
                        y, x = np.nonzero(support)
                        take = np.linspace(0, len(x) - 1, min(1024, len(x))).astype(int)
                        y, x = y[take], x[take]
                        target = truth_mask[y, x] & (
                            abs(raw[y, x] - a["truth_depth"][y, x]) <= 3 * calibrated.sigma(distance) + 0.001
                        )
                        confidence[f"{sensor}/d{row['standoff_m']:g}"].append(
                            np.column_stack((pred["confidence"][y, x], prob[y, x], target))
                        )
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
                        "rgb_calibration_error": calibration_error,
                        **outputs,
                    }
                )
            print(
                json.dumps(
                    {
                        "sequence": name,
                        "variant": variant,
                        "strict_outputs": sum(r["strict"]["accepted"] for r in results[-len(frames) :]),
                    }
                ),
                flush=True,
            )
    buckets = defaultdict(list)
    for row in results:
        if row["phase"] in ("calibration", "outage"):
            continue
        buckets[
            f"{row['sensor']}/d{row['standoff_m']:g}_h{row['truth_m']:g}/{row['motion']}/{row['variant']}"
        ].append(row)
    report = {
        "schema_version": 1,
        "development_seed": 10607,
        "evaluation_seed": 10709,
        "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "profile": str(args.profile),
        "evaluated_frames": len(results),
        "confidence_metrics": {key: calibration_metrics(v) for key, v in confidence.items()},
        "summaries": {key: {m: metrics(v, m) for m in METHODS} for key, v in buckets.items()},
        "frames": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
