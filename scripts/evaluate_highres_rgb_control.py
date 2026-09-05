#!/usr/bin/env python3
"""True high-resolution RGB witness; network and raw depth remain at 320x180."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from liquid_depth.range_calibration import RangeNoiseCalibration
from liquid_depth.rgb_witness import RGBContourWitness
from liquid_depth.surface_video_runtime import SequencePredictor
from liquid_depth.verified_tracking import VerifiedSurfaceTracker


def corrupt_rgb(image, variant):
    image = image.copy()
    if variant == "rgb_dim":
        image = (image.astype(float) * 0.35).astype(np.uint8)
    elif variant == "rgb_occlusion":
        h, w = image.shape[:2]
        image[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 0
    elif variant == "rgb_shift":
        shift = 3 * image.shape[1] / 320
        image = cv2.warpAffine(
            image, np.float32([[1, 0, shift], [0, 1, 0]]), (image.shape[1], image.shape[0])
        )
    return image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--rgb-data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    report = json.loads(args.report.read_text())
    payload = json.loads(args.profile.read_text())
    if report["profile_sha256"] != hashlib.sha256(args.profile.read_bytes()).hexdigest():
        raise ValueError("Ablation profile changed")
    if report["checkpoint_sha256"] != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
        raise ValueError("Ablation weights changed")
    lookup = {
        (r["sequence"], r["index"]): Path(r["path"])
        for r in json.loads((args.rgb_data / "sequences.json").read_text())
    }
    predictor = SequencePredictor(args.checkpoint)
    previous = None
    for row in report["frames"]:
        key = row["sequence"], row["variant"]
        path = Path(row["path"])
        i, variant = row["index"], row["variant"]
        a = np.load(path / "frame.npz")
        raw = a["depth"].copy()
        truth_mask = a["truth_mask"].astype(bool)
        rgb = cv2.imread(str(path / "rgb.png"))
        pose = a["camera_to_world"].copy()
        pose[:3, :3] = pose[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        k = a["intrinsics"]
        reference_path = lookup[row["sequence"], i]
        if previous != key:
            if previous is not None:
                print(json.dumps({"finished": previous}), flush=True)
            previous = key
            witness = RGBContourWitness()
            calibration_error = None
            high_a = np.load(reference_path / "frame.npz")
            high_rgb = cv2.imread(str(reference_path / "rgb.png"))
            high_k = high_a["intrinsics"]
            try:
                witness.calibrate(
                    high_rgb,
                    high_a["truth_mask"],
                    row["truth_m"],
                    high_k,
                    pose,
                    row["bottom_world_m"],
                    row["radius_x_m"],
                    row["radius_y_m"],
                )
            except ValueError as exc:
                calibration_error = str(exc)
            tracker = VerifiedSurfaceTracker(
                strict_rgb=True,
                memory_options={"range_calibration": RangeNoiseCalibration(payload, row["sensor"])},
            )
            cues = {}
        if variant.startswith("drop") and i >= 5:
            yy, xx = np.nonzero(truth_mask)
            phase = (xx + i * 2) % raw.shape[1]
            chosen = phase <= np.quantile(phase, int(variant[4:]) / 100)
            raw[yy[chosen], xx[chosen]] = 0
        if variant == "slow_echo" and i >= 5:
            raw[truth_mask & (raw > 0)] += 0.003 * max(1.0, row["standoff_m"]) * (i - 4)
        if variant.startswith("recovery") and 8 <= i < 68:
            raw[:] = 0
        if variant == "recovery_bad_echo" and i >= 68:
            raw[truth_mask & (raw > 0)] += 0.06
        effective = variant if i >= 5 else "normal"
        rgb = corrupt_rgb(rgb, effective)
        # Frames reuse an exactly static rendered RGB state. Cache deterministic
        # witness results only in this evaluation; charge measured uncached cost.
        cue_key = str(reference_path), effective, pose.tobytes()
        if cue_key not in cues:
            high_rgb = corrupt_rgb(cv2.imread(str(reference_path / "rgb.png")), effective)
            started = perf_counter()
            cue = witness.estimate(high_rgb, high_k, pose, resolution_checks=True)
            cues[cue_key] = (cue, (perf_counter() - started) * 1000)
        cue, cue_ms = cues[cue_key]
        started = perf_counter()
        pred = predictor.predict(rgb, raw)
        out = tracker.process(rgb, raw, pred, k, pose, row["bottom_world_m"], witness=cue)
        out["latency_ms"] = (perf_counter() - started) * 1000 + cue_ms
        out["rgb_calibration_error"] = calibration_error
        row["strict_highres"] = out
    report["highres_rgb"] = {
        "width": 1280,
        "height": 720,
        "same_depth_and_network_inputs": True,
        "rgb_timing": "sum of model/fusion elapsed and measured uncached RGB witness time; repeated identical rendered RGB cues cached only by evaluator",
        "generation": "actual rerender, not interpolated low-resolution RGB; repeated depth files in RGB-only cache are never used as sensor frames",
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print("completed", flush=True)


if __name__ == "__main__":
    main()
