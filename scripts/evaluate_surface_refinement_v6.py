#!/usr/bin/env python3
"""Paired v5 regression for sensor likelihood and observable local surfaces."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from evaluate_long_surface_candidates import batch_predict, summary

from liquid_depth.range_calibration import RangeNoiseCalibration
from liquid_depth.surface_refinement import RefinedSurfaceEstimator, StereoNoiseModel
from liquid_depth.surface_video_runtime import SequencePredictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    predictor = SequencePredictor(args.checkpoint)
    profile = json.loads(args.profile.read_text())
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != profile["checkpoint_sha256"]:
        raise ValueError("Profile/model mismatch")
    previous = json.loads(args.baseline.read_text())
    old = {(r["sequence"], r["variant"], r["index"]): r for r in previous["frames"]}
    groups = defaultdict(list)
    for row in json.loads((args.data / "sequences.json").read_text()):
        if row["index"] >= 20 and (row["index"] - 20) % args.stride == 0:
            groups[row["sequence"]].append(row)
    cache, results = {}, []
    for sequence, frames in groups.items():
        sensor = frames[0]["sensor"]
        policy = RangeNoiseCalibration(profile, sensor)
        noise = StereoNoiseModel.simulation_proxy(sensor) if sensor != "tof" else None
        variants = (
            ("normal", "drop75", "drop90", "slow_echo")
            if frames[0]["motion"] == "static"
            else ("normal", "drop75", "drop90")
        )
        for variant in variants:
            for start in range(0, len(frames), 8):
                chunk = frames[start : start + 8]
                inputs, metas = [], []
                for row in chunk:
                    if row["state"] not in cache:
                        state = Path(row["state"])
                        cache[row["state"]] = (
                            dict(np.load(state / "geometry.npz")),
                            cv2.imread(str(state / "rgb.png")),
                        )
                    a, rgb = cache[row["state"]]
                    raw = np.load(row["depth_path"])["depth"].copy()
                    mask, i = a["mask"].astype(bool), row["index"]
                    if variant.startswith("drop"):
                        y, x = np.nonzero(mask)
                        phase = (x + 2 * i) % raw.shape[1]
                        clear = phase <= np.quantile(phase, int(variant[4:]) / 100)
                        raw[y[clear], x[clear]] = 0
                    if variant == "slow_echo":
                        raw[mask & (raw > 0)] += 0.0003 * max(1.0, row["standoff_m"]) * (i - 19)
                    pose = a["camera_to_world"].copy()
                    pose[:3, :3] = pose[:3, :3] @ np.diag([1.0, -1.0, -1.0])
                    inputs.append((rgb, raw))
                    metas.append((row, a, pose))
                predictions = batch_predict(predictor, inputs)
                for (rgb, raw), (row, a, pose), pred in zip(inputs, metas, predictions, strict=True):
                    ref = old[(sequence, variant, row["index"])]
                    entry = {
                        k: ref[k]
                        for k in (
                            "sequence",
                            "seed",
                            "index",
                            "sensor",
                            "standoff_m",
                            "truth_m",
                            "motion",
                            "variant",
                            "truth",
                        )
                    }
                    entry["sensor_family"] = entry.pop("sensor")
                    for method in ("gravity", "early", "wave"):
                        entry[method] = ref[method]
                    valid = raw[pred["mask"] & (raw > 0)]
                    sigma = policy.sigma(float(np.median(valid))) if valid.size else 0.003
                    for mode in (
                        ("balanced", "sensor") if row["motion"] == "static" else ("balanced", "partition")
                    ):
                        engine = RefinedSurfaceEstimator(
                            mode=mode, stereo_noise=noise, sigma_m=max(0.003, sigma), max_surface_slope=2.0
                        )
                        out = engine.estimate(
                            rgb,
                            raw,
                            pred,
                            a["intrinsics"],
                            pose,
                            row["bottom_world_m"],
                            a["area_xy"],
                            (row["radius_x_m"], row["radius_y_m"]),
                        )
                        entry[mode] = {
                            "available": out["candidate_available"],
                            "statistics": out["statistics"],
                            "flags": out["quality_flags"],
                        }
                        if mode == "partition":
                            entry[mode].update(
                                intervals=out.get("statistics_intervals"),
                                observed_area_fraction=out.get("observed_area_fraction"),
                            )
                    results.append(entry)
            print(json.dumps({"sequence": sequence, "variant": variant, "frames": len(frames)}), flush=True)
    buckets = defaultdict(list)
    for r in results:
        for key in (
            f"{r['motion']}/{r['variant']}",
            f"{r['motion']}/{r['variant']}/d{r['standoff_m']:g}/h{r['truth_m']:g}",
            f"{r['motion']}/{r['variant']}/{r['sensor_family']}",
        ):
            buckets[key].append(r)
    summaries = {}
    for name, rows in buckets.items():
        methods = (
            ("gravity", "early", "balanced", "sensor")
            if rows[0]["motion"] == "static"
            else ("wave", "balanced", "partition")
        )
        summaries[name] = {m: summary(rows, m) for m in methods}
        if rows[0]["motion"] == "waves":
            q = [r for r in rows if r["partition"].get("intervals")]
            interval = {"available": len(q), "frames": len(rows)}
            for key in ("mean_depth_m", "min_depth_m", "max_depth_m"):
                interval[key] = {
                    "contains_truth": float(
                        np.mean(
                            [
                                r["partition"]["intervals"][key][0]
                                <= r["truth"][key]
                                <= r["partition"]["intervals"][key][1]
                                for r in q
                            ]
                        )
                    )
                    if q
                    else None,
                    "mean_width_mm": float(
                        np.mean([np.ptp(r["partition"]["intervals"][key]) for r in q]) * 1000
                    )
                    if q
                    else None,
                }
            interval["observed_area_mean"] = float(
                np.mean([r["partition"].get("observed_area_fraction") or 0 for r in rows])
            )
            summaries[name]["conditional_intervals"] = interval
    report = {
        "schema_version": 1,
        "scope": "v5 paired regression; no new network fit; simulation proxy noise not device calibration",
        "frames_count": len(results),
        "checkpoint_sha256": digest,
        "summaries": summaries,
        "frames": results,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
