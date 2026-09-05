#!/usr/bin/env python3
"""End-to-end rendered RGB-D -> learned mask/depth -> metric plane benchmark."""

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

from liquid_depth.surface_memory import MetricSurfaceMemory, robust_plane, world_points
from liquid_depth.surface_video_runtime import SequencePredictor


def summarize(records):
    result = {}
    for method in ("network", "fresh_geometry", "guarded_fresh", "surface_memory"):
        values = [r[method] for r in records]
        accepted = [(r, v) for r, v in zip(records, values) if v["accepted"]]
        errors = np.array([v["level_m"] - r["truth_m"] for r, v in accepted])
        truth = np.array([r["truth_m"] for r, v in accepted])
        tolerance = np.maximum(0.005, 0.02 * truth)
        bad = np.abs(errors) > tolerance
        jitter = []
        run = longest = 0
        previous = None
        for row, value in zip(records, values):
            good = value["accepted"] and abs(value["level_m"] - row["truth_m"]) <= max(
                0.005, 0.02 * row["truth_m"]
            )
            same = (
                previous is not None
                and row["sequence"] == previous[0]["sequence"]
                and row["variant"] == previous[0]["variant"]
                and row["index"] == previous[0]["index"] + 1
            )
            bad_frame = value["accepted"] and not good
            run = (run + 1 if same else 1) if bad_frame else 0
            longest = max(longest, run)
            if same and value["accepted"] and previous[1]["accepted"]:
                jitter.append(
                    (value["level_m"] - row["truth_m"]) - (previous[1]["level_m"] - previous[0]["truth_m"])
                )
            previous = (row, value)
        result[method] = {
            "error_step_rms_mm": float(np.sqrt(np.mean(np.square(jitter))) * 1000) if jitter else None,
            "max_consecutive_bad_frames": longest,
            "frames": len(records),
            "accepted_frames": len(accepted),
            "coverage": len(accepted) / len(records),
            "evaluable_output_rate": 1.0 if accepted else None,
            "mae_mm": float(np.abs(errors).mean() * 1000) if len(errors) else None,
            "p95_mm": float(np.percentile(np.abs(errors), 95) * 1000) if len(errors) else None,
            "bias_mm": float(errors.mean() * 1000) if len(errors) else None,
            "abs_rel": float((np.abs(errors) / truth).mean()) if len(errors) else None,
            "tolerance_pass_rate": float((~bad).mean()) if len(errors) else None,
            "false_accept_all_frames": float(bad.sum() / len(records)),
            "false_accept_given_output": float(bad.mean()) if len(errors) else None,
            "latency_p95_ms": float(np.percentile([v["latency_ms"] for v in values], 95)),
            "latency_max_ms": float(max(v["latency_ms"] for v in values)),
            "memory_activated_frames": sum(bool(v.get("memory_activated")) for v in values),
            "memory_active_p95_ms": float(
                np.percentile([v["latency_ms"] for v in values if v.get("memory_activated")], 95)
            )
            if any(v.get("memory_activated") for v in values)
            else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "normal",
            "drop75",
            "drop90",
            "drop98",
            "total",
            "cold_total",
            "long95",
            "wrong_echo",
            "pose_error",
        ],
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    predictor = SequencePredictor(args.checkpoint)
    rows = json.loads((args.data / "sequences.json").read_text())
    groups = defaultdict(list)
    for row in rows:
        groups[row["sequence"]].append(row)
    records = []
    for sequence, frames in groups.items():
        for variant in args.variants:
            memory = MetricSurfaceMemory()
            current = MetricSurfaceMemory()
            guarded = MetricSurfaceMemory()
            for row in frames:
                i = row["index"]
                data = np.load(Path(row["path"]) / "frame.npz")
                rgb = cv2.imread(str(Path(row["path"]) / "rgb.png"))
                raw = data["depth"].copy()
                mask = data["truth_mask"].astype(bool)
                failure = (i >= 8 and (i < 28 or variant == "long95")) or variant == "cold_total"
                if variant.startswith("drop") or variant in ("total", "cold_total", "long95"):
                    rate = (
                        1.0
                        if variant in ("total", "cold_total")
                        else 0.95
                        if variant == "long95"
                        else int(variant[4:]) / 100.0
                    )
                    if failure:
                        yy, xx = np.nonzero(mask)
                        # Contiguous missing area sweeps horizontally over time.
                        phase = (xx + int(i * 2)) % raw.shape[1]
                        cutoff = np.quantile(phase, rate)
                        raw[yy[phase <= cutoff], xx[phase <= cutoff]] = 0.0
                if variant == "wrong_echo" and failure:
                    raw[mask & (raw > 0)] += 0.08
                if variant == "slow_echo" and failure:
                    raw[mask & (raw > 0)] += 0.003 * (i - 7)
                transform = data["camera_to_world"].copy()
                transform[:3, :3] = transform[:3, :3] @ np.diag([1.0, -1.0, -1.0])
                if variant == "pose_error" and failure:
                    transform[2, 3] += 0.025
                valid_ratio = float(((raw > 0) & mask).sum() / max(mask.sum(), 1))
                started = perf_counter()
                prediction = predictor.predict(rgb, raw)
                prediction_ms = (perf_counter() - started) * 1000.0
                k = data["intrinsics"]
                selected = (
                    prediction["mask"] & (prediction["confidence"] >= 0.3) & (prediction["depth_m"] > 0)
                )
                yy, xx = np.nonzero(selected)
                pts = world_points(prediction["depth_m"][selected], np.column_stack((xx, yy)), k, transform)
                plane = robust_plane(pts) if len(pts) >= 64 else None
                network = {
                    "accepted": plane is not None,
                    "level_m": None,
                    "latency_ms": (perf_counter() - started) * 1000,
                }
                if plane:
                    network["level_m"] = plane["level_world_m"] - row["bottom_world_m"]
                measurements = {}
                for label, estimator, enabled in [
                    ("fresh_geometry", current, False),
                    ("guarded_fresh", guarded, False),
                    ("surface_memory", memory, True),
                ]:
                    stamp = perf_counter()
                    result = estimator.estimate(
                        rgb,
                        raw,
                        prediction,
                        k,
                        transform,
                        row["bottom_world_m"],
                        use_memory=enabled,
                        guard_jumps=label != "fresh_geometry",
                    )
                    result["latency_ms"] = (perf_counter() - stamp) * 1000 + prediction_ms
                    measurements[label] = result
                record = {
                    **row,
                    "variant": variant,
                    "phase": "failure" if failure else "warmup" if i < 8 else "recovery",
                    "raw_surface_valid_ratio": valid_ratio,
                    "network": network,
                    **measurements,
                }
                records.append(record)
            print(
                json.dumps(
                    {"sequence": sequence, "variant": variant, "summary": summarize(records[-len(frames) :])}
                ),
                flush=True,
            )
    buckets = defaultdict(list)
    for row in records:
        split = "heldout" if row["seed"] == 9323 else "development"
        buckets[f"{split}/{row['motion']}/{row['variant']}/{row['phase']}"].append(row)
    recovery = []
    for sequence in groups:
        for variant in args.variants:
            subset = [
                r
                for r in records
                if r["sequence"] == sequence and r["variant"] == variant and r["phase"] == "recovery"
            ]
            if not subset:
                continue
            for method in ("network", "fresh_geometry", "guarded_fresh", "surface_memory"):
                delay = None
                for index in range(len(subset) - 2):
                    window = subset[index : index + 3]
                    if all(
                        r[method]["accepted"]
                        and abs(r[method]["level_m"] - r["truth_m"]) <= max(0.005, 0.02 * r["truth_m"])
                        for r in window
                    ):
                        delay = (window[-1]["index"] - subset[0]["index"]) / 10.0
                        break
                recovery.append(
                    {
                        "sequence": sequence,
                        "variant": variant,
                        "method": method,
                        "recovery_confirmation_s": delay,
                        "criterion": "3 consecutive accepted in-tolerance frames",
                    }
                )
    report = {
        "schema_version": 2,
        "recovery": recovery,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "scope": "rendered_rgbd_learned_perception_metric_geometry",
        "assumptions": [
            "known calibrated intrinsics, gravity and bottom",
            "supplied simulated pose, including a separately injected pose-error stress",
            "Eevee RGB rendering with analytic sensor corruption; no USB hardware",
            "truth masks only create/score sensor corruption, never enter estimators",
            "no model weight training or per-test threshold tuning",
        ],
        "data_frames": len(rows),
        "evaluated_frames": len(records),
        "aggregate": summarize(records),
        "by_scenario": {name: summarize(values) for name, values in buckets.items()},
        "frames": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report["aggregate"]), flush=True)


if __name__ == "__main__":
    main()
