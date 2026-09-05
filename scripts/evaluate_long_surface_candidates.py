#!/usr/bin/env python3
"""Long-sequence diagnostic ablation: no candidate is automatically trusted."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from liquid_depth.range_calibration import RangeNoiseCalibration
from liquid_depth.surface_candidates import SurfaceCandidateEstimator, area_statistics
from liquid_depth.surface_memory import MetricSurfaceMemory
from liquid_depth.surface_video_runtime import SequencePredictor


def batch_predict(predictor, items):
    xs = []
    for rgb, z in items:
        color = cv2.cvtColor(cv2.resize(rgb, predictor.size), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        color = (color - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        depth = cv2.resize(z, predictor.size, interpolation=cv2.INTER_NEAREST)
        valid = np.isfinite(depth) & (depth > 0)
        encoded = np.where(
            valid,
            np.log(np.clip(depth, predictor.minimum, predictor.maximum) / predictor.minimum)
            / np.log(predictor.maximum / predictor.minimum),
            0,
        )
        xs.append(
            np.concatenate((color, encoded[..., None], valid[..., None]), axis=2)
            .astype(np.float32)
            .transpose(2, 0, 1)
        )
    with torch.inference_mode():
        out = predictor.model(torch.from_numpy(np.stack(xs)).to(predictor.device))
    masks = out["mask_logits"].sigmoid()[:, 0].cpu().numpy()
    conf = out["confidence"][:, 0].cpu().numpy()
    depth = out["depth_m"][:, 0].cpu().numpy()
    return [
        {
            "mask": cv2.resize(m, (items[i][1].shape[1], items[i][1].shape[0])) > 0.5,
            "confidence": cv2.resize(conf[i], (items[i][1].shape[1], items[i][1].shape[0])),
            "depth_m": cv2.resize(depth[i], (items[i][1].shape[1], items[i][1].shape[0])),
        }
        for i, m in enumerate(masks)
    ]


def summary(rows, method):
    q = [r for r in rows if r[method]["available"]]
    result = {"frames": len(rows), "candidates": len(q), "coverage": len(q) / max(len(rows), 1)}
    for name in ("mean_depth_m", "min_depth_m", "max_depth_m", "p05_depth_m", "p95_depth_m"):
        errors = np.array([r[method]["statistics"][name] - r["truth"][name] for r in q])
        targets = np.array([r["truth"][name] for r in q])
        result[name] = {
            "mae_mm": float(np.mean(abs(errors)) * 1000) if len(q) else None,
            "p95_mm": float(np.percentile(abs(errors), 95) * 1000) if len(q) else None,
            "abs_rel": float(np.mean(abs(errors) / np.maximum(abs(targets), 0.001))) if len(q) else None,
            "outside_tolerance": float(np.mean(abs(errors) > np.maximum(0.005, 0.02 * abs(targets))))
            if len(q)
            else None,
        }
    result["flags"] = dict(Counter(f for r in rows for f in r[method]["flags"]))
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
    predictor = SequencePredictor(args.checkpoint)
    profile = json.loads(args.profile.read_text())
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != profile["checkpoint_sha256"]:
        raise ValueError("Profile/model mismatch")
    groups = defaultdict(list)
    for row in json.loads((args.data / "sequences.json").read_text()):
        groups[row["sequence"]].append(row)
    rows_out = []
    cache = {}
    for sequence, frames in groups.items():
        policy = RangeNoiseCalibration(profile, frames[0]["sensor"])
        for variant in (
            ("normal", "drop75", "drop90", "slow_echo")
            if frames[0]["motion"] == "static"
            else ("normal", "drop75", "drop90")
        ):
            legacy = MetricSurfaceMemory(range_calibration=policy)
            estimators = {
                m: SurfaceCandidateEstimator(mode=m, range_calibration=policy)
                for m in ("free", "gravity", "early")
            }
            estimators["wave"] = SurfaceCandidateEstimator(surface_mode="waves", range_calibration=policy)
            for start in range(0, len(frames), 8):
                chunk = frames[start : start + 8]
                inputs = []
                meta = []
                for row in chunk:
                    if row["state"] not in cache:
                        state = Path(row["state"])
                        cache[row["state"]] = (
                            dict(np.load(state / "geometry.npz")),
                            cv2.imread(str(state / "rgb.png")),
                        )
                    a, rgb = cache[row["state"]]
                    raw = np.load(row["depth_path"])["depth"].copy()
                    mask = a["mask"].astype(bool)
                    i = row["index"]
                    if variant.startswith("drop") and i >= 20:
                        y, x = np.nonzero(mask)
                        phase = (x + 2 * i) % raw.shape[1]
                        clear = phase <= np.quantile(phase, int(variant[4:]) / 100)
                        raw[y[clear], x[clear]] = 0
                    if variant == "slow_echo" and i >= 20:
                        raw[mask & (raw > 0)] += 0.0003 * max(1.0, row["standoff_m"]) * (i - 19)
                    pose = a["camera_to_world"].copy()
                    pose[:3, :3] = pose[:3, :3] @ np.diag([1.0, -1.0, -1.0])
                    inputs.append((rgb, raw))
                    meta.append((row, a, pose))
                predictions = batch_predict(predictor, inputs)
                for (rgb, raw), (row, a, pose), pred in zip(inputs, meta, predictions, strict=True):
                    truth = area_statistics(a["area_truth_m"])
                    entry = {
                        k: row[k]
                        for k in ("sequence", "seed", "index", "sensor", "standoff_m", "truth_m", "motion")
                    }
                    entry.update(variant=variant, truth=truth)
                    out = legacy.estimate(rgb, raw, pred, a["intrinsics"], pose, row["bottom_world_m"])
                    entry["legacy"] = {
                        "available": out["accepted"],
                        "statistics": area_statistics([out["level_m"]]) if out["accepted"] else None,
                        "flags": out["reasons"],
                        "tilt_deg": out.get("tilt_deg"),
                    }
                    for mode, estimator in estimators.items():
                        out = estimator.estimate(
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
                            "statistics": out.get("statistics"),
                            "flags": out["quality_flags"],
                            "tilt_deg": out.get("tilt_deg"),
                            "history_used": out.get("early_history_used", False),
                            "local_level_m": out.get("local_plane_level_m"),
                        }
                    if row["index"] >= 20:
                        rows_out.append(entry)
            print(
                json.dumps({"sequence": sequence, "variant": variant, "scored": len(frames) - 20}), flush=True
            )
    methods = ("legacy", "free", "gravity", "early", "wave")
    buckets = defaultdict(list)
    for row in rows_out:
        buckets[f"{row['motion']}/{row['variant']}"].append(row)
        buckets[f"seed{row['seed']}/{row['motion']}/{row['variant']}"].append(row)
    report = {
        "schema_version": 1,
        "protocol": json.loads((args.data / "protocol.json").read_text()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "scope": "diagnostic candidates, NOT certified outputs; gravity/early static and quadratic wave surfaces",
        "scored_frames": len(rows_out),
        "summaries": {k: {m: summary(v, m) for m in methods} for k, v in buckets.items()},
        "frames": rows_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, allow_nan=False))
    print("complete", flush=True)


if __name__ == "__main__":
    main()
