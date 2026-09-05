#!/usr/bin/env python3
"""SR evaluated by metric liquid-level error, not visual sharpness alone."""

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
from liquid_depth.super_resolution import SwinIRX4, scaled_intrinsics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    sr = SwinIRX4(args.repository, args.checkpoint)
    sr.upscale(np.zeros((180, 320, 3), np.uint8))
    groups = defaultdict(list)
    for r in json.loads((args.data / "sequences.json").read_text()):
        groups[r["sequence"]].append(r)
    rows = []
    for name, frames in groups.items():
        witnesses = {}
        for row in frames:
            a = np.load(Path(row["path"]) / "frame.npz")
            high = cv2.imread(str(Path(row["path"]) / "rgb.png"))
            low = cv2.resize(high, (320, 180), interpolation=cv2.INTER_AREA)
            k_high = a["intrinsics"]
            k_low = scaled_intrinsics(k_high, 0.25)
            pose = a["camera_to_world"].copy()
            pose[:3, :3] = pose[:3, :3] @ np.diag([1.0, -1.0, -1.0])
            for degradation in ("clean",) if row["index"] == 0 else ("clean", "blur_jpeg"):
                image = low.copy()
                if degradation == "blur_jpeg":
                    image = cv2.GaussianBlur(image, (5, 5), 0.8)
                    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        raise RuntimeError("JPEG encoding failed")
                    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                srout = sr.upscale(image)
                bicubic = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_CUBIC)
                for method in ("low", "bicubic", "swinir", "swinir_guarded", "native"):
                    target = (
                        image
                        if method == "low"
                        else bicubic
                        if method == "bicubic"
                        else high
                        if method == "native"
                        else srout["rgb_bgr"]
                    )
                    k = k_low if method == "low" else k_high
                    source_scale = 4 if method == "swinir_guarded" else 1
                    if row["index"] == 0:
                        witness = RGBContourWitness()
                        mask = (
                            a["truth_mask"]
                            if method == "native"
                            else cv2.resize(a["truth_mask"], (320, 180), interpolation=cv2.INTER_NEAREST)
                        )
                        if method not in ("low", "native"):
                            mask = cv2.resize(mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
                        try:
                            witness.calibrate(
                                target,
                                mask,
                                row["known_level_m"],
                                k,
                                pose,
                                row["bottom_world_m"],
                                row["radius_x_m"],
                                row["radius_y_m"],
                            )
                        except ValueError:
                            pass
                        witnesses[method] = witness
                        continue
                    started = perf_counter()
                    cue = witnesses[method].estimate(
                        target, k, pose, resolution_checks=True, source_pixel_scale=source_scale
                    )
                    elapsed = (perf_counter() - started) * 1000 + (
                        srout["latency_ms"] if method.startswith("swinir") else 0
                    )
                    bound = cue.get("error_bound_proxy_m")
                    available = bool(cue.get("available"))
                    error = cue["level_m"] - row["truth_m"] if available else None
                    tolerance = max(0.005, 0.02 * row["truth_m"])
                    eligible = bool(
                        available
                        and bound is not None
                        and bound < max(0.005, 0.02 * max(0, cue["level_m"] - bound))
                    )
                    mse = (
                        float(np.mean((target.astype(float) - high.astype(float)) ** 2))
                        if method != "low"
                        else None
                    )
                    rows.append(
                        {
                            "sequence": name,
                            "index": row["index"],
                            "distance": row["standoff_m"],
                            "truth_m": row["truth_m"],
                            "method": method,
                            "degradation": degradation,
                            "available": available,
                            "error_m": error,
                            "bound_m": bound,
                            "resolution_eligible": eligible,
                            "outside_tolerance": bool(abs(error) > tolerance) if available else None,
                            "outside_claimed_bound": bool(abs(error) > bound)
                            if available and bound is not None
                            else None,
                            "latency_ms": elapsed,
                            "psnr": 10 * np.log10(255**2 / mse) if mse and mse > 0 else None,
                        }
                    )
        print(json.dumps({"sequence": name, "done": True}), flush=True)
    summaries = {}
    for d in sorted({r["distance"] for r in rows}):
        for degradation in ("clean", "blur_jpeg"):
            for method in ("low", "bicubic", "swinir", "swinir_guarded", "native"):
                q = [
                    r
                    for r in rows
                    if r["distance"] == d and r["method"] == method and r["degradation"] == degradation
                ]
                valid = [r for r in q if r["available"]]
                eligible = [r for r in q if r["resolution_eligible"]]
                summaries[f"d{d:g}/{degradation}/{method}"] = {
                    "frames": len(q),
                    "available": len(valid),
                    "mae_mm": float(np.mean([abs(r["error_m"]) for r in valid]) * 1000) if valid else None,
                    "p95_mm": float(np.percentile([abs(r["error_m"]) for r in valid], 95) * 1000)
                    if valid
                    else None,
                    "resolution_eligible": len(eligible),
                    "eligible_bad": sum(r["outside_tolerance"] for r in eligible),
                    "bound_violations": sum(r["outside_claimed_bound"] for r in valid),
                    "latency_p95_ms": float(np.percentile([r["latency_ms"] for r in q], 95)),
                }
    report = {
        "schema_version": 1,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "scope": "RGB witness only; classical pretrained SwinIR-S x4; native clean HR is an upper reference, not affected by LR blur/JPEG",
        "summaries": summaries,
        "frames": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
