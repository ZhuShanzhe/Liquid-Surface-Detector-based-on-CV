#!/usr/bin/env python3
"""Batch-one opt-in runtime latency, with external IO excluded."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from liquid_depth.rgb_continuous import RGBContinuousWitness
from liquid_depth.rgb_witness import RGBContourWitness
from liquid_depth.surface_refinement import StereoNoiseModel
from liquid_depth.surface_video_runtime import UniversalSurfaceVideoSystem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    system = UniversalSurfaceVideoSystem(args.checkpoint)
    rows = json.loads((args.root / "long_surface_v5/sequences.json").read_text())
    result = {
        "gpu": torch.cuda.get_device_name(),
        "batch": 1,
        "measured_frames": 20,
        "scope": "warm same-frame sequential probe; no camera/network/UI/pose-estimation latency",
    }
    for mode in ("balanced", "sensor", "partition"):
        row = next(
            r
            for r in rows
            if r["standoff_m"] == 1
            and r["index"] == 25
            and r["sensor"] == "active_stereo"
            and r["motion"] == ("waves" if mode == "partition" else "static")
        )
        a = np.load(Path(row["state"]) / "geometry.npz")
        rgb = cv2.imread(str(Path(row["state"]) / "rgb.png"))
        raw = np.load(row["depth_path"])["depth"]
        pose = a["camera_to_world"].copy()
        pose[:3, :3] = pose[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        times = []
        for i in range(25):
            out = system.process_refined_surface(
                rgb,
                raw,
                a["intrinsics"],
                pose,
                row["bottom_world_m"],
                area_xy=a["area_xy"],
                radii=(row["radius_x_m"], row["radius_y_m"]),
                mode=mode,
                stereo_noise=StereoNoiseModel.simulation_proxy("active_stereo"),
                max_surface_slope=2.0,
            )
            if i >= 5:
                times.append(out["total_ms"])
        result[mode] = {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}
    row = json.loads((args.root / "sr_level_v5/sequences.json").read_text())[0]
    a = np.load(Path(row["path"]) / "frame.npz")
    rgb = cv2.imread(str(Path(row["path"]) / "rgb.png"))
    pose = a["camera_to_world"].copy()
    pose[:3, :3] = pose[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    for name, cls in (("rgb_grid", RGBContourWitness), ("rgb_continuous", RGBContinuousWitness)):
        w = cls()
        w.calibrate(
            rgb,
            a["truth_mask"],
            row["known_level_m"],
            a["intrinsics"],
            pose,
            row["bottom_world_m"],
            row["radius_x_m"],
            row["radius_y_m"],
        )
        times = []
        for i in range(25):
            start = perf_counter()
            w.estimate(rgb, a["intrinsics"], pose, resolution_checks=True)
            if i >= 5:
                times.append((perf_counter() - start) * 1000)
        result[name] = {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
