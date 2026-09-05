#!/usr/bin/env python3
"""Serial batch-one latency probe, excluding camera/network IO and SR initialization."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from liquid_depth.super_resolution import SwinIRX4
from liquid_depth.surface_video_runtime import UniversalSurfaceVideoSystem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--sr-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    rows = json.loads((args.data / "sequences.json").read_text())
    row = next(r for r in rows if r["standoff_m"] == 1 and r["index"] == 0)
    frame = dict(np.load(Path(row["state"]) / "geometry.npz"))
    rgb = cv2.imread(str(Path(row["state"]) / "rgb.png"))
    raw = np.load(row["depth_path"])["depth"]
    pose = frame["camera_to_world"].copy()
    pose[:3, :3] = pose[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    system = UniversalSurfaceVideoSystem(
        args.checkpoint, range_profile=args.profile, sensor_family=row["sensor"]
    )
    samples = []
    for i in range(25):
        out = system.process_surface_candidates(
            rgb,
            raw,
            frame["intrinsics"],
            pose,
            row["bottom_world_m"],
            area_xy=frame["area_xy"],
            radii=(row["radius_x_m"], row["radius_y_m"]),
        )
        if i >= 5:
            samples.append(out["total_ms"])
    sr = SwinIRX4(args.repository, args.sr_checkpoint)
    sr_times = []
    for i in range(25):
        started = perf_counter()
        sr.upscale(rgb)
        if i >= 5:
            sr_times.append((perf_counter() - started) * 1000)
    result = {
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "batch": 1,
        "input_shape": list(rgb.shape),
        "frames": 20,
        "scope": "warm sequential same-frame probe; candidate includes model and geometric fit; SR excludes independent RGB verification",
    }
    for key, times in (("candidate", samples), ("sr_only", sr_times)):
        result[key] = {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
