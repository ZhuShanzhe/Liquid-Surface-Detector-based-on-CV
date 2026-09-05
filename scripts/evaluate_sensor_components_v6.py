#!/usr/bin/env python3
"""Independent fixed-vessel sensor-only ablation; oracle mask deliberately isolates sensor bias."""

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from liquid_depth.simulation import (
    camera_intrinsics,
    camera_to_world,
    render_geometric_labels,
    sample_scene,
    simulate_raw_depth,
)
from liquid_depth.surface_refinement import RefinedSurfaceEstimator, StereoNoiseModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cv2.setNumThreads(1)
    results = []
    direction = np.array([0.1, -0.35, 1.1])
    direction /= np.linalg.norm(direction)
    variants = {
        "all": {},
        **{
            "without_" + key: {key: False}
            for key in ("depth_noise", "disparity_noise", "quantization", "range_cutoff")
        },
    }
    for seed in (11647, 11749):
        for d in (1.0, 3.0, 6.0, 10.0):
            for radius in (0.3, 0.6):
                for h in (0.1, 0.3):
                    base = sample_scene(
                        3,
                        seed=seed,
                        width=320,
                        height=180,
                        min_distance_m=1,
                        max_distance_m=1.1,
                        camera_profile="industrial_top",
                    )
                    base = replace(
                        base,
                        scenario="ordinary",
                        surface_radius_x_m=radius,
                        surface_radius_y_m=radius * 0.8,
                        container_bottom_z_m=-h - 0.005,
                        container_rim_z_m=0.12,
                        wall_thickness_m=0.005,
                        camera_position_m=tuple(direction * d),
                        camera_target_m=(0.0, 0.0, 0.0),
                        tilt_x=0.0,
                        tilt_y=0.0,
                        wave_amplitude_m=0.0,
                        floating_object_count=0,
                        corruption_severity=0.1,
                        liquid_turbidity=0.7,
                        container_taper_ratio=1.0,
                    )
                    labels = render_geometric_labels(base)
                    k, pose = camera_intrinsics(base), camera_to_world(base)
                    pose[:3, :3] = pose[:3, :3] @ np.diag([1.0, -1.0, -1.0])
                    pred = {"mask": labels["mask"].astype(bool)}
                    rgb = np.zeros((180, 320, 3), np.uint8)
                    area = np.array([[0.0, 0.0]])
                    for sensor in ("active_stereo", "structured_light", "tof"):
                        model = StereoNoiseModel.simulation_proxy(sensor) if sensor != "tof" else None
                        engines = {
                            "balanced": RefinedSurfaceEstimator(mode="balanced"),
                            "sensor": RefinedSurfaceEstimator(mode="sensor", stereo_noise=model),
                        }
                        for i in range(args.frames):
                            scene = replace(base, sensor_family=sensor, index=i * 17)
                            for variant, flags in variants.items():
                                raw = simulate_raw_depth(scene, labels, components=flags)["raw_depth_m"]
                                row = {
                                    "seed": seed,
                                    "distance": d,
                                    "radius": radius,
                                    "truth": h,
                                    "sensor_family": sensor,
                                    "index": i,
                                    "variant": variant,
                                }
                                for method in ("balanced", "sensor") if variant == "all" else ("balanced",):
                                    out = engines[method].estimate(
                                        rgb, raw, pred, k, pose, -h, area, (radius, radius * 0.8)
                                    )
                                    row[method] = out["level_m"] - h if out["candidate_available"] else None
                                results.append(row)
                        print(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "distance": d,
                                    "radius": radius,
                                    "depth": h,
                                    "sensor": sensor,
                                    "frames": args.frames,
                                }
                            ),
                            flush=True,
                        )
    buckets = defaultdict(list)
    for r in results:
        for key in (
            f"{r['sensor_family']}/{r['variant']}",
            f"d{r['distance']:g}/{r['sensor_family']}/{r['variant']}",
        ):
            buckets[key].append(r)
    summaries = {}
    for key, rows in buckets.items():
        summaries[key] = {}
        for method in ("balanced", "sensor") if rows[0]["variant"] == "all" else ("balanced",):
            valid = [r for r in rows if r[method] is not None]
            errors = np.array([r[method] for r in valid])
            summaries[key][method] = {
                "frames": len(rows),
                "available": len(valid),
                "bias_mm": float(errors.mean() * 1000) if len(valid) else None,
                "mae_mm": float(abs(errors).mean() * 1000) if len(valid) else None,
                "p95_mm": float(np.percentile(abs(errors), 95) * 1000) if len(valid) else None,
                "pass_rate": float(np.mean([abs(r[method]) <= max(0.005, 0.02 * r["truth"]) for r in valid]))
                if valid
                else None,
            }
    args.output.write_text(
        json.dumps(
            {
                "scope": "New seeds 11647/11749; true surface mask, known pose/bottom; sensor-only, not end-to-end",
                "frames_per_combination": args.frames,
                "summaries": summaries,
                "rows": results,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
