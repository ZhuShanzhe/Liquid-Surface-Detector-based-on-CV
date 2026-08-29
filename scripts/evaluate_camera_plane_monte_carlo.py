#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from liquid_depth.camera_qualification import (
    DEFAULT_QUALIFICATION_DISTANCES_M,
    qualify_plane_captures,
    simulate_plane_captures,
)
from liquid_depth.site_calibration_simulation import CAMERA_ERROR_PROFILES


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "mean": float(np.mean(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo five-distance qualification across virtual RGB-D units"
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--frames-per-distance", type=int, default=60)
    parser.add_argument("--calibration-frames", type=int, default=30)
    parser.add_argument("--validation-window-frames", type=int, default=5)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")

    profiles = {}
    for profile_index, (profile_name, profile) in enumerate(CAMERA_ERROR_PROFILES.items()):
        trials = []
        for trial in range(args.trials):
            seed = args.seed + profile_index * 100_000 + trial
            captures = simulate_plane_captures(
                profile,
                DEFAULT_QUALIFICATION_DISTANCES_M,
                frames_per_distance=args.frames_per_distance,
                seed=seed,
                shape=(args.height, args.width),
            )
            result = qualify_plane_captures(
                captures,
                depth_scale_to_m=1.0,
                calibration_frames_per_distance=args.calibration_frames,
                validation_window_frames=args.validation_window_frames,
                profile=profile,
            )
            trials.append(
                {
                    "seed": seed,
                    "raw": result["raw_all"],
                    "corrected_per_frame": result["corrected_in_profile_range"],
                    "corrected_temporal": result["corrected_temporal_in_profile_range"],
                    "calibration": result["calibration"],
                    "by_distance": {
                        distance: value["corrected_temporal"]
                        for distance, value in result["by_distance"].items()
                        if value["in_profile_range"]
                    },
                }
            )

        aggregate = {}
        for metric_group in ("raw", "corrected_per_frame", "corrected_temporal"):
            aggregate[metric_group] = {
                metric: _distribution([float(item[metric_group][metric]) for item in trials])
                for metric in ("mae_m", "abs_rel", "within_tolerance_rate")
            }
        aggregate["site_abs_rel_at_most_1pct_rate"] = float(
            np.mean([item["corrected_temporal"]["abs_rel"] <= 0.01 for item in trials])
        )
        aggregate["site_pass_rate_at_least_90pct_rate"] = float(
            np.mean([item["corrected_temporal"]["within_tolerance_rate"] >= 0.90 for item in trials])
        )
        profiles[profile_name] = {
            "profile": profile.to_dict(),
            "aggregate": aggregate,
            "trials": trials,
        }

    report = {
        "method": (
            "Each trial samples one fixed camera-unit scale/offset/nonlinearity, "
            "60 independent diffuse-plane frames at 0.3/1/3/5/8 m, fits the first "
            "30 frames, and evaluates the untouched last 30. Five-frame temporal "
            "medians represent at most 0.17 s at 30 FPS. Out-of-profile distances "
            "are reported by single-unit reports but excluded from profile aggregates."
        ),
        "trials_per_profile": args.trials,
        "distances_m": list(DEFAULT_QUALIFICATION_DISTANCES_M),
        "frames_per_distance": args.frames_per_distance,
        "calibration_frames_per_distance": args.calibration_frames,
        "validation_window_frames": args.validation_window_frames,
        "simulation_shape": [args.height, args.width],
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for profile_name, payload in profiles.items():
        temporal = payload["aggregate"]["corrected_temporal"]
        print(
            profile_name,
            f"median_mae_mm={temporal['mae_m']['median'] * 1000:.3f}",
            f"median_abs_rel_pct={temporal['abs_rel']['median'] * 100:.3f}",
            f"median_pass_pct={temporal['within_tolerance_rate']['median'] * 100:.1f}",
            f"sites_absrel_le_1pct={payload['aggregate']['site_abs_rel_at_most_1pct_rate'] * 100:.1f}%",
        )


if __name__ == "__main__":
    main()
