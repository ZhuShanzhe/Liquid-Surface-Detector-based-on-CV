#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from liquid_depth.camera_qualification import (
    DEFAULT_QUALIFICATION_DISTANCES_M,
    load_plane_capture_directory,
    qualify_plane_captures,
    save_qualification_report,
    simulate_plane_captures,
)
from liquid_depth.site_calibration_simulation import CAMERA_ERROR_PROFILES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify RGB-D depth accuracy on diffuse planes at known distances"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--simulate", action="store_true")
    mode.add_argument("--capture-root", type=Path)
    parser.add_argument("--profile", choices=sorted(CAMERA_ERROR_PROFILES))
    parser.add_argument(
        "--distances-m",
        type=float,
        nargs="+",
        default=list(DEFAULT_QUALIFICATION_DISTANCES_M),
    )
    parser.add_argument("--frames-per-distance", type=int, default=60)
    parser.add_argument("--calibration-frames", type=int, default=30)
    parser.add_argument("--validation-window-frames", type=int, default=5)
    parser.add_argument("--depth-scale-to-m", type=float, default=0.001)
    parser.add_argument("--roi-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.simulate and args.profile is None:
        parser.error("--simulate requires --profile")
    profile = CAMERA_ERROR_PROFILES.get(args.profile) if args.profile else None
    if args.simulate:
        captures = simulate_plane_captures(
            profile,
            args.distances_m,
            frames_per_distance=args.frames_per_distance,
            seed=args.seed,
        )
        depth_scale_to_m = 1.0
        source = "synthetic_diffuse_plane"
    else:
        captures = load_plane_capture_directory(args.capture_root)
        depth_scale_to_m = args.depth_scale_to_m
        source = args.capture_root.expanduser().resolve().as_posix()

    report = qualify_plane_captures(
        captures,
        depth_scale_to_m=depth_scale_to_m,
        calibration_frames_per_distance=args.calibration_frames,
        roi_fraction=args.roi_fraction,
        validation_window_frames=args.validation_window_frames,
        profile=profile,
    )
    report["source"] = source
    report["camera_profile"] = profile.to_dict() if profile else None
    report["is_simulation"] = bool(args.simulate)
    temporal = report["corrected_temporal_in_profile_range"]
    gate_passed = (
        temporal["samples"] > 0 and temporal["abs_rel"] <= 0.01 and temporal["within_tolerance_rate"] >= 0.80
    )
    report["qualification_gate"] = {
        "passed": gate_passed,
        "maximum_abs_rel": 0.01,
        "minimum_within_tolerance_rate": 0.80,
        "deployable": gate_passed and not args.simulate,
    }
    report["camera_depth_correction"] = {
        "scale": report["calibration"]["scale"],
        "offset_m": report["calibration"]["offset_m"],
        "status": (
            "simulation_only" if args.simulate else ("verified" if gate_passed else "qualification_failed")
        ),
        "source_report": args.output.resolve().as_posix(),
    }
    save_qualification_report(args.output, report)
    corrected = report["corrected_in_profile_range"]
    print(f"report={args.output.resolve()}")
    print(
        "in_range_corrected_per_frame "
        f"mae_mm={corrected['mae_m'] * 1000:.3f} "
        f"abs_rel_pct={corrected['abs_rel'] * 100:.3f} "
        f"pass_pct={corrected['within_tolerance_rate'] * 100:.1f}"
    )
    print(
        "in_range_corrected_temporal "
        f"mae_mm={temporal['mae_m'] * 1000:.3f} "
        f"abs_rel_pct={temporal['abs_rel'] * 100:.3f} "
        f"pass_pct={temporal['within_tolerance_rate'] * 100:.1f}"
    )


if __name__ == "__main__":
    main()
