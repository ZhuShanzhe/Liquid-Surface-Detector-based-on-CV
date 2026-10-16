"""Paired ablation: apply the partial-loss estimator to ordinary frames."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from evaluate_shared_empty_v45 import install

from liquid_depth.empty_geometry_v41 import height_calibration
from liquid_depth.liquid_height_metrics import (
    HeightCalibration,
    reference_statistics,
    world_points,
)
from liquid_depth.supported_scene_meter import FormalSceneMeter


def legacy_broad_median(meter, depth):
    cal = height_calibration(meter.bundle)
    xyz = world_points(depth, cal)
    valid = meter.ordinary_container_roi & np.isfinite(depth) & (depth > 0)
    values = xyz[..., 2][valid] - cal.bottom_z_m
    return float(np.median(values)) if len(values) else None


def summarize(rows, key):
    numeric = [r for r in rows if r[key] is not None]
    errors = np.asarray([abs(r[key] - r["true_height_m"]) for r in numeric])
    passes = sum(abs(r[key] - r["true_height_m"]) <= max(0.020, 0.02 * r["true_height_m"]) for r in numeric)
    return {
        "frames": len(rows),
        "numeric": len(numeric),
        "coverage_pct": 100 * len(numeric) / len(rows) if rows else None,
        "all_frame_pass_pct": 100 * passes / len(rows) if rows else None,
        "mae_mm": float(1000 * errors.mean()) if len(errors) else None,
        "p95_mm": float(1000 * np.percentile(errors, 95)) if len(errors) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wet-root", type=Path, required=True)
    parser.add_argument("--dry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cases", type=int, default=99)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in range(args.cases):
        folder = args.wet_root / f"{case * 6 + 5:05d}"
        meta = json.loads((folder / "metadata.json").read_text())
        cal = replace(
            HeightCalibration.from_simulation(meta),
            bottom_z_m=float(meta["interior_bottom_z_m"]),
        )
        truth = float(reference_statistics(meta, cal)["mean_m"])
        distance = float(meta["camera_to_bottom_m"])
        try:
            item = install(case, args.wet_root, args.dry_root, args.seed)
            profile = (
                {"disparity_step_px": 1 / 16, "disparity_noise_px": 0.26}
                if meta["sensor_family"] == "structured_light"
                else None
            )
            meter = FormalSceneMeter.from_empty_geometry(
                item["bundle"],
                item["dry"],
                max_return_depth_m=10.0,
                quantized_disparity_profile=profile,
            )
            failure = None
        except (ValueError, FileNotFoundError) as exc:
            meter = None
            failure = str(exc)
        for rep in range(args.repeats):
            if meter is None:
                ordinary = partial = None
                route = "geometry_initialization_failed"
            else:
                raw = np.load(folder / f"raw_native_{rep:02d}.npy", allow_pickle=False)
                ordinary = legacy_broad_median(meter, raw)
                formal = meter.measure("ordinary_diffuse", depth_m=raw)
                partial = formal["liquid_depth_m"]
                route = formal["diagnostics"]["underlying_return_route"]
            rows.append(
                {
                    "case": case,
                    "repeat": rep,
                    "sensor": meta["sensor_family"],
                    "camera_to_bottom_m": distance,
                    "true_height_m": truth,
                    "ordinary_height_m": ordinary,
                    "partial_method_height_m": partial,
                    "partial_method_route": route,
                    "geometry_failure": failure,
                }
            )
        print("case", case + 1, flush=True)
    groups = {
        "all": rows,
        "far_shallow": [r for r in rows if r["camera_to_bottom_m"] >= 6.0 and r["true_height_m"] <= 0.5],
        "other": [r for r in rows if not (r["camera_to_bottom_m"] >= 6.0 and r["true_height_m"] <= 0.5)],
    }
    for sensor in sorted({r["sensor"] for r in rows}):
        groups[sensor] = [r for r in rows if r["sensor"] == sensor]
        groups["far_shallow_" + sensor] = [r for r in groups["far_shallow"] if r["sensor"] == sensor]
    summary = {
        "protocol": {
            "wet_root": str(args.wet_root),
            "dry_root": str(args.dry_root),
            "seed": args.seed,
            "cases": args.cases,
            "repeats": args.repeats,
            "sensor_profile": {
                "max_return_depth_m": 10.0,
                "structured_light_disparity_step_px": 1 / 16,
                "structured_light_disparity_noise_px": 0.26,
            },
            "candidate_uses_truth": False,
            "tolerance": "max(20 mm, 2% true bottom-relative liquid height)",
        },
        "groups": {
            name: {
                "ordinary": summarize(group, "ordinary_height_m"),
                "partial_method": summarize(group, "partial_method_height_m"),
            }
            for name, group in groups.items()
        },
    }
    (args.output / "records.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["groups"], indent=2), flush=True)


if __name__ == "__main__":
    main()
