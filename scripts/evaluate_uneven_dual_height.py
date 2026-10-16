"""Research-only dual-height test for uneven liquid and floats."""

import argparse
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_shared_empty_v45 import install

from liquid_depth.liquid_height_metrics import HeightCalibration, world_points
from liquid_depth.simulation import SyntheticScene, surface_height_and_gradient
from liquid_depth.uneven_surface_meter import UnevenSurfaceMeter


def scene_from_meta(meta):
    names = {f.name for f in fields(SyntheticScene)}
    return SyntheticScene(**{k: v for k, v in meta.items() if k in names})


def float_specs(case, cal):
    rng = np.random.default_rng(71803 + case)
    return [
        (
            rng.uniform(-0.5, 0.5) * cal.radii_m[0],
            rng.uniform(-0.5, 0.5) * cal.radii_m[1],
            rng.uniform(0.08, 0.13) * cal.radii_m[0],
            rng.uniform(0.08, 0.13) * cal.radii_m[1],
            rng.uniform(0.030, 0.060),
        )
        for _ in range(3)
    ]


def bump_field(x, y, specs):
    bump = np.zeros_like(x, dtype=float)
    for cx, cy, rx, ry, peak in specs:
        r2 = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
        bump = np.maximum(bump, peak * np.maximum(0, 1 - r2))
    return bump


def truth(meta, cal, specs):
    xy = cal.area_grid(size=224)
    z = surface_height_and_gradient(scene_from_meta(meta), xy[:, 0], xy[:, 1])[0]
    liquid = z - cal.bottom_z_m
    return float(liquid.mean()), float((liquid + bump_field(xy[:, 0], xy[:, 1], specs)).max())


def add_floats(raw, target, mask, cal, specs, *, preserve_missing=False):
    if not specs:
        return raw
    xyz = world_points(target, cal)
    bump = bump_field(xyz[..., 0], xyz[..., 1], specs)
    valid = mask & (bump > 0) & (target > 0)
    if preserve_missing:
        valid &= np.isfinite(raw) & (raw > 0)
    v, u = np.indices(raw.shape)
    k, t = cal.intrinsics, cal.camera_to_world_cv
    dz = t[2, 0] * (u - k[0, 2]) / k[0, 0]
    dz += t[2, 1] * (v - k[1, 2]) / k[1, 1] + t[2, 2]
    out = raw.copy()
    source = np.where(raw > 0, raw, target)
    out[valid] = source[valid] + bump[valid] / dz[valid]
    return out


def metric(rows, pred, truth_key, *, absolute_tolerance_m, relative_tolerance):
    good = [r for r in rows if r[pred] is not None]
    err = np.array([abs(r[pred] - r[truth_key]) for r in good])
    passed = sum(
        abs(r[pred] - r[truth_key]) <= max(absolute_tolerance_m, relative_tolerance * r[truth_key])
        for r in good
    )
    n = len(rows)
    return {
        "frames": n,
        "numeric": len(good),
        "coverage_pct": 100 * len(good) / n if n else None,
        "all_frame_pass_pct": 100 * passed / n if n else None,
        "mae_mm": float(1000 * err.mean()) if len(err) else None,
        "p95_mm": float(1000 * np.percentile(err, 95)) if len(err) else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wet-root", type=Path, required=True)
    p.add_argument("--dry-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", type=int, default=99)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--seed", type=int, default=202610122159)
    p.add_argument("--temporal-liquid", action="store_true")
    p.add_argument("--dry-paired-shallow", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--evidence-repair", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--float-preserve-missing", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # Validate every installation before scoring any wet frame. A bad prior is
    # repaired/reannotated upstream; it must never become a missing prediction.
    preflight, audits = [], []
    for case in range(args.cases):
        try:
            item = install(
                case, args.wet_root, args.dry_root, args.seed,
                max_annotation_attempts=16,
            )
            profile = (
                {"disparity_step_px": 1 / 16, "disparity_noise_px": 0.26}
                if item["meta"]["sensor_family"] == "structured_light"
                else None
            )
            meter = UnevenSurfaceMeter.from_empty_geometry(
                item["bundle"], item["dry"], max_return_depth_m=10.0,
                quantized_disparity_profile=profile,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise RuntimeError(f"calibration_preflight_failed:case={case}:{exc}") from exc
        preflight.append((item, meter))
        audits.append(item["audit"])
    args.output.mkdir(parents=True, exist_ok=False)
    if args.validate_only:
        (args.output / "installation-audit.json").write_text(
            json.dumps(audits, indent=2) + "\n"
        )
        return
    rows = []
    for case in range(args.cases):
        item, meter = preflight[case]
        folder = args.wet_root / f"{case * 6 + 4:05d}"
        meta = item["meta"]
        cal = replace(HeightCalibration.from_simulation(meta), bottom_z_m=meta["interior_bottom_z_m"])

        geom = np.load(folder / "geometry.npz", allow_pickle=False)
        specs = float_specs(case, cal) if meta["nominal_liquid_height_m"] > 0 else []
        for scenario, selected in (("wave_only", []), ("wave_plus_float", specs)):
            mean, highest = truth(meta, cal, selected)
            history = []
            for rep in range(args.repeats):
                raw = np.load(folder / f"raw_{rep:02d}_base.npy", allow_pickle=False)
                depth = add_floats(
                    raw,
                    geom["target"],
                    geom["analytic_mask"].astype(bool),
                    cal,
                    selected,
                    preserve_missing=args.float_preserve_missing,
                )
                result = meter.measure(
                    depth,
                    recent_depth_frames=history[-3:],
                    stationary_confirmed=args.temporal_liquid,
                    dry_paired_shallow_enabled=args.dry_paired_shallow,
                    evidence_repair_enabled=args.evidence_repair,
                )
                liquid = result["liquid_surface_height_m"]
                top = result["highest_visible_height_m"]
                count = result["highest_roi_points"]
                flags = result["quality_flags"]
                route = result["liquid_route"]
                value_source = result["liquid_value_source"]
                liquid_flags = result["liquid_quality_flags"]
                temporal_count = result["temporal_liquid_frames_used"]
                paired = result["dry_paired_shallow"]
                repair = result["uneven_evidence_repair"]
                evidence_status = result["highest_evidence_status"]
                evidence_warnings = result["highest_evidence_warnings"]
                history.append(depth)
                rows.append(
                    {
                        "case": case,
                        "repeat": rep,
                        "scenario": scenario,
                        "sensor": meta["sensor_family"],
                        "band": meta["band"],
                        "camera_to_bottom_m": meta["camera_to_bottom_m"],
                        "wave_amplitude_m": meta["wave_amplitude_m"],
                        "true_liquid_mean_m": mean,
                        "true_highest_m": highest,
                        "liquid_estimate_m": liquid,
                        "highest_estimate_m": top,
                        "valid_roi_points": count,
                        "quality_flags": flags,
                        "highest_evidence_status": evidence_status,
                        "highest_evidence_warnings": evidence_warnings,
                        "liquid_route": route,
                        "liquid_value_source": value_source,
                        "liquid_quality_flags": liquid_flags,
                        "temporal_liquid_frames_used": temporal_count,
                        "dry_paired_shallow": paired,
                        "uneven_evidence_repair": repair,
                    }
                )
        print("case", case + 1, flush=True)
    groups = {"all": rows}
    groups["wet_only"] = [r for r in rows if r["band"] > 0]
    groups["wet_far_shallow_deferred"] = [
        r for r in rows if r["true_liquid_mean_m"] > 0
        and r["camera_to_bottom_m"] >= 6 and r["true_liquid_mean_m"] <= 1
    ]
    groups["wet_nonfar"] = [
        r for r in rows if r["true_liquid_mean_m"] > 0
        and not (r["camera_to_bottom_m"] >= 6 and r["true_liquid_mean_m"] <= 1)
    ]
    groups["wet_1to3"] = [r for r in rows if 1 <= r["true_liquid_mean_m"] < 3]
    groups["wet_4to9"] = [r for r in rows if 4 <= r["true_liquid_mean_m"] < 9]
    groups["dry_near"] = [
        r for r in rows if r["true_liquid_mean_m"] == 0 and r["camera_to_bottom_m"] < 6
    ]
    for scenario in ("wave_only", "wave_plus_float"):
        groups[scenario] = [r for r in rows if r["scenario"] == scenario]
        groups[scenario + "_wet_only"] = [r for r in rows if r["scenario"] == scenario and r["band"] > 0]
    for sensor in ("active_stereo", "structured_light", "tof"):
        groups[sensor] = [r for r in rows if r["sensor"] == sensor]
    for band in range(11):
        groups[f"band_{band}"] = [r for r in rows if r["band"] == band]
    summary = {
        "protocol": {
            "wet_root": str(args.wet_root),
            "dry_root": str(args.dry_root),
            "cases": args.cases,
            "repeats": args.repeats,
            "liquid_tolerance": "max(20 mm, 2% true mean height)",
            "highest_tolerance": "max(50 mm, 5% true highest height)",
            "temporal_liquid": args.temporal_liquid,
            "dry_paired_shallow": args.dry_paired_shallow,
            "evidence_repair": args.evidence_repair,
            "truth": "bottom-relative area mean and maximum",
            "float_proxy": "three opaque elliptical domes in depth;",
            "rgb_float_rendered": False,
            "float_missing_depth_preserved": args.float_preserve_missing,
            "research_only": True,
        },
        "groups": {
            name: {
                "liquid_mean": metric(part, "liquid_estimate_m", "true_liquid_mean_m",
                                      absolute_tolerance_m=0.020, relative_tolerance=0.02),
                "highest": metric(part, "highest_estimate_m", "true_highest_m",
                                  absolute_tolerance_m=0.050, relative_tolerance=0.05),
            }
            for name, part in groups.items()
        },
        "installation_failures": [a for a in audits if "error" in a],
    }
    (args.output / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "installation-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(json.dumps(summary["groups"], indent=2))


if __name__ == "__main__":
    main()
