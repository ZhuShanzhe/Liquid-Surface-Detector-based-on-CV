"""Paired glare/low-light evaluation with a dry RGB-D prior per installation."""
import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import torch
from evaluate_liquid_height_v21 import Predictor
from evaluate_other_scenes_v34 import predicted
from evaluate_shared_empty_v45 import install

from liquid_depth.empty_scene_routes_v44 import surface_candidate
from liquid_depth.illumination_scene_meter import IlluminationSceneMeter
from liquid_depth.liquid_height_metrics import HeightCalibration, reference_statistics

SCENES = {"low_light": 1, "glare": 2}
BANDS = ("dry", ".01-.1", ".1-1", "1-2", "2-3", "3-4",
         "4-5", "5-6", "6-7", "7-8", "8-9")


def summarize(rows):
    n = len(rows)
    numeric = [r for r in rows if r["prediction_m"] is not None]
    error = np.array([abs(r["prediction_m"] - r["truth_m"]) for r in numeric])
    passed = sum(abs(r["prediction_m"] - r["truth_m"]) <=
                 max(.02, .02 * r["truth_m"]) for r in numeric)
    wet = [r for r in numeric if r["truth_m"] > 0]
    relative = np.array([abs(r["prediction_m"] - r["truth_m"]) /
                         r["truth_m"] for r in wet])
    return {
        "frames": n, "installations": len({r["case"] for r in rows}),
        "numeric_coverage_pct": 100 * len(numeric) / n,
        "all_frame_pass_pct": 100 * passed / n,
        "mae_mm": float(1000 * error.mean()) if len(error) else None,
        "p95_mm": float(1000 * np.percentile(error, 95)) if len(error) else None,
        "abs_rel_pct": float(100 * relative.mean()) if len(relative) else None,
        "absolute_bins_all_frame_pct": (100 * np.histogram(
            error, [0, .02, .05, .1, np.inf])[0] / n).tolist(),
        "relative_bins_numeric_wet_pct": (100 * np.histogram(
            relative, [0, .01, .02, .05, .1, np.inf])[0] /
            len(relative)).tolist() if len(relative) else None,
        "median_rgb_luma": float(np.median([r["luma"] for r in rows])),
        "mean_dark_roi_fraction": float(np.mean([r["dark_fraction"] for r in rows])),
        "mean_saturated_roi_fraction": float(np.mean(
            [r["saturated_fraction"] for r in rows])),
        "mean_surface_depth_consistent_fraction": float(np.mean(
            [r["surface_depth_consistent_fraction"] for r in rows])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wet-root", type=Path, required=True)
    p.add_argument("--dry-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--cases", type=int, default=99)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--baseline", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--geometry-only", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    items, audits = {}, []
    for case in range(a.cases):
        try:
            item = install(case, a.wet_root, a.dry_root, a.seed,
                           source_slot_offset=1, source_slot_stride=18,
                           max_annotation_attempts=3)
            profile = ({"disparity_step_px": 1 / 16, "disparity_noise_px": .26}
                       if item["meta"]["sensor_family"] == "structured_light" else None)
            item["meter"] = IlluminationSceneMeter.from_empty_geometry(
                item["bundle"], item["dry"], max_return_depth_m=10.,
                quantized_disparity_profile=profile)
            items[case] = item
            audits.append(item["audit"])
        except (ValueError, FileNotFoundError) as exc:
            audits.append({"case": case, "initialization_error": str(exc)})
    (a.output / "installation-audit.json").write_text(json.dumps(audits, indent=2))
    if len(items) != a.cases:
        raise RuntimeError(f"calibration_preflight_failed:{len(items)}/{a.cases}")
    print("GEOMETRY", len(items), a.cases, flush=True)
    if a.geometry_only:
        return
    model = Predictor("v8") if a.baseline else None
    rows = []
    for case in range(a.cases):
        item = items[case]
        for level, (scene, offset) in product(range(3), SCENES.items()):
            folder = a.wet_root / f"{(case * 3 + level) * 6 + offset:05d}"
            meta = json.loads((folder / "metadata.json").read_text())
            if meta["scenario"] != scene:
                raise RuntimeError("scenario_slot_mismatch")
            rgb = cv2.imread(str(folder / "rgb.png"))
            if rgb is None:
                raise FileNotFoundError(folder / "rgb.png")
            frames = [f for f in json.loads((folder / "frames.json").read_text())
                      if f["repeat"] < a.repeats and f["variant"] == "base"]
            raw = [np.load(f["raw_path"], allow_pickle=False) for f in frames]
            masks = predicted(model, [rgb] * len(raw), raw) if model else None
            measured = [item["meter"].measure(scene, rgb_bgr=rgb, depth_m=d)
                        for d in raw]
            truth = float(reference_statistics(
                meta, replace(HeightCalibration.from_simulation(meta),
                              bottom_z_m=float(meta["interior_bottom_z_m"])))["mean_m"])
            with np.load(folder / "geometry.npz", allow_pickle=False) as geo:
                surface = geo["mask"].astype(bool)
                target = geo["target"]
            for rep, frame in enumerate(frames):
                quality = measured[rep]["diagnostics"]["illumination_quality"]
                base = {
                    "case": case, "slot": frame["slot"], "repeat": frame["repeat"],
                    "scene": scene, "level": meta.get("illumination_level",
                        ("mild", "moderate", "severe")[level]),
                    "sensor": meta["sensor_family"], "band": BANDS[meta["band"]],
                    "truth_m": truth, "camera_to_bottom_m": meta["camera_to_bottom_m"],
                    "surface_depth_consistent_fraction": float(np.mean(
                        np.abs(raw[rep][surface] - target[surface]) <=
                        np.maximum(.003, .01 * target[surface]))) if surface.any() else 0.,
                    "geometry_source": item["bundle"]["source"],
                    "luma": quality["median_luma_8bit"],
                    "dark_fraction": quality["dark_fraction_luma_lt_24"],
                    "saturated_fraction":
                        quality["saturated_fraction_any_channel_ge_250"],
                }
                candidates = [("merged_dry_geometry_v1",
                               measured[rep]["liquid_depth_m"],
                               "|".join(measured[rep]["reasons"]))]
                if masks:
                    old = surface_candidate(item["bundle"], rgb, raw[rep],
                                            masks[rep]["mask"], scene=scene)
                    candidates.append(("legacy_v45_masked",
                                       old["numeric_candidate_m"],
                                       "|".join(old["reasons"])))
                for route, value, reason in candidates:
                    number = float(value) if value is not None and np.isfinite(value) else None
                    rows.append({**base, "route": route, "prediction_m": number,
                                 "reason": reason if number is None else None})
        print(json.dumps({"case_done": case + 1, "rows": len(rows)}), flush=True)
    with (a.output / "records.jsonl").open("w") as handle:
        for record in rows:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    grouped = defaultdict(list)
    for record in rows:
        prefix = f"{record['scene']}|{record['level']}"
        route = record["route"]
        for key in (f"{prefix}|{route}", f"{record['scene']}|all|{route}",
                    f"{prefix}|{record['band']}|{route}",
                    f"{prefix}|{record['sensor']}|{route}"):
            grouped[key].append(record)
        if record["camera_to_bottom_m"] >= 6. and 0 < record["truth_m"] <= 1.:
            grouped[f"{prefix}|far_shallow|{route}"].append(record)
    (a.output / "summary.json").write_text(json.dumps(
        {key: summarize(group) for key, group in sorted(grouped.items())},
        indent=2, allow_nan=False))
    (a.output / "protocol.json").write_text(json.dumps({
        "seed": a.seed, "cases": a.cases, "repeats": a.repeats,
        "dry_depth_prior_per_installation": True,
        "operator_prior_truth_used_only_in_calibration_simulation": True,
        "truth_used_only_for_scoring": True, "all_calibrations_preflight": True,
        "tolerance_m": "max(0.02,0.02*true_height)",
        "no_zero_snap_or_clipping": True, "research_only": True,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("generate_illumination_stratified_v1.py"),
                Path(__file__).with_name("generate_other_scenes_v34.py"),
                Path(__file__).with_name("evaluate_shared_empty_v45.py"),
                Path(__file__).parents[1] / "src/liquid_depth/illumination_scene_meter.py",
                Path(__file__).parents[1] / "src/liquid_depth/supported_scene_meter.py")},
    }, indent=2))
    print("COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()
