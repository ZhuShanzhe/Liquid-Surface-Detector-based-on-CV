"""Replay uneven-surface repairs against frozen per-frame measurements."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_shared_empty_v45 import install
from evaluate_uneven_dual_height import add_floats, float_specs

from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.supported_scene_meter import FormalSceneMeter
from liquid_depth.uneven_surface_repair import (
    peak_candidates,
    prepare_evidence,
    repaired_liquid,
)


def metric(rows, key, truth):
    good = [r for r in rows if r["predictions"].get(key) is not None]
    e = np.array([abs(r["predictions"][key] - r[truth]) for r in good])
    p = sum(abs(r["predictions"][key] - r[truth]) <= max(0.02, 0.02 * r[truth]) for r in good)
    rel = [abs(r["predictions"][key] - r[truth]) / r[truth] for r in good if r[truth] > 0]
    return {
        "frames": len(rows),
        "numeric": len(good),
        "coverage_pct": 100 * len(good) / len(rows),
        "pass_pct": 100 * p / len(rows),
        "mae_mm": float(e.mean() * 1000) if len(e) else None,
        "p95_mm": float(np.percentile(e, 95) * 1000) if len(e) else None,
        "max_mm": float(e.max() * 1000) if len(e) else None,
        "absrel_pct": float(np.mean(rel) * 100) if rel else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wet-root", type=Path, required=True)
    p.add_argument("--dry-root", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=202610122159)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = [json.loads(x) for x in args.reference.read_text().splitlines()]
    lookup = {(r["case"], r["repeat"], r["scenario"]): r for r in reference}
    rows = []
    for case in sorted({r["case"] for r in reference}):
        item = install(case, args.wet_root, args.dry_root, args.seed)
        profile = (
            {"disparity_step_px": 1 / 16, "disparity_noise_px": 0.26}
            if item["meta"]["sensor_family"] == "structured_light"
            else None
        )
        base = FormalSceneMeter.from_empty_geometry(
            item["bundle"], item["dry"], max_return_depth_m=10.0, quantized_disparity_profile=profile
        )
        folder = args.wet_root / f"{case * 6 + 4:05d}"
        meta = json.loads((folder / "metadata.json").read_text())
        cal = replace(HeightCalibration.from_simulation(meta), bottom_z_m=meta["interior_bottom_z_m"])
        geom = np.load(folder / "geometry.npz")
        specs = float_specs(case, cal) if meta["nominal_liquid_height_m"] > 0 else []
        for scene in ("wave_only", "wave_plus_float"):
            reps = sorted(r["repeat"] for r in reference if r["case"] == case and r["scenario"] == scene)
            for rep in reps:
                original = lookup[(case, rep, scene)]
                raw = np.load(folder / f"raw_{rep:02d}_base.npy")
                depth = add_floats(
                    raw,
                    geom["target"],
                    geom["analytic_mask"].astype(bool),
                    cal,
                    specs if scene == "wave_plus_float" else [],
                )
                ev = prepare_evidence(base, item["dry"], depth)
                current = original["liquid_estimate_m"]
                source = original["liquid_value_source"] or ""
                repaired, reason = repaired_liquid(ev, current, source)
                blend, blend_reason = repaired_liquid(ev, current, source, blend=True)
                peaks = peak_candidates(ev, repaired)
                preds = {
                    "liquid_baseline": current,
                    "liquid_repaired": repaired,
                    "liquid_blend": blend,
                    "peak_baseline": original["highest_estimate_m"],
                    "peak_liquid_comparator": repaired,
                }
                preds.update({"peak_" + k: v for k, v in peaks.items() if k != "selected_count"})
                diag = {
                    k: ev[k]
                    for k in [
                        "paired_height_m",
                        "paired_support",
                        "paired_tiles",
                        "paired_tile_se_m",
                        "raw_quantile_m",
                    ]
                }
                rows.append(
                    {
                        **original,
                        "predictions": preds,
                        "repair_reason": reason,
                        "blend_reason": blend_reason,
                        "repair_evidence": diag,
                        "peak_selected": peaks.get("selected_count", 0),
                    }
                )
        print("case", case + 1, flush=True)
    methods = sorted({k for r in rows for k in r["predictions"]})
    groups = {
        "all": rows,
        "wet": [r for r in rows if r["band"] > 0],
        "shallow": [r for r in rows if r["band"] in (1, 2)],
        "deeper": [r for r in rows if r["band"] >= 3],
    }
    for band in range(11):
        groups[f"band_{band}"] = [r for r in rows if r["band"] == band]
    for scene in ("wave_only", "wave_plus_float"):
        groups[scene] = [r for r in rows if r["scenario"] == scene and r["band"] > 0]
    summary = {
        name: {
            key: metric(part, key, "true_liquid_mean_m" if key.startswith("liquid_") else "true_highest_m")
            for key in methods
        }
        for name, part in groups.items()
    }
    (args.output / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["wet"], indent=2))


if __name__ == "__main__":
    main()
