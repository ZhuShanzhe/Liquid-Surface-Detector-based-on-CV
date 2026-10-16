"""Summarize paired uneven-surface repair trials, including tails and error bins."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(folder):
    return [json.loads(x) for x in (folder / "records.jsonl").read_text().splitlines()]


def metric(rows, pred, truth, *, absolute_tolerance_m, relative_tolerance):
    good = [r for r in rows if r[pred] is not None and np.isfinite(r[pred])]
    err = np.array([abs(r[pred] - r[truth]) for r in good])
    rel = np.array([abs(r[pred] - r[truth]) / r[truth] for r in good if r[truth] > 0])
    passed = sum(e <= max(absolute_tolerance_m, relative_tolerance * r[truth])
                 for e, r in zip(err, good))
    bins = np.array([0, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, np.inf])
    rel_bins = np.array([0, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100, np.inf])
    n = len(rows)
    return {
        "frames": n,
        "numeric": len(good),
        "coverage_pct": 100 * len(good) / n if n else None,
        "all_frame_pass_pct": 100 * passed / n if n else None,
        "mae_mm": float(err.mean() * 1000) if len(err) else None,
        "median_mm": float(np.median(err) * 1000) if len(err) else None,
        "p95_mm": float(np.percentile(err, 95) * 1000) if len(err) else None,
        "p99_mm": float(np.percentile(err, 99) * 1000) if len(err) else None,
        "max_mm": float(err.max() * 1000) if len(err) else None,
        "absrel_pct": float(rel.mean() * 100) if len(rel) else None,
        "absolute_bins_all_pct": (np.histogram(err, bins)[0] * 100 / n).tolist() if n else [],
        "relative_bins_numeric_pct": (np.histogram(rel, rel_bins)[0] * 100 / len(rel)).tolist()
        if len(rel)
        else [],
    }


def summarize(rows):
    groups = {
        "all": rows,
        "wet": [r for r in rows if r["band"] > 0],
        "dry": [r for r in rows if r["band"] == 0],
        "far_shallow": [r for r in rows if 0 < r["true_liquid_mean_m"] <= 1 and r["camera_to_bottom_m"] >= 6],
    }
    for band in range(11):
        groups[f"band_{band}"] = [r for r in rows if r["band"] == band]
    for sensor in ("active_stereo", "structured_light", "tof"):
        groups[sensor] = [r for r in rows if r["sensor"] == sensor and r["band"] > 0]
    return {
        name: {
            "liquid": metric(part, "liquid_estimate_m", "true_liquid_mean_m",
                             absolute_tolerance_m=0.020, relative_tolerance=0.02),
            "peak": metric(part, "highest_estimate_m", "true_highest_m",
                           absolute_tolerance_m=0.050, relative_tolerance=0.05),
        }
        for name, part in groups.items()
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    pairs = {
        "previous_v45": ("uneven-dual-current-v45-20261014", "uneven-repair-v45-final-v4-20261014"),
        "new_a": ("uneven-repair-a-baseline-v2-20261014", "uneven-repair-a-final-v4-20261014"),
        "new_b": ("uneven-repair-b-baseline-v2-20261014", "uneven-repair-b-final-v4-20261014"),
        "strict_c": ("uneven-repair-c-strict-baseline-20261014", "uneven-repair-c-strict-final-v4-20261014"),
        "strict_d": (
            "uneven-repair-d-strict-baseline-v4-20261014",
            "uneven-repair-d-strict-final-v4-20261014",
        ),
    }
    result = {
        "liquid_tolerance": "max(20 mm, 2% true mean height)",
        "peak_tolerance": "max(50 mm, 5% true highest height)",
        "absolute_bin_edges_mm": [0, 5, 10, 20, 50, 100, 200, "inf"],
        "relative_bin_edges_pct": [0, 0.5, 1, 2, 3, 5, 10, "inf"],
        "route_default_enabled": False,
        "pairs": {},
    }
    pools = {"legacy_new_a_b": [[], []], "strict_new_c_d": [[], []]}
    for name, (old_name, new_name) in pairs.items():
        old, new = load(a.root / old_name), load(a.root / new_name)
        assert len(old) == len(new)
        assert all(
            (x["case"], x["repeat"], x["scenario"], x["true_liquid_mean_m"])
            == (y["case"], y["repeat"], y["scenario"], y["true_liquid_mean_m"])
            for x, y in zip(old, new)
        )
        result["pairs"][name] = {
            "baseline": summarize(old),
            "candidate": summarize(new),
            "baseline_source": str(a.root / old_name),
            "candidate_source": str(a.root / new_name),
        }
        category = (
            "legacy_new_a_b"
            if name.startswith("new_")
            else "strict_new_c_d"
            if name.startswith("strict_")
            else None
        )
        if category is not None:
            pools[category][0].extend(old)
            pools[category][1].extend(new)
    for category, (pooled_old, pooled_new) in pools.items():
        result[category] = {"baseline": summarize(pooled_old), "candidate": summarize(pooled_new)}
    src = Path(__file__).parents[1] / "src/liquid_depth/uneven_surface_repair.py"
    result["repair_source_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result, indent=2))
    print("wrote", a.output)


if __name__ == "__main__":
    main()
