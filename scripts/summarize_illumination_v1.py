"""Post-hoc paired and wet/dry audit of illumination evaluations."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def metrics(rows):
    n = len(rows)
    errors = np.asarray([abs(r["prediction_m"] - r["truth_m"])
                         for r in rows if r["prediction_m"] is not None])
    passed = sum(r["prediction_m"] is not None and
                 abs(r["prediction_m"] - r["truth_m"]) <=
                 max(.02, .02 * r["truth_m"]) for r in rows)
    return {
        "frames": n,
        "coverage_pct": 100 * len(errors) / n if n else None,
        "all_frame_pass_pct": 100 * passed / n if n else None,
        "mae_mm": float(1000 * errors.mean()) if len(errors) else None,
        "p95_mm": float(1000 * np.percentile(errors, 95)) if len(errors) else None,
        "absolute_bins_all_frame_pct": (100 * np.histogram(
            errors, [0, .02, .05, .1, np.inf])[0] / n).tolist() if n else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(line) for line in a.records.read_text().splitlines()]
    pairs = defaultdict(dict)
    for row in rows:
        key = (row["case"], row["slot"], row["repeat"])
        pairs[key][row["route"]] = row
    new_only = defaultdict(list)
    if any(set(pair) != {"legacy_v45_masked", "merged_dry_geometry_v1"}
           for pair in pairs.values()):
        raise ValueError("complete_paired_routes_required")
    grouped = defaultdict(list)
    common = defaultdict(lambda: {"old": [], "new": []})
    for pair in pairs.values():
        old, new = pair["legacy_v45_masked"], pair["merged_dry_geometry_v1"]
        subset = "dry" if new["truth_m"] == 0 else "wet"
        for route, row in pair.items():
            grouped[f"{row['scene']}|{row['level']}|{subset}|{route}"].append(row)
        if old["prediction_m"] is not None and new["prediction_m"] is not None:
            key = f"{new['scene']}|{new['level']}"
            common[key]["old"].append(old)
            common[key]["new"].append(new)
        elif old["prediction_m"] is None and new["prediction_m"] is not None:
            new_only[f"{new['scene']}|{new['level']}"].append(new)
    result = {
        "total_unique_frames": len(pairs),
        "wet_dry": {key: metrics(group) for key, group in sorted(grouped.items())},
        "both_numeric_same_frames": {
            key: {"old": metrics(group["old"]), "new": metrics(group["new"])}
            for key, group in sorted(common.items())},
        "new_only_numeric_frames": {
            key: metrics(group) for key, group in sorted(new_only.items())},
    }
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print("AUDIT", len(pairs), len(rows))


if __name__ == "__main__":
    main()
