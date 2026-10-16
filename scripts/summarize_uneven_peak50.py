"""Pool strict uneven-surface holdouts under separate mean/peak tolerances."""

import argparse
import json
from pathlib import Path

import numpy as np


def load(folder):
    return [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()]


def metric(rows, prediction, truth, absolute_tol, relative_tol):
    valid = [row for row in rows if row[prediction] is not None]
    errors = np.asarray([abs(row[prediction] - row[truth]) for row in valid])
    passed = sum(
        abs(row[prediction] - row[truth]) <= max(absolute_tol, relative_tol * row[truth])
        for row in valid
    )
    total = len(rows)
    return {
        "frames": total,
        "numeric": len(valid),
        "coverage_pct": 100 * len(valid) / total if total else None,
        "all_frame_pass_pct": 100 * passed / total if total else None,
        "mae_mm": 1000 * float(errors.mean()) if len(errors) else None,
        "p95_mm": 1000 * float(np.percentile(errors, 95)) if len(errors) else None,
        "max_mm": 1000 * float(errors.max()) if len(errors) else None,
    }


def groups(rows):
    parts = {
        "all": rows,
        "wet_all": [r for r in rows if r["true_liquid_mean_m"] > 0],
        "dry_all": [r for r in rows if r["true_liquid_mean_m"] == 0],
        "wet_nonfar": [
            r for r in rows if r["true_liquid_mean_m"] > 0
            and not (r["camera_to_bottom_m"] >= 6 and r["true_liquid_mean_m"] <= 1)
        ],
        "wet_1to3": [r for r in rows if 1 <= r["true_liquid_mean_m"] < 3],
        "wet_4to9": [r for r in rows if 4 <= r["true_liquid_mean_m"] < 9],
        "dry_near": [
            r for r in rows if r["true_liquid_mean_m"] == 0
            and r["camera_to_bottom_m"] < 6
        ],
        "far_shallow_deferred": [
            r for r in rows if 0 < r["true_liquid_mean_m"] <= 1
            and r["camera_to_bottom_m"] >= 6
        ],
    }
    for low in range(9):
        parts[f"wet_{low}to{low + 1}"] = [
            r for r in rows
            if (0 < r["true_liquid_mean_m"] < 1 if low == 0 else
                low <= r["true_liquid_mean_m"] < low + 1)
        ]
    for scene in ("wave_only", "wave_plus_float"):
        parts[scene + "_wet"] = [
            r for r in rows if r["scenario"] == scene and r["true_liquid_mean_m"] > 0
        ]
    for sensor in ("active_stereo", "structured_light", "tof"):
        parts["sensor_" + sensor + "_wet"] = [
            r for r in rows if r["sensor"] == sensor and r["true_liquid_mean_m"] > 0
        ]
    return parts


def summarize(rows):
    return {
        name: {
            "mean": metric(part, "liquid_estimate_m", "true_liquid_mean_m", 0.020, 0.02),
            "highest": metric(part, "highest_estimate_m", "true_highest_m", 0.050, 0.05),
        }
        for name, part in groups(rows).items()
    }


def evidence_status(rows):
    wet = groups(rows)["wet_nonfar"]
    answer = {}
    for status in ("supported", "weak", "missing"):
        selected = [r for r in wet if r.get("highest_evidence_status") == status]
        failed = sum(
            r["highest_estimate_m"] is None
            or abs(r["highest_estimate_m"] - r["true_highest_m"])
            > max(0.050, 0.05 * r["true_highest_m"])
            for r in selected
        )
        answer[status] = {"frames": len(selected), "outside_tolerance": failed}
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", default=["f", "h", "i"])
    parser.add_argument("--calfix-seeds", nargs="+", default=["h"])
    parser.add_argument("--merged-v11", action="store_true")
    args = parser.parse_args()
    result = {
        "mean_tolerance": "max(20 mm, 2% true bottom-relative mean)",
        "highest_tolerance": "max(50 mm, 5% true bottom-relative highest)",
        "far_shallow_definition": "camera_to_bottom>=6m and 0<mean_height<=1m",
        "float_missing_returns_preserved": True,
        "research_only": True,
        "route": "merged_uneven_v11_research" if args.merged_v11 else "v10_candidate",
        "per_seed": {},
    }
    pooled_old, pooled_new = [], []
    for seed in args.seeds:
        suffix = "-calfix-v11" if seed in args.calfix_seeds else ""
        before = args.root / f"uneven-repair-{seed}-strict-baseline{suffix}-20261014"
        reference = args.root / f"uneven-repair-{seed}-strict-peak50-v10{suffix}-20261014"
        after = (args.root / f"uneven-merged-v11-final-{seed}-20261015"
                 if args.merged_v11 else reference)
        old, new = load(before), load(after)
        if len(old) != len(new) or any(
            (a["case"], a["repeat"], a["scenario"], a["true_liquid_mean_m"])
            != (b["case"], b["repeat"], b["scenario"], b["true_liquid_mean_m"])
            for a, b in zip(old, new)
        ):
            raise ValueError(f"unpaired_test_frames:{seed}")
        if args.merged_v11 and new != load(reference):
            raise ValueError(f"merged_route_differs_from_validated_candidate:{seed}")
        baseline_audit = json.loads((before / "installation-audit.json").read_text())
        audit = json.loads((after / "installation-audit.json").read_text())
        if len(audit) != 99 or len(baseline_audit) != 99:
            raise ValueError(f"incomplete_calibration_audit:{seed}")
        if any("error" in x for x in audit + baseline_audit):
            raise ValueError(f"invalid_calibration_prior_in_summary:{seed}")
        comparable = lambda items: [
            {k: v for k, v in item.items()
             if k not in ("annotation_attempt", "rejected_annotation_reasons")}
            for item in items
        ]
        if comparable(audit) != comparable(baseline_audit):
            raise ValueError(f"unpaired_calibration_priors:{seed}")
        if len(old) != 1584 or len(new) != 1584:
            raise ValueError(f"incomplete_test_frames:{seed}")
        result["per_seed"][seed] = {
            "baseline": summarize(old),
            "candidate": summarize(new),
            "candidate_evidence_status": evidence_status(new),
            "installation_failures": [x for x in audit if "error" in x],
            "baseline_source": str(before),
            "candidate_source": str(after),
            "validated_reference_source": str(reference),
            "merged_reference_exact_match": True if args.merged_v11 else None,
        }
        pooled_old.extend(old)
        pooled_new.extend(new)
    result["pooled"] = {
        "baseline": summarize(pooled_old),
        "candidate": summarize(pooled_new),
        "candidate_evidence_status": evidence_status(pooled_new),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

