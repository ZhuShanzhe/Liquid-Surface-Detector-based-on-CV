"""Stratify the formal partial-loss route by true liquid height and measured loss.

Oracle surface masks are used here only for offline scoring and deletion
reconstruction, never by FormalSceneMeter at inference.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from evaluate_uniform_loss_v48 import delete_to_actual_loss

DEPTH_BANDS = (
    "empty", "0-0.1m", "0.1-1m", "1-2m", "2-3m", "3-4m",
    "4-5m", "5-6m", "6-7m", "7-8m", "8-9m", "9m+",
)
LOSS_BANDS = (
    "0-<5%", "5-<15%", "15-<25%", "25-<35%", "35-<45%", "45-50%",
)


def depth_band(height_m: float) -> str:
    if height_m == 0:
        return "empty"
    if height_m < .1:
        return "0-0.1m"
    if height_m < 1:
        return "0.1-1m"
    if height_m < 9:
        return f"{int(height_m)}-{int(height_m) + 1}m"
    return "9m+"


def loss_band(loss_pct: float) -> str:
    if not 0 <= loss_pct <= 50.05:
        raise ValueError("actual_loss_outside_0_to_50_pct")
    return LOSS_BANDS[min(5, int((loss_pct + 5) // 10))]


def stats(rows: list[dict], field: str) -> dict:
    errors = np.array([abs(row[field] - row["truth_m"]) for row in rows
                       if row[field] is not None], dtype=float)
    relative = np.array([abs(row[field] - row["truth_m"]) / row["truth_m"]
                         for row in rows if row[field] is not None
                         and row["truth_m"] > 0], dtype=float)
    n = len(rows)
    passed = sum(row[field] is not None and
                 abs(row[field] - row["truth_m"]) <=
                 max(.020, .02 * row["truth_m"]) for row in rows)
    return {
        "frames": n,
        "numeric": len(errors),
        "coverage_pct": 100 * len(errors) / n if n else None,
        "all_frame_pass_pct": 100 * passed / n if n else None,
        "mae_mm": 1000 * float(errors.mean()) if len(errors) else None,
        "p95_mm": 1000 * float(np.percentile(errors, 95)) if len(errors) else None,
        "abs_rel_pct_mean": 100 * float(relative.mean()) if len(relative) else None,
        "rel_error_p95_pct": (100 * float(np.percentile(relative, 95))
                              if len(relative) else None),
        "rel_error_bins_pct": [0, 1, 2, 5, 10, 20, "inf"],
        "rel_error_bin_counts": np.histogram(
            relative * 100, [0, 1, 2, 5, 10, 20, np.inf])[0].tolist(),
        "abs_error_bins_mm": [0, 5, 10, 20, 50, 100, 200, "inf"],
        "abs_error_bin_counts": np.histogram(
            errors * 1000, [0, 5, 10, 20, 50, 100, 200, np.inf]
        )[0].tolist(),
        "actual_loss_pct_mean": float(np.mean(
            [row["measured_loss_pct"] for row in rows])) if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--wet-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    evaluated_protocol = json.loads(
        (args.evaluation / "protocol.json").read_text())

    @lru_cache(maxsize=128)
    def mask_for(case: int) -> np.ndarray:
        folder = args.wet_root / f"{case * 6 + 5:05d}"
        with np.load(folder / "geometry.npz") as data:
            return data["mask"].astype(bool)

    @lru_cache(maxsize=256)
    def native_for(case: int, repeat: int) -> np.ndarray:
        folder = args.wet_root / f"{case * 6 + 5:05d}"
        return np.load(folder / f"raw_native_{repeat:02d}.npy",
                       allow_pickle=False)

    rows = []
    for line in (args.evaluation / "records.jsonl").read_text().splitlines():
        row = json.loads(line)
        case, repeat = int(row["case"]), int(row["repeat"])
        mask = mask_for(case)
        raw = native_for(case, repeat)
        if row["kind"] == "controlled":
            target = float(row["actual_loss_pct"])
            frame = delete_to_actual_loss(
                raw, mask, target, args.seed + case * 10007 + repeat * 509)
        else:
            target = None
            frame = raw
        measured = 100 * (1 - int((mask & (frame > 0)).sum()) /
                          max(int(mask.sum()), 1))
        row["target_loss_pct"] = target
        row["measured_loss_pct"] = measured
        row["depth_band"] = depth_band(float(row["truth_m"]))
        row["loss_band"] = loss_band(measured)
        rows.append(row)

    grouped = {
        "depth": defaultdict(list),
        "loss": defaultdict(list),
        "sensor": defaultdict(list),
        "depth_loss": defaultdict(lambda: defaultdict(list)),
    }
    for row in rows:
        grouped["depth"][row["depth_band"]].append(row)
        grouped["loss"][row["loss_band"]].append(row)
        grouped["sensor"][row["sensor"]].append(row)
        grouped["depth_loss"][row["depth_band"]][row["loss_band"]].append(row)

    def pair(group: list[dict]) -> dict:
        result = {"roi_quantile": stats(group, "v52_m"),
                  "single_frame": stats(group, "v56_m"),
                  "formal_selective": stats(group, "joint_m")}
        if evaluated_protocol.get("elastic_evidence_enabled"):
            result["v64_baseline"] = stats(group, "v64_m")
            result["elastic_candidate"] = stats(group, "joint_m")
        return result

    result = {
        "protocol": {
            "measurand": "actual liquid height above interior container bottom",
            "tolerance": "max(20 mm, 2% of true liquid height)",
            "actual_loss": "oracle-mask return fraction used offline only",
            "loss_band_edges": list(LOSS_BANDS),
            "depth_band_edges": list(DEPTH_BANDS),
            "formal_policy": (
                "single-frame ROI quantile; structured-light dry-layer rescue only when "
                "no supported wet layer and quantile is at least 0.25 m"
                if evaluated_protocol.get("layer_vote_enabled") else
                "legacy range-censored disparity with selective stationary temporal fit"),
            "zero_snap": False, "height_clip": False, "numeric_gate": False,
            "field_qualified": False,
        },
        "all": pair(rows),
        "by_depth": {key: pair(grouped["depth"][key])
                     for key in DEPTH_BANDS if key in grouped["depth"]},
        "by_loss": {key: pair(grouped["loss"][key])
                    for key in LOSS_BANDS if key in grouped["loss"]},
        "by_sensor": {key: pair(value)
                      for key, value in sorted(grouped["sensor"].items())},
        "depth_by_loss": {
            depth: {loss: pair(grouped["depth_loss"][depth][loss])
                    for loss in LOSS_BANDS
                    if loss in grouped["depth_loss"][depth]}
            for depth in DEPTH_BANDS if depth in grouped["depth_loss"]
        },
    }
    (args.evaluation / "stratified_summary.json").write_text(
        json.dumps(result, indent=2))
    with (args.evaluation / "stratified_records.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    print("STRATIFIED", len(rows),
          "depth_bands", len(result["by_depth"]),
          "loss_bands", len(result["by_loss"]))


if __name__ == "__main__":
    main()
