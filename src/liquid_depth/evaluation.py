from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_ground_truth(path: str | Path) -> dict[str, dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "frame_id" not in rows[0]:
        raise ValueError("Ground truth CSV requires a frame_id column")
    depth_column = next(
        (name for name in ("depth", "depth_cm", "liquid_depth") if name in rows[0]), None
    )
    if depth_column is None:
        raise ValueError("Ground truth CSV requires one of: depth, depth_cm, liquid_depth")
    normalized = {}
    for row in rows:
        item = dict(row)
        item["target_depth"] = item[depth_column]
        item["target_depth_column"] = depth_column
        normalized[item["frame_id"].strip()] = item
    return normalized


def load_predictions(root: str | Path) -> dict[str, dict]:
    results = {}
    for path in sorted(Path(root).rglob("depth_result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame_id = str(payload["frame_id"])
        if frame_id in results:
            raise ValueError(f"Duplicate prediction for frame_id={frame_id}: {path}")
        results[frame_id] = payload
    return results


def _metrics(rows: list[dict], tolerance: float) -> dict:
    total = len(rows)
    predicted = [row for row in rows if row["prediction"] is not None]
    accepted = [row for row in predicted if row["accepted"]]

    def error_stats(subset: list[dict]) -> dict:
        if not subset:
            return {
                "count": 0,
                "mae": None,
                "rmse": None,
                "median_ae": None,
                "p95_ae": None,
                "bias": None,
            }
        errors = np.asarray([row["prediction"] - row["target"] for row in subset], dtype=np.float64)
        absolute = np.abs(errors)
        return {
            "count": len(subset),
            "mae": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "median_ae": float(np.median(absolute)),
            "p95_ae": float(np.percentile(absolute, 95)),
            "bias": float(errors.mean()),
            "within_tolerance_ratio": float((absolute <= tolerance).mean()),
        }

    return {
        "ground_truth_count": total,
        "prediction_coverage": len(predicted) / total if total else 0.0,
        "acceptance_coverage": len(accepted) / total if total else 0.0,
        "all_predictions": error_stats(predicted),
        "accepted_predictions": error_stats(accepted),
    }


def evaluate(
    ground_truth_csv: str | Path,
    predictions_root: str | Path,
    tolerance: float = 1.0,
) -> dict:
    ground_truth = load_ground_truth(ground_truth_csv)
    predictions = load_predictions(predictions_root)
    rows = []
    for frame_id, target_row in ground_truth.items():
        prediction = predictions.get(frame_id)
        value = None if prediction is None else float(prediction["liquid_depth"])
        if value is not None and not math.isfinite(value):
            value = None
        rows.append(
            {
                "frame_id": frame_id,
                "target": float(target_row["target_depth"]),
                "prediction": value,
                "accepted": bool(prediction and prediction.get("accepted", False)),
                "scenario": target_row.get("scenario", "unspecified").strip() or "unspecified",
            }
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(row)
    return {
        "target_depth_column": next(iter(ground_truth.values()))["target_depth_column"],
        "tolerance": tolerance,
        "overall": _metrics(rows, tolerance),
        "by_scenario": {name: _metrics(items, tolerance) for name, items in sorted(grouped.items())},
        "missing_prediction_ids": [row["frame_id"] for row in rows if row["prediction"] is None],
        "unexpected_prediction_ids": sorted(set(predictions) - set(ground_truth)),
    }
