from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_accuracy_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path)
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.get("version") not in {1, 2}:
        raise ValueError("accuracy profile must be a supported version 1 or 2 mapping")
    bands = profile.get("bands")
    if not isinstance(bands, list) or not bands:
        raise ValueError("accuracy profile must contain at least one band")
    previous_max = None
    for band in bands:
        for field in ("name", "min_m", "max_m", "target", "deployment"):
            if field not in band:
                raise ValueError(f"accuracy band is missing {field}")
        minimum = float(band["min_m"])
        maximum = float(band["max_m"])
        if minimum >= maximum:
            raise ValueError(f"invalid accuracy band {band['name']}: min_m >= max_m")
        if previous_max is not None and minimum < previous_max:
            raise ValueError("accuracy bands must be sorted and non-overlapping")
        previous_max = maximum
        for level in ("target", "deployment"):
            for metric in ("mae", "p95", "bias", "temporal_std"):
                specification = band[level].get(metric)
                if not isinstance(specification, dict):
                    raise TypeError(f"{band['name']}.{level}.{metric} is required")
                if float(specification["absolute_floor_mm"]) < 0:
                    raise ValueError("absolute error floors must be non-negative")
                if float(specification["relative_percent"]) < 0:
                    raise ValueError("relative error limits must be non-negative")
    return profile


def find_accuracy_band(profile: dict[str, Any], depth_m: float) -> dict[str, Any] | None:
    bands = profile["bands"]
    for index, band in enumerate(bands):
        lower = float(band["min_m"])
        upper = float(band["max_m"])
        if depth_m >= lower and (depth_m < upper or (index == len(bands) - 1 and depth_m <= upper)):
            return band
    return None


def allowed_error_m(depth_m: np.ndarray, specification: dict[str, float]) -> np.ndarray:
    absolute_floor_m = float(specification["absolute_floor_mm"]) / 1000.0
    relative = float(specification["relative_percent"]) / 100.0
    return np.maximum(absolute_floor_m, np.abs(depth_m) * relative)


def _band_summary(
    truth_m: np.ndarray,
    prediction_m: np.ndarray,
    specification: dict[str, Any],
) -> dict[str, Any]:
    signed_error = prediction_m - truth_m
    absolute_error = np.abs(signed_error)
    relative_error = absolute_error / np.maximum(np.abs(truth_m), 1e-9)
    mae_tolerance = allowed_error_m(truth_m, specification["mae"])
    p95_tolerance = allowed_error_m(truth_m, specification["p95"])
    bias_tolerance = allowed_error_m(truth_m, specification["bias"])
    normalized_mae = float(np.mean(absolute_error / mae_tolerance))
    normalized_p95 = float(np.percentile(absolute_error / p95_tolerance, 95))
    normalized_bias = float(abs(np.mean(signed_error)) / np.mean(bias_tolerance))
    return {
        "accepted_samples": len(truth_m),
        "mae_mm": float(absolute_error.mean() * 1000.0),
        "rmse_mm": float(np.sqrt(np.mean(absolute_error**2)) * 1000.0),
        "signed_bias_mm": float(signed_error.mean() * 1000.0),
        "p95_absolute_error_mm": float(np.percentile(absolute_error, 95) * 1000.0),
        "mape_percent": float(relative_error.mean() * 100.0),
        "p95_relative_error_percent": float(np.percentile(relative_error, 95) * 100.0),
        "normalized_mae": normalized_mae,
        "normalized_p95": normalized_p95,
        "normalized_bias": normalized_bias,
        "within_mae_tolerance_ratio": float(np.mean(absolute_error <= mae_tolerance)),
        "within_p95_tolerance_ratio": float(np.mean(absolute_error <= p95_tolerance)),
        "passes_error_gate": bool(normalized_mae <= 1.0 and normalized_p95 <= 1.0 and normalized_bias <= 1.0),
    }


def evaluate_accuracy_profile(
    truth_m: np.ndarray | list[float],
    prediction_m: np.ndarray | list[float],
    profile: dict[str, Any],
    *,
    confidence: np.ndarray | list[float] | None = None,
    confidence_threshold: float = 0.0,
    level: str = "target",
) -> dict[str, Any]:
    if level not in {"target", "deployment"}:
        raise ValueError("level must be target or deployment")
    truth = np.asarray(truth_m, dtype=np.float64)
    prediction = np.asarray(prediction_m, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("truth_m and prediction_m must be equal-length one-dimensional arrays")
    if confidence is None:
        confidence_values = np.ones_like(truth)
    else:
        confidence_values = np.asarray(confidence, dtype=np.float64)
        if confidence_values.shape != truth.shape:
            raise ValueError("confidence must have the same shape as truth_m")
    finite = np.isfinite(truth) & np.isfinite(prediction) & np.isfinite(confidence_values)
    accepted = finite & (confidence_values >= confidence_threshold)
    minimum_coverage = float(profile["policy"]["minimum_overall_accepted_coverage"])
    report: dict[str, Any] = {
        "profile": profile["name"],
        "level": level,
        "samples": len(truth),
        "accepted": int(accepted.sum()),
        "coverage": float(accepted.mean()) if len(truth) else 0.0,
        "confidence_threshold": confidence_threshold,
        "minimum_required_coverage": minimum_coverage,
        "bands": {},
    }
    in_profile = np.zeros_like(accepted)
    all_error_gates_pass = True
    for index, band in enumerate(profile["bands"]):
        lower = float(band["min_m"])
        upper = float(band["max_m"])
        band_population = finite & (truth >= lower) & (truth < upper)
        if index == len(profile["bands"]) - 1:
            band_population = finite & (truth >= lower) & (truth <= upper)
        in_profile |= band_population
        band_accepted = band_population & accepted
        band_report: dict[str, Any] = {
            "samples": int(band_population.sum()),
            "accepted": int(band_accepted.sum()),
            "coverage": float(band_accepted.sum() / max(band_population.sum(), 1)),
        }
        if band_accepted.any():
            band_report.update(_band_summary(truth[band_accepted], prediction[band_accepted], band[level]))
            all_error_gates_pass &= band_report["passes_error_gate"]
        report["bands"][band["name"]] = band_report
    report["outside_profile_samples"] = int((finite & ~in_profile).sum())
    report["passes_coverage_gate"] = bool(report["coverage"] >= minimum_coverage)
    report["passes_profile"] = bool(
        report["passes_coverage_gate"] and all_error_gates_pass and report["outside_profile_samples"] == 0
    )
    return report
