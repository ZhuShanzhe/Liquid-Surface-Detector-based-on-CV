from __future__ import annotations

import numpy as np


def _as_finite_vector(values, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(vector) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values")
    return vector


def confidence_error_correlation(errors, confidence) -> float:
    error = _as_finite_vector(errors, "errors")
    score = _as_finite_vector(confidence, "confidence")
    if error.shape != score.shape:
        raise ValueError("errors and confidence must have the same shape")
    if np.std(error) <= 1e-12 or np.std(score) <= 1e-12:
        return 0.0
    return float(np.corrcoef(score, error)[0, 1])


def summarize_selective_curve(
    errors,
    confidence,
    coverages: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.1),
) -> list[dict[str, float | int]]:
    """Summarize curve error after retaining highest-confidence samples."""
    error = _as_finite_vector(errors, "errors")
    score = _as_finite_vector(confidence, "confidence")
    if error.shape != score.shape:
        raise ValueError("errors and confidence must have the same shape")
    order = np.argsort(-score, kind="stable")
    result = []
    for coverage in coverages:
        if not 0.0 < float(coverage) <= 1.0:
            raise ValueError("coverages must be in (0, 1]")
        count = max(1, int(np.ceil(len(error) * float(coverage))))
        selected = error[order[:count]]
        result.append(
            {
                "target_coverage": float(coverage),
                "actual_coverage": count / len(error),
                "samples": count,
                "mean_px": float(selected.mean()),
                "median_px": float(np.median(selected)),
                "p95_px": float(np.percentile(selected, 95)),
            }
        )
    return result


def summarize_curve_group(errors, confidence) -> dict[str, object]:
    error = _as_finite_vector(errors, "errors")
    score = _as_finite_vector(confidence, "confidence")
    if error.shape != score.shape:
        raise ValueError("errors and confidence must have the same shape")
    return {
        "samples": len(error),
        "mean_px": float(error.mean()),
        "median_px": float(np.median(error)),
        "p95_px": float(np.percentile(error, 95)),
        "confidence_error_correlation": confidence_error_correlation(error, score),
        "selective_risk": summarize_selective_curve(error, score),
    }
