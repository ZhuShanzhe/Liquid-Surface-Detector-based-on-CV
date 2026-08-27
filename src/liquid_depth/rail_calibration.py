from __future__ import annotations

from typing import Any

import numpy as np


def _fit_affine(y_px: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(
        np.column_stack((y_px, np.ones_like(y_px))),
        depth_m,
        rcond=None,
    )[0]


def _predict_affine(coefficients: np.ndarray, y_px: np.ndarray) -> np.ndarray:
    return coefficients[0] * y_px + coefficients[1]


def _fit_projective(y_px: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    design = np.column_stack((y_px, np.ones_like(y_px), -depth_m * y_px))
    return np.linalg.lstsq(design, depth_m, rcond=None)[0]


def _predict_projective(coefficients: np.ndarray, y_px: np.ndarray) -> np.ndarray:
    denominator = coefficients[2] * y_px + 1.0
    if np.any(np.abs(denominator) < 1e-5):
        raise ValueError("Projective calibration has a singularity in the requested range")
    return (coefficients[0] * y_px + coefficients[1]) / denominator


def _loocv(
    y_px: np.ndarray,
    depth_m: np.ndarray,
    fit,
    predict,
) -> tuple[float, np.ndarray]:
    predictions = np.empty(len(y_px), dtype=np.float64)
    for index in range(len(y_px)):
        keep = np.arange(len(y_px)) != index
        predictions[index] = predict(
            fit(y_px[keep], depth_m[keep]),
            y_px[index : index + 1],
        )[0]
    errors = predictions - depth_m
    return float(np.mean(np.abs(errors))), errors


def _stable_projective(coefficients: np.ndarray, y_min: float, y_max: float) -> bool:
    sample = np.linspace(y_min, y_max, 101)
    denominator = coefficients[2] * sample + 1.0
    derivative_numerator = coefficients[0] - coefficients[2] * coefficients[1]
    return bool(np.all(np.abs(denominator) > 0.05) and abs(float(derivative_numerator)) > 1e-9)


def fit_rail_calibration(
    points: list[dict[str, float]],
    *,
    minimum_points: int = 3,
    max_horizontal_spread_px: float = 20.0,
    min_depth_span_m: float = 0.02,
) -> dict[str, Any]:
    if len(points) < minimum_points:
        raise ValueError(f"Rail calibration requires at least {minimum_points} points")
    x_px = np.asarray([item["x_px"] for item in points], dtype=np.float64)
    y_px = np.asarray([item["y_px"] for item in points], dtype=np.float64)
    depth_m = np.asarray([item["depth_m"] for item in points], dtype=np.float64)
    if not np.all(np.isfinite(np.column_stack((x_px, y_px, depth_m)))):
        raise ValueError("Rail calibration contains non-finite values")
    horizontal_spread = float(np.ptp(x_px))
    if horizontal_spread > max_horizontal_spread_px:
        raise ValueError(
            f"Clicked points span {horizontal_spread:.1f} px horizontally; "
            f"keep them on one reference rail within {max_horizontal_spread_px:.1f} px"
        )
    if float(np.ptp(depth_m)) < min_depth_span_m:
        raise ValueError(f"Known depths must span at least {min_depth_span_m * 1000.0:.1f} mm")
    if len(np.unique(np.round(y_px, 3))) != len(y_px):
        raise ValueError("Rail calibration points must have distinct vertical coordinates")
    order = np.argsort(y_px)
    sorted_depth = depth_m[order]
    differences = np.diff(sorted_depth)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError("Known depth must change monotonically along the reference rail")

    affine = _fit_affine(y_px, depth_m)
    affine_cv_mae, affine_cv_errors = _loocv(
        y_px,
        depth_m,
        _fit_affine,
        _predict_affine,
    )
    projective: np.ndarray | None = None
    projective_cv_mae: float | None = None
    projective_cv_errors: list[float] | None = None
    projective_stable = False
    if len(points) >= 4:
        projective = _fit_projective(y_px, depth_m)
        projective_cv_mae, projective_cv_errors = _loocv(
            y_px,
            depth_m,
            _fit_projective,
            _predict_projective,
        )
        projective_stable = _stable_projective(
            projective,
            float(y_px.min()),
            float(y_px.max()),
        )
    use_projective = (
        projective_stable and projective_cv_mae is not None and projective_cv_mae < affine_cv_mae * 0.9
    )
    if use_projective:
        assert projective is not None
        assert projective_cv_errors is not None
        assert projective_cv_mae is not None
        model = "projective"
        coefficients = projective
        predictions = _predict_projective(coefficients, y_px)
        cv_mae = projective_cv_mae
        cv_errors = projective_cv_errors
    else:
        model = "affine"
        coefficients = affine
        predictions = _predict_affine(coefficients, y_px)
        cv_mae = affine_cv_mae
        cv_errors = affine_cv_errors
    residual = predictions - depth_m
    return {
        "mode": "fixed_rail",
        "model": model,
        "coefficients": coefficients.tolist(),
        "rail_x_px": float(np.median(x_px)),
        "horizontal_spread_px": horizontal_spread,
        "calibration_y_range_px": [float(y_px.min()), float(y_px.max())],
        "calibration_depth_range_m": [float(depth_m.min()), float(depth_m.max())],
        "points": [
            {
                "x_px": float(x),
                "y_px": float(y),
                "depth_m": float(depth),
            }
            for x, y, depth in zip(x_px, y_px, depth_m, strict=True)
        ],
        "fit_mae_m": float(np.mean(np.abs(residual))),
        "fit_max_abs_error_m": float(np.max(np.abs(residual))),
        "loocv_mae_m": cv_mae,
        "loocv_max_abs_error_m": float(np.max(np.abs(cv_errors))),
        "candidate_models": {
            "affine_loocv_mae_m": affine_cv_mae,
            "projective_loocv_mae_m": projective_cv_mae,
            "projective_stable": projective_stable,
        },
    }


def rail_depth_from_y(
    calibration: dict[str, Any],
    y_px: float,
    *,
    extrapolation_margin_px: float = 2.0,
) -> float:
    low, high = (float(item) for item in calibration["calibration_y_range_px"])
    if not low - extrapolation_margin_px <= y_px <= high + extrapolation_margin_px:
        raise ValueError("rail_intersection_outside_calibrated_range")
    coefficients = np.asarray(calibration["coefficients"], dtype=np.float64)
    value = np.asarray([y_px], dtype=np.float64)
    if calibration["model"] == "affine":
        return float(_predict_affine(coefficients, value)[0])
    if calibration["model"] == "projective":
        return float(_predict_projective(coefficients, value)[0])
    raise ValueError(f"Unsupported rail calibration model: {calibration['model']}")


def intersect_curve_with_rail(
    curve_pixels: np.ndarray,
    point_confidences: np.ndarray,
    rail_x_px: float,
) -> tuple[float, float]:
    curve = np.asarray(curve_pixels, dtype=np.float64)
    confidence = np.asarray(point_confidences, dtype=np.float64).reshape(-1)
    if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) != len(confidence):
        raise ValueError("Curve and point confidences have incompatible shapes")
    order = np.argsort(curve[:, 0])
    x = curve[order, 0]
    y = curve[order, 1]
    score = confidence[order]
    unique_x, unique_indices = np.unique(x, return_index=True)
    y = y[unique_indices]
    score = score[unique_indices]
    if len(unique_x) < 2 or not unique_x[0] <= rail_x_px <= unique_x[-1]:
        raise ValueError("predicted_curve_does_not_cross_calibration_rail")
    return (
        float(np.interp(rail_x_px, unique_x, y)),
        float(np.interp(rail_x_px, unique_x, score)),
    )
