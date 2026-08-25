from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class QualityAssessment:
    accepted: bool
    confidence: float
    rejection_reasons: tuple[str, ...]
    scores: dict[str, float]


def _lower_score(value: float, threshold: float) -> float:
    if threshold <= 0:
        return max(0.0, min(1.0, value))
    return max(0.0, min(1.0, value / threshold))


def _upper_score(value: float, limit: float) -> float:
    if limit <= 0:
        return 1.0 if value <= 0 else 0.0
    if value <= 0:
        return 1.0
    return max(0.0, min(1.0, limit / value))


def assess_quality(metrics: dict[str, float | int], thresholds: dict) -> QualityAssessment:
    checks = (
        (
            "plane_support",
            float(metrics["inlier_ratio"]),
            float(thresholds.get("min_inlier_ratio", 0.30)),
            "low_plane_inlier_ratio",
            "lower",
        ),
        (
            "plane_residual",
            float(metrics["median_residual_m"]),
            float(thresholds.get("max_median_residual_m", 0.006)),
            "high_plane_residual",
            "upper",
        ),
        (
            "mask_area",
            float(metrics["mask_area_px"]),
            float(thresholds.get("min_mask_area", 10000)),
            "mask_too_small",
            "lower",
        ),
        (
            "segmentation",
            float(metrics["mean_segmentation_confidence"]),
            float(thresholds.get("min_mean_segmentation_confidence", 0.0)),
            "low_segmentation_confidence",
            "lower",
        ),
        (
            "depth",
            float(metrics["mean_depth_confidence"]),
            float(thresholds.get("min_mean_depth_confidence", 0.0)),
            "low_depth_confidence",
            "lower",
        ),
        (
            "parallelism",
            float(metrics["plane_angle_deg"]),
            float(thresholds.get("max_plane_angle_deg", 15.0)),
            "liquid_bottom_plane_not_parallel",
            "upper",
        ),
    )
    reasons: list[str] = []
    scores: dict[str, float] = {}
    for name, value, threshold, reason, direction in checks:
        passed = value >= threshold if direction == "lower" else value <= threshold
        if not passed:
            reasons.append(reason)
        scores[name] = (
            _lower_score(value, threshold) if direction == "lower" else _upper_score(value, threshold)
        )

    positive = [max(score, 1e-6) for score in scores.values()]
    confidence = math.prod(positive) ** (1.0 / len(positive))
    return QualityAssessment(not reasons, float(confidence), tuple(reasons), scores)
