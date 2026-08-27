from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .container_geometry import (
    ContactGeometryEstimate,
    ContainerModel,
    estimate_level_from_contact_curve,
)


@dataclass(frozen=True)
class SparseContactSelection:
    """A confidence-filtered, spatially distributed subset of contact pixels."""

    points_px: np.ndarray
    confidences: np.ndarray
    source_indices: np.ndarray
    input_points: int
    reliable_points: int
    horizontal_span_ratio: float
    occupied_bins: int
    horizontal_bins: int

    @property
    def mean_confidence(self) -> float:
        return float(self.confidences.mean()) if len(self.confidences) else 0.0

    @property
    def reliable_ratio(self) -> float:
        return self.reliable_points / max(self.input_points, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_points": self.input_points,
            "reliable_points": self.reliable_points,
            "selected_points": len(self.points_px),
            "reliable_ratio": self.reliable_ratio,
            "mean_selected_confidence": self.mean_confidence,
            "horizontal_span_ratio": self.horizontal_span_ratio,
            "occupied_horizontal_bins": self.occupied_bins,
            "horizontal_bins": self.horizontal_bins,
            "source_indices": self.source_indices.tolist(),
        }


@dataclass(frozen=True)
class SparseLevelEstimate:
    """Metric level result augmented with sparse-observability diagnostics."""

    geometry: ContactGeometryEstimate | None
    selection: SparseContactSelection
    selection_rejection_reasons: tuple[str, ...]

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        geometry_reasons = self.geometry.rejection_reasons if self.geometry else ()
        return tuple(dict.fromkeys((*self.selection_rejection_reasons, *geometry_reasons)))

    @property
    def accepted(self) -> bool:
        return self.geometry is not None and self.geometry.level_m is not None and not self.rejection_reasons

    @property
    def level_m(self) -> float | None:
        return self.geometry.level_m if self.geometry else None

    @property
    def uncertainty_m(self) -> float | None:
        return self.geometry.uncertainty_m if self.geometry else None

    @property
    def confidence(self) -> float:
        if not self.accepted or self.geometry is None:
            return 0.0
        bin_score = min(1.0, self.selection.occupied_bins / 3.0)
        selection_score = (
            max(self.selection.horizontal_span_ratio, 1e-6)
            * max(bin_score, 1e-6)
            * max(self.selection.mean_confidence, 1e-6)
        ) ** (1.0 / 3.0)
        return float(np.sqrt(self.geometry.geometric_confidence * selection_score))

    def to_dict(self) -> dict[str, Any]:
        if self.geometry is None:
            payload: dict[str, Any] = {
                "level_m": None,
                "uncertainty_m": None,
                "curve_points": self.selection.input_points,
                "matched_points": 0,
                "inlier_points": 0,
                "coverage": 0.0,
                "inlier_ratio": 0.0,
                "median_reprojection_px": None,
                "p95_reprojection_px": None,
            }
        else:
            payload = self.geometry.to_dict()
        payload.update(
            {
                "accepted": self.accepted,
                "geometric_confidence_uncalibrated": self.confidence,
                "rejection_reasons": list(self.rejection_reasons),
                "sparse_selection": self.selection.to_dict(),
            }
        )
        return payload


def _robust_horizontal_range(points: np.ndarray) -> tuple[float, float]:
    x = points[:, 0]
    if len(x) >= 20:
        low, high = np.percentile(x, (5.0, 95.0))
    else:
        low, high = x.min(), x.max()
    if high - low < 1.0:
        center = 0.5 * (low + high)
        return center - 0.5, center + 0.5
    return float(low), float(high)


def select_sparse_contact_points(
    contact_curve_pixels: np.ndarray,
    point_confidences: np.ndarray | None = None,
    *,
    min_confidence: float = 0.5,
    max_points: int = 24,
    horizontal_bins: int = 8,
) -> SparseContactSelection:
    """Keep reliable contact pixels while preserving horizontal observability."""
    curve = np.asarray(contact_curve_pixels, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2:
        raise ValueError("contact_curve_pixels must have shape (N, 2), N >= 2")
    if not np.all(np.isfinite(curve)):
        raise ValueError("contact_curve_pixels contains non-finite values")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be within [0, 1]")
    if max_points < 2:
        raise ValueError("max_points must be at least two")
    if horizontal_bins < 1:
        raise ValueError("horizontal_bins must be positive")

    if point_confidences is None:
        confidence = np.ones(len(curve), dtype=np.float64)
    else:
        confidence = np.asarray(point_confidences, dtype=np.float64).reshape(-1)
        if confidence.shape != (len(curve),):
            raise ValueError("point_confidences must contain one value per curve point")
        if not np.all(np.isfinite(confidence)):
            raise ValueError("point_confidences contains non-finite values")
        confidence = np.clip(confidence, 0.0, 1.0)

    reliable_indices = np.flatnonzero(confidence >= min_confidence)
    low, high = _robust_horizontal_range(curve)
    denominator = high - low
    if len(reliable_indices) == 0:
        return SparseContactSelection(
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            len(curve),
            0,
            0.0,
            0,
            horizontal_bins,
        )

    normalized_x = np.clip((curve[:, 0] - low) / denominator, 0.0, 1.0)
    bin_index = np.minimum(
        (normalized_x * horizontal_bins).astype(np.int64),
        horizontal_bins - 1,
    )
    quota = max(1, max_points // horizontal_bins)
    selected: list[int] = []
    for current_bin in range(horizontal_bins):
        candidates = reliable_indices[bin_index[reliable_indices] == current_bin]
        if len(candidates):
            ranked = candidates[np.argsort(-confidence[candidates], kind="stable")]
            selected.extend(ranked[:quota].tolist())

    if len(selected) < min(max_points, len(reliable_indices)):
        chosen = set(selected)
        remaining = [index for index in reliable_indices if int(index) not in chosen]
        remaining.sort(key=lambda index: (-confidence[index], int(index)))
        selected.extend(remaining[: max_points - len(selected)])

    selected_indices = np.asarray(selected[:max_points], dtype=np.int64)
    selected_indices.sort()
    selected_points = curve[selected_indices]
    selected_confidence = confidence[selected_indices]
    selected_normalized_x = normalized_x[selected_indices]
    span = (
        float(selected_normalized_x.max() - selected_normalized_x.min()) if len(selected_indices) > 1 else 0.0
    )
    occupied = len(np.unique(bin_index[selected_indices]))
    return SparseContactSelection(
        selected_points,
        selected_confidence,
        selected_indices,
        len(curve),
        len(reliable_indices),
        min(1.0, max(0.0, span)),
        occupied,
        horizontal_bins,
    )


def estimate_level_from_sparse_contact(
    model: ContainerModel,
    contact_curve_pixels: np.ndarray,
    camera_matrix: np.ndarray,
    rotation_m2c: np.ndarray,
    translation_m2c_m: np.ndarray,
    *,
    point_confidences: np.ndarray | None = None,
    min_point_confidence: float = 0.5,
    max_selected_points: int = 24,
    horizontal_bins: int = 8,
    min_reliable_points: int = 6,
    min_horizontal_span_ratio: float = 0.5,
    min_occupied_bins: int = 3,
    geometry_options: dict[str, Any] | None = None,
) -> SparseLevelEstimate:
    """Estimate level from a small reliable subset and reject weak geometry."""
    if min_reliable_points < 2:
        raise ValueError("min_reliable_points must be at least two")
    if not 0.0 <= min_horizontal_span_ratio <= 1.0:
        raise ValueError("min_horizontal_span_ratio must be within [0, 1]")
    if not 1 <= min_occupied_bins <= horizontal_bins:
        raise ValueError("min_occupied_bins must be within [1, horizontal_bins]")
    selection = select_sparse_contact_points(
        contact_curve_pixels,
        point_confidences,
        min_confidence=min_point_confidence,
        max_points=max_selected_points,
        horizontal_bins=horizontal_bins,
    )
    reasons: list[str] = []
    if selection.reliable_points < min_reliable_points:
        reasons.append("insufficient_reliable_contact_points")
    if len(selection.points_px) < min_reliable_points:
        reasons.append("insufficient_selected_contact_points")
    if (
        selection.horizontal_span_ratio < min_horizontal_span_ratio
        or selection.occupied_bins < min_occupied_bins
    ):
        reasons.append("insufficient_spatial_coverage")

    geometry = None
    if len(selection.points_px) >= 2:
        options = dict(geometry_options or {})
        options.setdefault("min_matches", min_reliable_points)
        geometry = estimate_level_from_contact_curve(
            model,
            selection.points_px,
            camera_matrix,
            rotation_m2c,
            translation_m2c_m,
            **options,
        )
    return SparseLevelEstimate(geometry, selection, tuple(reasons))


def analyze_contact_pixel_sensitivity(
    model: ContainerModel,
    contact_curve_pixels: np.ndarray,
    camera_matrix: np.ndarray,
    rotation_m2c: np.ndarray,
    translation_m2c_m: np.ndarray,
    *,
    point_confidences: np.ndarray | None = None,
    jitter_sigmas_px: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
    vertical_offsets_px: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0),
    trials: int = 100,
    seed: int = 2026,
    estimate_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Propagate random and systematic contact-pixel errors into metric level."""
    if trials < 1:
        raise ValueError("trials must be positive")
    curve = np.asarray(contact_curve_pixels, dtype=np.float64)
    options = dict(estimate_options or {})

    def estimate(points: np.ndarray) -> SparseLevelEstimate:
        return estimate_level_from_sparse_contact(
            model,
            points,
            camera_matrix,
            rotation_m2c,
            translation_m2c_m,
            point_confidences=point_confidences,
            **options,
        )

    baseline = estimate(curve)
    if not baseline.accepted or baseline.level_m is None:
        raise ValueError(
            "Baseline sparse contact estimate was rejected: " + ",".join(baseline.rejection_reasons)
        )
    reference = baseline.level_m
    rng = np.random.default_rng(seed)
    random_results = []
    for sigma in jitter_sigmas_px:
        if sigma < 0:
            raise ValueError("jitter sigmas must be non-negative")
        levels = []
        for _ in range(trials):
            perturbed = curve + rng.normal(0.0, sigma, size=curve.shape)
            result = estimate(perturbed)
            if result.accepted and result.level_m is not None:
                levels.append(result.level_m)
        errors = np.abs(np.asarray(levels) - reference)
        random_results.append(
            {
                "sigma_px": float(sigma),
                "trials": trials,
                "accepted_trials": len(levels),
                "accepted_ratio": len(levels) / trials,
                "mae_mm_from_baseline": float(errors.mean() * 1000.0) if len(errors) else None,
                "p95_mm_from_baseline": float(np.percentile(errors, 95) * 1000.0) if len(errors) else None,
            }
        )

    systematic_results = []
    accepted_offsets = []
    level_deltas = []
    for offset in vertical_offsets_px:
        perturbed = curve + np.array([0.0, float(offset)])
        result = estimate(perturbed)
        delta = (
            float((result.level_m - reference) * 1000.0)
            if result.accepted and result.level_m is not None
            else None
        )
        systematic_results.append(
            {
                "vertical_offset_px": float(offset),
                "accepted": result.accepted,
                "level_delta_mm": delta,
                "rejection_reasons": list(result.rejection_reasons),
            }
        )
        if delta is not None:
            accepted_offsets.append(float(offset))
            level_deltas.append(delta)
    sensitivity = (
        float(np.polyfit(accepted_offsets, level_deltas, 1)[0]) if len(accepted_offsets) >= 2 else None
    )
    return {
        "baseline": baseline.to_dict(),
        "random_jitter": random_results,
        "systematic_vertical_bias": systematic_results,
        "local_vertical_sensitivity_mm_per_px": sensitivity,
        "seed": seed,
    }
