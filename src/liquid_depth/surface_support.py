from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PlanarSupportAssessment:
    """Spatial observability of a robust plane in a partly occluded surface."""

    accepted: bool
    state: str
    occupied_tiles: int
    surface_tiles: int
    tile_coverage: float
    horizontal_span_ratio: float
    vertical_span_ratio: float
    convex_hull_coverage_ratio: float
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "state": self.state,
            "occupied_tiles": self.occupied_tiles,
            "surface_tiles": self.surface_tiles,
            "tile_coverage": self.tile_coverage,
            "horizontal_span_ratio": self.horizontal_span_ratio,
            "vertical_span_ratio": self.vertical_span_ratio,
            "convex_hull_coverage_ratio": self.convex_hull_coverage_ratio,
            "rejection_reasons": list(self.rejection_reasons),
        }


def assess_planar_support(
    surface_mask: np.ndarray,
    inlier_pixels: np.ndarray,
    *,
    fit_inlier_ratio: float,
    horizontal_tiles: int = 6,
    vertical_tiles: int = 4,
    min_points_per_tile: int = 3,
    min_tile_coverage: float = 0.30,
    min_horizontal_span_ratio: float = 0.45,
    min_vertical_span_ratio: float = 0.25,
    min_convex_hull_coverage_ratio: float = 0.12,
    min_fit_inlier_ratio: float = 0.25,
) -> PlanarSupportAssessment:
    """Accept a distributed partial plane and reject a compact accidental patch.

    Floating material, foam, glare, and missing depth may remove much of the liquid
    surface. RANSAC can still estimate the representative level from the remaining
    points, but only when those points cover multiple parts of the surface.
    """

    mask = np.asarray(surface_mask) > 0
    pixels = np.asarray(inlier_pixels, dtype=np.int64)
    if mask.ndim != 2:
        raise ValueError("surface_mask must be two-dimensional")
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("inlier_pixels must have shape (N, 2) in (u, v) order")
    if horizontal_tiles < 1 or vertical_tiles < 1 or min_points_per_tile < 1:
        raise ValueError("tile counts and min_points_per_tile must be positive")
    locations = np.argwhere(mask)
    if len(locations) == 0:
        return PlanarSupportAssessment(
            False,
            "missing_surface",
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            ("surface_mask_empty",),
        )

    y0, x0 = locations.min(axis=0)
    y1, x1 = locations.max(axis=0) + 1
    width, height = max(x1 - x0, 1), max(y1 - y0, 1)
    valid = (pixels[:, 0] >= x0) & (pixels[:, 0] < x1) & (pixels[:, 1] >= y0) & (pixels[:, 1] < y1)
    pixels = pixels[valid]
    surface_tile_mask = np.zeros((vertical_tiles, horizontal_tiles), dtype=bool)
    inlier_counts = np.zeros_like(surface_tile_mask, dtype=np.int64)

    mask_y, mask_x = np.where(mask)
    mask_tx = np.minimum((mask_x - x0) * horizontal_tiles // width, horizontal_tiles - 1)
    mask_ty = np.minimum((mask_y - y0) * vertical_tiles // height, vertical_tiles - 1)
    surface_tile_mask[mask_ty, mask_tx] = True
    if len(pixels):
        tx = np.minimum((pixels[:, 0] - x0) * horizontal_tiles // width, horizontal_tiles - 1)
        ty = np.minimum((pixels[:, 1] - y0) * vertical_tiles // height, vertical_tiles - 1)
        np.add.at(inlier_counts, (ty, tx), 1)
        horizontal_span = float((pixels[:, 0].max() - pixels[:, 0].min()) / width)
        vertical_span = float((pixels[:, 1].max() - pixels[:, 1].min()) / height)
    else:
        horizontal_span = vertical_span = 0.0
    occupied = (inlier_counts >= min_points_per_tile) & surface_tile_mask
    surface_tiles = int(surface_tile_mask.sum())
    occupied_tiles = int(occupied.sum())
    tile_coverage = occupied_tiles / max(surface_tiles, 1)
    hull_coverage = 0.0
    if len(pixels) >= 3:
        import cv2

        hull = cv2.convexHull(
            pixels.astype(np.float32).reshape(-1, 1, 2)
        )
        hull_area = float(cv2.contourArea(hull))
        hull_coverage = hull_area / max(float(mask.sum()), 1.0)
    hull_coverage = float(np.clip(hull_coverage, 0.0, 1.0))

    reasons: list[str] = []
    if fit_inlier_ratio < min_fit_inlier_ratio:
        reasons.append("insufficient_planar_consensus")
    if tile_coverage < min_tile_coverage:
        reasons.append("insufficient_planar_tile_coverage")
    if horizontal_span < min_horizontal_span_ratio:
        reasons.append("insufficient_planar_horizontal_span")
    if vertical_span < min_vertical_span_ratio:
        reasons.append("insufficient_planar_vertical_span")
    if hull_coverage < min_convex_hull_coverage_ratio:
        reasons.append("insufficient_planar_convex_hull_coverage")
    accepted = not reasons
    state = "stable_planar"
    if accepted and (fit_inlier_ratio < 0.6 or tile_coverage < 0.6):
        state = "partial_planar"
    elif not accepted:
        state = "nonplanar_or_occluded"
    return PlanarSupportAssessment(
        accepted,
        state,
        occupied_tiles,
        surface_tiles,
        float(tile_coverage),
        float(np.clip(horizontal_span, 0.0, 1.0)),
        float(np.clip(vertical_span, 0.0, 1.0)),
        hull_coverage,
        tuple(reasons),
    )
