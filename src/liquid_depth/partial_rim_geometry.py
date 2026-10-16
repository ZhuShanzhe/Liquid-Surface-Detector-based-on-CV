"""Recover a rectangular rim from visible edge clicks and independent metrology.

The four corners may lie outside the image. Each edge is labelled by the operator;
only points inside the dry image are accepted. The fitted rectangle can be
extrapolated, so the result remains research-only and carries residual diagnostics.
"""

import numpy as np
from scipy.optimize import least_squares

from .empty_geometry_v41 import _intersect, _pixels, _project


def complete_rectangular_rim(raw_shape, intrinsics, spec, up, rim_distance):
    measured = np.asarray(spec.get("measured_rim_dimensions_m", []), dtype=float)
    if measured.shape != (2,) or not np.isfinite(measured).all() or min(measured) <= 0:
        raise ValueError("two_independently_measured_rim_dimensions_required")
    edges = spec.get("visible_rim_edges_px", [])
    if len(edges) < 3:
        raise ValueError("partial_rim_requires_three_visible_edges")
    labels = []
    observations = []
    for edge in edges:
        label = edge.get("edge")
        if label not in (0, 1, 2, 3):
            raise ValueError("invalid_visible_rim_edge_label")
        clicks = _pixels(edge.get("pixels", []), raw_shape, 2)
        labels.extend([label] * len(clicks))
        observations.extend(clicks)
    if len(set(labels)) < 3 or len(labels) < 12:
        raise ValueError("partial_rim_support_insufficient")
    labels = np.asarray(labels, dtype=int)
    points = _intersect(np.asarray(observations), intrinsics, up, rim_distance)
    xaxis = np.array([1.0, 0.0, 0.0])
    xaxis -= up * np.dot(xaxis, up)
    xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(up, xaxis)
    closest_rim = -rim_distance * up
    xy = np.column_stack(((points - closest_rim) @ xaxis,
                          (points - closest_rim) @ yaxis))
    half_width, half_length = measured / 2

    def residual(params):
        cx, cy, angle = params
        ex = np.array([np.cos(angle), np.sin(angle)])
        ey = np.array([-np.sin(angle), np.cos(angle)])
        relative = xy - np.array([cx, cy])
        signed = np.where(
            labels == 0, -(relative @ ey) - half_length,
            np.where(labels == 1, (relative @ ex) - half_width,
                     np.where(labels == 2, (relative @ ey) - half_length,
                              -(relative @ ex) - half_width)),
        )
        return signed

    solutions = []
    for angle in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        fit = least_squares(residual, [*xy.mean(axis=0), angle],
                            loss="soft_l1", f_scale=0.002, max_nfev=120)
        solutions.append((float(np.median(np.abs(residual(fit.x)))), fit.x))
    median_error, (cx, cy, angle) = min(solutions, key=lambda item: item[0])
    if median_error > max(0.006, 0.025 * min(measured)):
        raise ValueError("partial_rim_edge_fit_inconsistent")
    ex = np.array([np.cos(angle), np.sin(angle)])
    ey = np.array([-np.sin(angle), np.cos(angle)])
    local = np.array([[-half_width, -half_length],
                      [half_width, -half_length],
                      [half_width, half_length],
                      [-half_width, half_length]])
    corners_xy = np.array([cx, cy]) + local[:, :1] * ex + local[:, 1:] * ey
    corners_camera = (closest_rim + corners_xy[:, :1] * xaxis
                      + corners_xy[:, 1:] * yaxis)
    corners_px = np.asarray(_project(corners_camera, intrinsics))
    return corners_px, {
        "visible_rim_edges": len(set(labels)),
        "visible_rim_edge_points": len(labels),
        "partial_rim_edge_residual_median_m": median_error,
        "extrapolated_rim_corners": int(
            np.sum((corners_px[:, 0] < 0) | (corners_px[:, 0] >= raw_shape[1])
                   | (corners_px[:, 1] < 0) | (corners_px[:, 1] >= raw_shape[0]))),
    }
