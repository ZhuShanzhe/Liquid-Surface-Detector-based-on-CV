"""Partial-rim calibration uses visible clicks, not off-image corners."""

import numpy as np
import pytest

from liquid_depth.partial_rim_geometry import complete_rectangular_rim


def test_complete_cropped_rectangle_from_visible_edges():
    shape = (180, 320)
    k = np.array([[200.0, 0.0, 159.5], [0.0, 200.0, 89.5], [0.0, 0.0, 1.0]])
    up = np.array([0.0, 0.0, -1.0])
    dimensions = [0.8, 0.5]
    xy = np.array([[-0.4, 0.0], [0.4, 0.0], [0.4, 0.5], [-0.4, 0.5]])
    corners = np.column_stack((159.5 + 200 * xy[:, 0],
                               89.5 - 200 * xy[:, 1]))
    assert np.sum(corners[:, 1] < 0) == 2
    rng = np.random.default_rng(41)
    edges = []
    for label in range(4):
        t = np.linspace(0, 1, 65)[:, None]
        pixels = corners[label] * (1 - t) + corners[(label + 1) % 4] * t
        visible = pixels[(pixels[:, 0] >= 1.5) & (pixels[:, 0] < 318.5)
                         & (pixels[:, 1] >= 1.5) & (pixels[:, 1] < 178.5)][::4]
        if len(visible) >= 2:
            edges.append({"edge": label,
                          "pixels": (visible + rng.normal(0, 0.05, visible.shape)).tolist()})
    fitted, quality = complete_rectangular_rim(
        shape, k, {"measured_rim_dimensions_m": dimensions,
                   "visible_rim_edges_px": edges}, up, 1.0
    )
    assert quality["visible_rim_edges"] == 3
    assert quality["extrapolated_rim_corners"] == 2
    assert quality["partial_rim_edge_residual_median_m"] < 0.002
    assert np.max(np.linalg.norm(fitted - corners, axis=1)) < 2.0


def test_partial_rim_rejects_insufficient_visible_edges():
    with pytest.raises(ValueError, match="three_visible_edges"):
        complete_rectangular_rim(
            (180, 320), np.eye(3),
            {"measured_rim_dimensions_m": [0.8, 0.5],
             "visible_rim_edges_px": []},
            np.array([0.0, 0.0, -1.0]), 1.0
        )
