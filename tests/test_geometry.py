import numpy as np

from liquid_depth.geometry import fit_plane, masked_points


def test_masked_points_projects_camera_coordinates():
    depth_mm = np.full((3, 4), 1000, dtype=np.uint16)
    mask = np.zeros_like(depth_mm, dtype=np.uint8)
    mask[1, 2] = 255
    matrix = np.array([[2.0, 0, 1.0], [0, 2.0, 1.0], [0, 0, 1.0]])
    points, pixels = masked_points(depth_mm, mask, matrix, percentiles=(0, 100))
    np.testing.assert_allclose(points, [[0.5, 0.0, 1.0]])
    np.testing.assert_array_equal(pixels, [[2, 1]])


def test_ransac_recovers_plane_with_outliers():
    rng = np.random.default_rng(5)
    xy = rng.uniform(-0.3, 0.3, size=(1000, 2))
    z = 1.2 + 0.1 * xy[:, 0] - 0.05 * xy[:, 1] + rng.normal(0, 0.0005, 1000)
    points = np.column_stack((xy, z))
    points[:100] = rng.uniform(-1, 1, size=(100, 3))
    pixels = np.column_stack((np.arange(len(points)), np.zeros(len(points)))).astype(np.int32)
    fit = fit_plane(points, pixels, threshold_m=0.003, iterations=500, seed=11)
    assert fit.inlier_ratio > 0.75
    assert fit.median_residual_m < 0.001
    expected = np.array([0.1, -0.05, -1.0])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(fit.plane.normal, expected, atol=0.005)

