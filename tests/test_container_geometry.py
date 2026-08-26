import numpy as np

from liquid_depth.container_geometry import (
    estimate_level_from_contact_curve,
    project_model_points,
    sample_axisymmetric_container,
)


def _synthetic_case():
    model = sample_axisymmetric_container(
        np.array([0.0, 0.3]),
        np.array([0.05, 0.05]),
        vertical_samples=301,
        angular_samples=180,
    )
    camera_matrix = np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    )
    rotation = np.eye(3)
    translation = np.array([0.0, -0.15, 1.0])
    return model, camera_matrix, rotation, translation


def test_contact_curve_recovers_metric_container_level_with_outliers():
    model, camera_matrix, rotation, translation = _synthetic_case()
    projected, levels, _ = project_model_points(
        model,
        camera_matrix,
        rotation,
        translation,
    )
    curve = projected[np.isclose(levels, 0.18)][::3]
    rng = np.random.default_rng(7)
    curve = curve + rng.normal(0.0, 0.15, size=curve.shape)
    curve = np.vstack((curve, np.array([[20.0, 20.0], [1200.0, 700.0]])))

    result = estimate_level_from_contact_curve(
        model,
        curve,
        camera_matrix,
        rotation,
        translation,
    )

    assert result.accepted
    assert result.level_m is not None
    assert abs(result.level_m - 0.18) < 0.0015
    assert result.coverage > 0.9
    assert result.uncertainty_m is not None
    assert result.uncertainty_m < 0.003
    assert 0.5 < result.geometric_confidence <= 1.0


def test_contact_curve_rejects_pixels_without_model_support():
    model, camera_matrix, rotation, translation = _synthetic_case()
    curve = np.column_stack((np.linspace(20.0, 100.0, 20), np.full(20, 30.0)))

    result = estimate_level_from_contact_curve(
        model,
        curve,
        camera_matrix,
        rotation,
        translation,
    )

    assert not result.accepted
    assert result.level_m is None
    assert result.geometric_confidence == 0.0
    assert "insufficient_geometry_matches" in result.rejection_reasons
    assert "low_curve_coverage" in result.rejection_reasons


def test_contact_curve_rejects_two_incompatible_levels():
    model, camera_matrix, rotation, translation = _synthetic_case()
    projected, levels, _ = project_model_points(
        model,
        camera_matrix,
        rotation,
        translation,
    )
    low = projected[np.isclose(levels, 0.08)][::6]
    high = projected[np.isclose(levels, 0.24)][::6]
    curve = np.vstack((low, high))

    result = estimate_level_from_contact_curve(
        model,
        curve,
        camera_matrix,
        rotation,
        translation,
    )

    assert not result.accepted
    assert "inconsistent_or_multimodal_contact_height" in result.rejection_reasons
