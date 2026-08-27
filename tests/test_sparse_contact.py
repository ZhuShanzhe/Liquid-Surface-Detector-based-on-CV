import numpy as np

from liquid_depth.container_geometry import (
    project_model_points,
    sample_axisymmetric_container,
)
from liquid_depth.sparse_contact import (
    analyze_contact_pixel_sensitivity,
    estimate_level_from_sparse_contact,
    select_sparse_contact_points,
)


def _synthetic_case():
    model = sample_axisymmetric_container(
        np.array([0.0, 0.3]),
        np.array([0.05, 0.05]),
        vertical_samples=301,
        angular_samples=180,
    )
    camera_matrix = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    rotation = np.eye(3)
    translation = np.array([0.0, -0.15, 1.0])
    projected, levels, _ = project_model_points(
        model,
        camera_matrix,
        rotation,
        translation,
    )
    curve = projected[np.isclose(levels, 0.18)][::3]
    return model, camera_matrix, rotation, translation, curve


def test_sparse_selection_preserves_horizontal_observability():
    x = np.linspace(0.0, 100.0, 101)
    curve = np.column_stack((x, 20.0 + 0.01 * x))
    confidence = np.linspace(0.7, 1.0, len(curve))

    selection = select_sparse_contact_points(
        curve,
        confidence,
        min_confidence=0.7,
        max_points=12,
        horizontal_bins=6,
    )

    assert len(selection.points_px) == 12
    assert selection.occupied_bins == 6
    assert selection.horizontal_span_ratio > 0.8
    assert selection.source_indices.tolist() == sorted(selection.source_indices.tolist())


def test_sparse_estimator_rejects_confident_points_clustered_in_one_region():
    model, camera_matrix, rotation, translation, curve = _synthetic_case()
    confidence = np.full(len(curve), 0.1)
    center = np.median(curve[:, 0])
    nearest = np.argsort(np.abs(curve[:, 0] - center))[:10]
    confidence[nearest] = 0.95

    result = estimate_level_from_sparse_contact(
        model,
        curve,
        camera_matrix,
        rotation,
        translation,
        point_confidences=confidence,
        min_point_confidence=0.8,
        min_reliable_points=6,
        min_horizontal_span_ratio=0.5,
        min_occupied_bins=3,
    )

    assert not result.accepted
    assert "insufficient_spatial_coverage" in result.rejection_reasons
    assert result.selection.reliable_points == 10


def test_pixel_sensitivity_reports_metric_error_and_vertical_jacobian():
    model, camera_matrix, rotation, translation, curve = _synthetic_case()

    report = analyze_contact_pixel_sensitivity(
        model,
        curve,
        camera_matrix,
        rotation,
        translation,
        jitter_sigmas_px=(0.5, 1.0),
        vertical_offsets_px=(-1.0, 1.0),
        trials=20,
        seed=7,
        estimate_options={
            "min_point_confidence": 0.0,
            "max_selected_points": 24,
        },
    )

    assert report["baseline"]["accepted"]
    assert len(report["baseline"]["sparse_selection"]["source_indices"]) <= 24
    assert all(item["accepted_ratio"] > 0.8 for item in report["random_jitter"])
    sensitivity = report["local_vertical_sensitivity_mm_per_px"]
    assert sensitivity is not None
    assert 1.2 < sensitivity < 2.2
