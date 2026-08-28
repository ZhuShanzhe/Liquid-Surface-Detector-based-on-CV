import numpy as np

from liquid_depth.geometry import split_surface_mask
from liquid_depth.quality import assess_quality
from liquid_depth.sampling import balanced_sample_weights
from liquid_depth.segmentation import overlay_mask
from liquid_depth.temporal import RobustKalmanFilter


def test_quality_reports_explicit_rejection_reasons():
    result = assess_quality(
        {
            "inlier_ratio": 0.1,
            "median_residual_m": 0.01,
            "mask_area_px": 20000,
            "mean_segmentation_confidence": 0.9,
            "mean_depth_confidence": 0.1,
            "plane_angle_deg": 20.0,
        },
        {
            "min_inlier_ratio": 0.3,
            "max_median_residual_m": 0.006,
            "min_mask_area": 10000,
            "min_mean_segmentation_confidence": 0.2,
            "min_mean_depth_confidence": 0.2,
            "max_plane_angle_deg": 15.0,
        },
    )
    assert not result.accepted
    assert set(result.rejection_reasons) == {
        "low_plane_inlier_ratio",
        "high_plane_residual",
        "low_depth_confidence",
        "liquid_bottom_plane_not_parallel",
    }
    assert 0.0 <= result.confidence <= 1.0


def test_quality_rejects_unusable_illumination_when_thresholds_are_configured():
    metrics = {
        "inlier_ratio": 0.8,
        "median_residual_m": 0.001,
        "mask_area_px": 20000,
        "mean_segmentation_confidence": 0.9,
        "mean_depth_confidence": 0.9,
        "plane_angle_deg": 1.0,
        "luma_p50": 0.02,
        "dark_pixel_ratio": 0.95,
        "saturated_pixel_ratio": 0.0,
        "dynamic_range": 0.01,
    }
    result = assess_quality(
        metrics,
        {
            "min_luma_p50": 0.05,
            "max_dark_pixel_ratio": 0.9,
            "max_saturated_pixel_ratio": 0.35,
            "min_dynamic_range": 0.03,
        },
    )
    assert not result.accepted
    assert set(result.rejection_reasons) == {
        "scene_too_dark",
        "excessive_dark_pixels",
        "insufficient_image_dynamic_range",
    }


def test_temporal_filter_rejects_large_jump_without_moving_state():
    tracker = RobustKalmanFilter(max_jump=1.0, measurement_variance=0.1)
    first = tracker.update(10.0, 0.9)
    second = tracker.update(10.2, 0.9)
    rejected = tracker.update(14.0, 0.9)
    assert first.accepted and second.accepted
    assert not rejected.accepted
    assert rejected.reason == "temporal_innovation_too_large"
    assert rejected.recovered
    assert rejected.hold_frames == 1
    assert rejected.value == second.value


def test_surface_mask_separates_interior_and_meniscus():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:56, 8:56] = 255
    interior, meniscus = split_surface_mask(mask, interior_erode_px=4, meniscus_width_px=4)
    assert np.count_nonzero(interior) < np.count_nonzero(mask)
    assert np.count_nonzero(meniscus) > 0
    assert not np.any((interior > 0) & (meniscus > 0))


def test_balanced_sampling_upweights_rare_difficult_cases():
    rows = [{"difficulty_tags": "ordinary"} for _ in range(9)]
    rows.append({"difficulty_tags": "transparent,glare"})
    weights = balanced_sample_weights(rows)
    assert weights[-1] > weights[0]
    assert np.isclose(np.mean(weights), 1.0)


def test_empty_mask_overlay_preserves_rgb():
    rgb = np.full((12, 16, 3), 80, dtype=np.uint8)
    mask = np.zeros((12, 16), dtype=np.uint8)
    np.testing.assert_array_equal(overlay_mask(rgb, mask), rgb)
