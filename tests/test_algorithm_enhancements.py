import cv2
import numpy as np
import torch

from liquid_depth.illumination import (
    adaptive_exposure_correction,
    measure_illumination,
)
from liquid_depth.models.layered import (
    PermutationInvariantLayerLoss,
    RayLayerHead,
    select_layer_by_metric_prior,
    select_liquid_interface,
)
from liquid_depth.surface_support import assess_planar_support


def test_low_light_correction_is_conditional_and_bounded():
    dark = np.full((120, 160, 3), 12, dtype=np.uint8)
    result = adaptive_exposure_correction(
        dark,
        {
            "enabled": True,
            "trigger_luma_p50": 0.2,
            "target_luma_p50": 0.4,
            "min_gamma": 0.45,
        },
    )
    assert result.applied
    assert 0.45 <= result.gamma <= 1.0
    assert result.after.luma_p50 > result.before.luma_p50
    assert result.image_bgr.dtype == np.uint8

    normal = np.full((120, 160, 3), 140, dtype=np.uint8)
    skipped = adaptive_exposure_correction(normal, {"enabled": True})
    assert not skipped.applied
    np.testing.assert_array_equal(skipped.image_bgr, normal)


def test_illumination_metrics_detect_dark_and_saturated_frames():
    dark = measure_illumination(np.zeros((40, 60, 3), dtype=np.uint8))
    bright = measure_illumination(np.full((40, 60, 3), 255, dtype=np.uint8))
    assert dark.dark_pixel_ratio == 1.0
    assert bright.saturated_pixel_ratio == 1.0


def test_partial_planar_support_accepts_distributed_points_and_rejects_cluster():
    mask = np.zeros((100, 160), dtype=np.uint8)
    cv2.ellipse(mask, (80, 50), (65, 35), 0, 0, 360, 255, -1)
    tile_centers = np.array(
        [(x, y) for y in (30, 45, 60, 72) for x in (28, 48, 68, 88, 108, 128)],
        dtype=np.int64,
    )
    distributed = np.repeat(tile_centers, 3, axis=0)
    accepted = assess_planar_support(mask, distributed, fit_inlier_ratio=0.55)
    assert accepted.accepted
    assert accepted.state in {"stable_planar", "partial_planar"}
    assert accepted.convex_hull_coverage_ratio >= 0.12

    clustered = np.column_stack((np.arange(70, 82), np.full(12, 50)))
    rejected = assess_planar_support(mask, clustered, fit_inlier_ratio=0.8)
    assert not rejected.accepted
    assert "insufficient_planar_horizontal_span" in rejected.rejection_reasons
    assert "insufficient_planar_convex_hull_coverage" in rejected.rejection_reasons


def test_layer_head_and_set_likelihood_are_differentiable_and_permutation_invariant():
    torch.manual_seed(4)
    head = RayLayerHead(8, num_layers=4, max_depth_m=5.0, hidden_channels=8)
    features = torch.randn(2, 8, 6, 8, requires_grad=True)
    prediction = head(features)
    assert prediction["layer_depths_m"].shape == (2, 4, 6, 8)
    assert torch.all(
        prediction["layer_depths_sorted_m"][:, 1:] >= prediction["layer_depths_sorted_m"][:, :-1]
    )

    target = torch.stack(
        (
            torch.full((2, 6, 8), 1.0),
            torch.full((2, 6, 8), 2.0),
            torch.full((2, 6, 8), 3.0),
        ),
        dim=1,
    )
    valid = torch.ones_like(target)
    criterion = PermutationInvariantLayerLoss()
    first = criterion(prediction, target, valid)
    second = criterion(prediction, target[:, [2, 0, 1]], valid[:, [2, 0, 1]])
    torch.testing.assert_close(first["set_intensity"], second["set_intensity"])
    torch.testing.assert_close(first["total"], second["total"])
    first["total"].backward()
    assert torch.isfinite(features.grad).all()


def test_metric_prior_selects_supported_layer_and_rejects_large_deviation():
    layers = torch.tensor([[[[0.4]], [[0.9]], [[1.5]], [[2.2]]]])
    confidence = torch.tensor([[[[0.8]], [[0.7]], [[0.9]], [[0.6]]]])
    selected = select_layer_by_metric_prior(
        layers,
        confidence,
        torch.tensor([[[0.95]]]),
        maximum_deviation_m=0.2,
    )
    assert selected["layer_index"].item() == 1
    assert selected["accepted"].item()
    rejected = select_layer_by_metric_prior(
        layers,
        confidence,
        torch.tensor([[[4.0]]]),
        maximum_deviation_m=0.2,
    )
    assert not rejected["accepted"].item()


def test_liquid_interface_reports_explicit_rejection_codes():
    prediction = {
        "layer_depths_m": torch.tensor([[[[1.0]], [[2.0]]]]),
        "liquid_interface_probability": torch.tensor([[[[0.9]], [[0.1]]]]),
        "layer_confidence": torch.tensor([[[[0.81]], [[0.81]]]]),
    }
    accepted = select_liquid_interface(
        prediction,
        torch.tensor([[[[1.0]]]]),
        confidence_threshold=0.5,
    )
    assert accepted["accepted"].item()
    assert accepted["rejection_code"].item() == 0

    low_confidence = select_liquid_interface(
        prediction,
        torch.tensor([[[[1.0]]]]),
        confidence_threshold=0.95,
    )
    assert not low_confidence["accepted"].item()
    assert low_confidence["rejection_code"].item() == 2

    disagreement = select_liquid_interface(
        prediction,
        torch.tensor([[[[5.0]]]]),
        confidence_threshold=0.0,
    )
    assert not disagreement["accepted"].item()
    assert disagreement["rejection_code"].item() == 1
