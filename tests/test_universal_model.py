from __future__ import annotations

import math

import torch

from liquid_depth.models.universal import UniversalLiquidSurfaceNet, UniversalMultiTaskLoss


def test_universal_model_forward_and_loss():
    model = UniversalLiquidSurfaceNet(base_channels=4, min_depth_m=0.1, max_depth_m=10.0)
    inputs = torch.zeros(2, 5, 32, 48)
    output = model(inputs)
    assert output["confidence_logits"].shape == (2, 1, 32, 48)
    assert model.confidence_head is not None
    assert output["depth_m"].shape == (2, 1, 32, 48)
    assert torch.all(output["depth_m"] >= 0.1)
    assert torch.all(output["depth_m"] <= 10.0)
    target = {
        "mask": torch.ones(2, 1, 32, 48),
        "depth_m": torch.ones(2, 1, 32, 48),
        "normal": torch.nn.functional.normalize(torch.ones(2, 3, 32, 48), dim=1),
        "valid": torch.ones(2, 1, 32, 48),
        "normal_valid": torch.ones(2, 1, 32, 48),
    }
    losses = UniversalMultiTaskLoss()(output, target)
    assert torch.isfinite(losses["total"])
    assert losses["relative_log"] >= 0
    assert losses["surface_level"] >= 0
    assert losses["surface_absolute_m"] >= 0
    assert torch.isfinite(losses["uncertainty_calibration"])


def test_universal_model_keeps_legacy_confidence_contract():
    model = UniversalLiquidSurfaceNet(
        base_channels=4,
        separate_confidence_head=False,
    )
    output = model(torch.zeros(1, 5, 16, 16))
    torch.testing.assert_close(output["confidence"], torch.sigmoid(-output["log_variance"]))


def test_level_calibration_head_starts_as_identity_and_receives_gradients():
    torch.manual_seed(7)
    baseline = UniversalLiquidSurfaceNet(base_channels=4)
    calibrated = UniversalLiquidSurfaceNet(
        base_channels=4,
        level_calibration_enabled=True,
    )
    calibrated.load_state_dict(baseline.state_dict(), strict=False)
    inputs = torch.rand(2, 5, 32, 48)
    inputs[:, 4] = 1.0
    baseline_depth = baseline(inputs)["depth_m"]
    output = calibrated(inputs)
    torch.testing.assert_close(output["depth_m"], baseline_depth)
    torch.testing.assert_close(output["calibration_scale"], torch.ones(2, 1))
    torch.testing.assert_close(output["calibration_bias_m"], torch.zeros(2, 1))
    output["depth_m"].mean().backward()
    assert calibrated.level_calibration_head is not None
    assert calibrated.level_calibration_head[-1].weight.grad is not None


def test_surface_tail_losses_prioritize_ordinary_samples():
    model = UniversalLiquidSurfaceNet(
        base_channels=4,
        level_calibration_enabled=True,
    )
    inputs = torch.zeros(2, 5, 16, 16)
    output = model(inputs)
    target = {
        "mask": torch.ones(2, 1, 16, 16),
        "depth_m": torch.stack((torch.ones(1, 16, 16), torch.ones(1, 16, 16) * 2.0)),
        "normal": torch.nn.functional.normalize(torch.ones(2, 3, 16, 16), dim=1),
        "valid": torch.ones(2, 1, 16, 16),
        "normal_valid": torch.ones(2, 1, 16, 16),
        "ordinary": torch.tensor([1.0, 0.0]),
    }
    losses = UniversalMultiTaskLoss(
        surface_tolerance_weight=1.0,
        surface_quantile_weight=1.0,
        ordinary_loss_boost=2.0,
    )(output, target)
    assert losses["surface_tolerance"] >= 0
    assert losses["surface_quantile_cvar"] >= 0
    assert torch.isfinite(losses["total"])


def test_robust_depth_anchor_preserves_reliable_sensor_level():
    model = UniversalLiquidSurfaceNet(
        base_channels=4,
        robust_depth_anchor_enabled=True,
        robust_anchor_bias_limit_m=10.0,
    )
    with torch.no_grad():
        model.mask_head.weight.zero_()
        model.mask_head.bias.fill_(10.0)
    inputs = torch.zeros(2, 5, 32, 48)
    raw_depth_m = 1.5
    inputs[:, 3] = math.log(raw_depth_m / 0.1) / math.log(10.0 / 0.1)
    inputs[:, 4] = 1.0

    prediction = model(inputs)

    medians = prediction["depth_m"].flatten(1).median(dim=1).values
    assert torch.allclose(medians, torch.full_like(medians, raw_depth_m), atol=1e-5)
    assert torch.allclose(
        prediction["robust_anchor_support_ratio"],
        torch.ones_like(prediction["robust_anchor_support_ratio"]),
    )
