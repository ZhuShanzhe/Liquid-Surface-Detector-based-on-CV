from __future__ import annotations

import torch

from liquid_depth.models.universal import UniversalLiquidSurfaceNet, UniversalMultiTaskLoss


def test_universal_model_forward_and_loss():
    model = UniversalLiquidSurfaceNet(base_channels=4, min_depth_m=0.1, max_depth_m=10.0)
    inputs = torch.zeros(2, 5, 32, 48)
    output = model(inputs)
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
