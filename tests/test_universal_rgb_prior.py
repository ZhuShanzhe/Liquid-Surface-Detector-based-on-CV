from __future__ import annotations

import torch

from liquid_depth.models.universal import UniversalLiquidSurfaceNet


def test_rgb_prior_supports_missing_depth_without_changing_contract():
    torch.manual_seed(4)
    model = UniversalLiquidSurfaceNet(base_channels=4, rgb_prior_enabled=True)
    inputs = torch.zeros(2, 5, 32, 48)
    inputs[:, :3] = torch.rand(2, 3, 32, 48)
    missing = model(inputs)
    inputs[:, 4] = 1.0
    available = model(inputs)
    assert missing["depth_m"].shape == (2, 1, 32, 48)
    assert missing["confidence"].shape == (2, 1, 32, 48)
    assert not torch.equal(missing["depth_m"], available["depth_m"])
    assert model.rgb_prior_scale is not None
    assert torch.isfinite(missing["depth_m"]).all()
