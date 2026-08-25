import cv2
import numpy as np
import torch

from liquid_depth.models import LiquidSurfaceMultiTaskNet, MultiTaskLoss
from liquid_depth.refinement import TorchScriptDepthRefiner


def test_multitask_network_contract_and_loss():
    model = LiquidSurfaceMultiTaskNet(base_channels=8, max_depth_m=3.0)
    inputs = torch.randn(2, 5, 32, 32)
    prediction = model(inputs)
    assert prediction["mask_logits"].shape == (2, 1, 32, 32)
    assert prediction["depth_m"].shape == (2, 1, 32, 32)
    assert prediction["normal"].shape == (2, 3, 32, 32)
    assert prediction["confidence"].shape == (2, 1, 32, 32)
    assert torch.all((prediction["depth_m"] >= 0.0) & (prediction["depth_m"] <= 3.0))

    target = {
        "mask": torch.ones(2, 1, 32, 32),
        "depth_m": torch.ones(2, 1, 32, 32),
        "normal": torch.nn.functional.normalize(torch.randn(2, 3, 32, 32), dim=1),
        "valid": torch.ones(2, 1, 32, 32),
        "expected_plane_normal": torch.tensor([0.0, 0.0, -1.0]).view(1, 3, 1, 1),
    }
    losses = MultiTaskLoss()(prediction, target)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()


def test_multitask_torchscript_uses_metric_refiner_contract(tmp_path):
    model = LiquidSurfaceMultiTaskNet(base_channels=8, max_depth_m=3.0).eval()
    example = torch.zeros(1, 5, 32, 32)
    traced = torch.jit.trace(model, example, strict=False)
    path = tmp_path / "multitask.ts"
    torch.jit.save(traced, path)
    refiner = TorchScriptDepthRefiner(path, [32, 32], 3.0)
    rgb = np.zeros((24, 40, 3), dtype=np.uint8)
    raw_depth = np.full((24, 40), 1000, dtype=np.uint16)
    result = refiner.predict(rgb, raw_depth)
    assert result.backend == "torchscript_multitask"
    assert result.depth_m.shape == raw_depth.shape
    assert result.confidence.shape == raw_depth.shape
    assert np.all(np.isfinite(result.depth_m))
    assert cv2.countNonZero((result.confidence > 0).astype(np.uint8)) > 0
