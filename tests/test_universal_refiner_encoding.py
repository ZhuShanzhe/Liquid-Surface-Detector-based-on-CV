from __future__ import annotations

import numpy as np
import torch

from liquid_depth.refinement import TorchScriptDepthRefiner


class _EchoEncodedDepth(torch.nn.Module):
    def forward(self, inputs):
        depth = inputs[:, 3:4]
        return {"depth_m": depth, "confidence": torch.ones_like(depth)}


def test_torchscript_refiner_applies_universal_log_depth_encoding(tmp_path):
    example = torch.zeros(1, 5, 16, 16)
    model = torch.jit.trace(_EchoEncodedDepth(), example, strict=False)
    path = tmp_path / "echo_log_depth.ts"
    torch.jit.save(model, path)
    refiner = TorchScriptDepthRefiner(path, [16, 16], 10.0, False, "log", 0.1)
    rgb = np.zeros((12, 20, 3), dtype=np.uint8)
    raw_depth = np.full((12, 20), 1000, dtype=np.uint16)
    result = refiner.predict(rgb, raw_depth)
    np.testing.assert_allclose(result.depth_m, 0.5, atol=1e-6)
    np.testing.assert_allclose(result.confidence, 1.0)
