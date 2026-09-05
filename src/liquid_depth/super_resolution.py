"""Optional deterministic SR adapter. Upsampled pixels are not new measurements."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def scaled_intrinsics(matrix, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Positive finite scale required")
    k = np.asarray(matrix, float).copy()
    k[:2, :2] *= scale
    k[:2, 2] = (k[:2, 2] + 0.5) * scale - 0.5
    return k


class SwinIRX4:
    def __init__(self, repository, checkpoint, device=None):
        path = Path(repository) / "models/network_swinir.py"
        spec = importlib.util.spec_from_file_location("liquid_external_swinir", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = (
            module.SwinIR(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6],
                embed_dim=60,
                num_heads=[6, 6, 6, 6],
                mlp_ratio=2,
                upsampler="pixelshuffledirect",
                resi_connection="1conv",
            )
            .to(self.device)
            .eval()
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["params"], strict=True)

    def upscale(self, rgb_bgr):
        started = perf_counter()
        h, w = rgb_bgr.shape[:2]
        image = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(image.transpose(2, 0, 1)[None]).to(self.device)
        x = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode="reflect")
        with torch.inference_mode():
            y = self.model(x)[0, :, : h * 4, : w * 4].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        output = cv2.cvtColor(np.rint(y * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return {
            "rgb_bgr": output,
            "scale": 4,
            "source_pixel_scale": 4,
            "adds_independent_measurement": False,
            "latency_ms": (perf_counter() - started) * 1000,
        }
