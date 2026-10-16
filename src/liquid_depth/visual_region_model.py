"""Small RGB+installed-ROI semantic model. No true depth or labels at inference."""

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(4, cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(4, cout),
        nn.SiLU(),
    )


class VisualRegionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = block(4, 16)
        self.e2 = block(16, 32)
        self.e3 = block(32, 64)
        self.d2 = block(96, 32)
        self.d1 = block(48, 16)
        self.head = nn.Conv2d(16, 4, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.e3(F.avg_pool2d(b, 2))
        d = self.d2(
            torch.cat([F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1)
        )
        return self.head(
            self.d1(
                torch.cat([F.interpolate(d, size=a.shape[-2:], mode="bilinear", align_corners=False), a], 1)
            )
        )


class RegionPredictor:
    def __init__(self, checkpoint, device="cpu"):
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        self.device = device
        self.model = VisualRegionNet().to(device).eval()
        self.model.load_state_dict(state["model"])
        self.temperature = float(state.get("temperature", 1.0))

    @torch.inference_mode()
    def __call__(self, rgb_bgr, roi):
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.concatenate([rgb, np.asarray(roi, np.float32)[..., None]], axis=-1)
        logits = self.model(torch.from_numpy(x.transpose(2, 0, 1)[None]).to(self.device))
        return (logits[0] / self.temperature).softmax(0).cpu().numpy()
