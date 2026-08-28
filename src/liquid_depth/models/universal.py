from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .multitask import ConvBlock, MultiTaskLoss


class UniversalLiquidSurfaceNet(nn.Module):
    """0.1-10 m RGB-D model with log-uniform metric depth parameterization.

    Module names intentionally match LiquidSurfaceMultiTaskNet so existing v2
    weights can be used as an optional initialization while the metric output
    contract remains a separate v3 checkpoint.
    """

    def __init__(
        self,
        base_channels: int = 32,
        min_depth_m: float = 0.1,
        max_depth_m: float = 10.0,
    ) -> None:
        super().__init__()
        if not 0 < min_depth_m < max_depth_m:
            raise ValueError("Expected 0 < min_depth_m < max_depth_m")
        c = int(base_channels)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.log_depth_span = float(math.log(max_depth_m / min_depth_m))
        self.enc1 = ConvBlock(5, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.bottleneck = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = ConvBlock(c * 8 + c * 4, c * 4)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = ConvBlock(c * 2 + c, c)
        self.mask_head = nn.Conv2d(c, 1, 1)
        self.depth_head = nn.Conv2d(c, 1, 1)
        self.normal_head = nn.Conv2d(c, 3, 1)
        self.log_variance_head = nn.Conv2d(c, 1, 1)

    @staticmethod
    def _up(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != 5:
            raise ValueError("Expected BCHW normalized RGB, log-depth, validity")
        e1 = self.enc1(inputs)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        features = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat((self._up(features, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self._up(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self._up(d2, e1), e1), dim=1))
        depth_unit = torch.sigmoid(self.depth_head(d1))
        depth_m = self.min_depth_m * torch.exp(depth_unit * self.log_depth_span)
        log_variance = self.log_variance_head(d1).clamp(-7.0, 5.0)
        return {
            "mask_logits": self.mask_head(d1),
            "depth_m": depth_m,
            "normal": F.normalize(self.normal_head(d1), dim=1, eps=1e-6),
            "log_variance": log_variance,
            "confidence": torch.sigmoid(-log_variance),
        }


class UniversalMultiTaskLoss(nn.Module):
    def __init__(
        self,
        *,
        relative_weight: float = 1.0,
        tolerance_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.base = MultiTaskLoss(tolerance_weight=tolerance_weight)
        self.relative_weight = float(relative_weight)

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        losses = self.base(prediction, target)
        valid = target["valid"].float()
        log_error = (
            torch.log(prediction["depth_m"].clamp_min(1e-4))
            - torch.log(target["depth_m"].clamp_min(1e-4))
        ).abs()
        relative = (log_error * valid).sum() / valid.sum().clamp_min(1.0)
        losses["relative_log"] = relative
        losses["total"] = losses["total"] + self.relative_weight * relative
        return losses
