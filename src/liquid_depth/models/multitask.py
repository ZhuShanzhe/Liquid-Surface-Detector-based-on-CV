from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = next(
            value
            for value in range(min(8, output_channels), 0, -1)
            if output_channels % value == 0
        )
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class LiquidSurfaceMultiTaskNet(nn.Module):
    """Five-channel RGB-D network for mask, metric depth, normal, and uncertainty."""

    def __init__(self, base_channels: int = 32, max_depth_m: float = 3.0) -> None:
        super().__init__()
        c = base_channels
        self.max_depth_m = float(max_depth_m)
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
        return F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != 5:
            raise ValueError("Expected BCHW input with five channels: normalized RGB, depth, validity")
        e1 = self.enc1(inputs)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        features = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat((self._up(features, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self._up(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self._up(d2, e1), e1), dim=1))
        log_variance = self.log_variance_head(d1).clamp(-7.0, 5.0)
        return {
            "mask_logits": self.mask_head(d1),
            "depth_m": torch.sigmoid(self.depth_head(d1)) * self.max_depth_m,
            "normal": F.normalize(self.normal_head(d1), dim=1, eps=1e-6),
            "log_variance": log_variance,
            "confidence": torch.sigmoid(-log_variance),
        }


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        mask_weight: float = 1.0,
        depth_weight: float = 2.0,
        normal_weight: float = 0.5,
        gradient_weight: float = 0.25,
        physics_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.weights = (mask_weight, depth_weight, normal_weight, gradient_weight, physics_weight)

    @staticmethod
    def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return (values * valid).sum() / valid.sum().clamp_min(1.0)

    def forward(
        self, prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        mask = target["mask"].float()
        valid = target.get("valid", torch.isfinite(target["depth_m"])).float() * mask
        mask_bce = F.binary_cross_entropy_with_logits(prediction["mask_logits"], mask)
        probabilities = prediction["mask_logits"].sigmoid()
        dice = 1.0 - (2.0 * (probabilities * mask).sum() + 1.0) / (probabilities.sum() + mask.sum() + 1.0)
        mask_loss = mask_bce + dice

        depth_error = (prediction["depth_m"] - target["depth_m"]).abs()
        depth_nll = depth_error * torch.exp(-prediction["log_variance"]) + prediction["log_variance"]
        depth_loss = self._masked_mean(depth_nll, valid)
        horizontal = prediction["depth_m"][..., :, 1:] - prediction["depth_m"][..., :, :-1]
        target_horizontal = target["depth_m"][..., :, 1:] - target["depth_m"][..., :, :-1]
        horizontal_valid = valid[..., :, 1:] * valid[..., :, :-1]
        vertical = prediction["depth_m"][..., 1:, :] - prediction["depth_m"][..., :-1, :]
        target_vertical = target["depth_m"][..., 1:, :] - target["depth_m"][..., :-1, :]
        vertical_valid = valid[..., 1:, :] * valid[..., :-1, :]
        gradient_loss = self._masked_mean((horizontal - target_horizontal).abs(), horizontal_valid)
        gradient_loss += self._masked_mean((vertical - target_vertical).abs(), vertical_valid)

        normal_valid = target.get("normal_valid", valid).float()
        normal_cosine = 1.0 - (prediction["normal"] * target["normal"]).sum(dim=1, keepdim=True)
        normal_loss = self._masked_mean(normal_cosine, normal_valid)

        expected_normal = target.get("expected_plane_normal")
        if expected_normal is None:
            physics_loss = prediction["depth_m"].new_zeros(())
        else:
            expected_normal = F.normalize(expected_normal, dim=1, eps=1e-6)
            plane_cosine = 1.0 - (prediction["normal"] * expected_normal).sum(dim=1, keepdim=True).abs()
            physics_loss = self._masked_mean(plane_cosine, valid)

        wm, wd, wn, wg, wp = self.weights
        total = wm * mask_loss + wd * depth_loss + wn * normal_loss + wg * gradient_loss + wp * physics_loss
        return {
            "total": total,
            "mask": mask_loss,
            "depth_nll": depth_loss,
            "normal": normal_loss,
            "gradient": gradient_loss,
            "physics": physics_loss,
        }
