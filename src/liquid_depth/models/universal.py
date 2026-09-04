from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .layered import RayLayerHead
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
        rgb_prior_enabled: bool = False,
        separate_confidence_head: bool = True,
        level_calibration_enabled: bool = False,
        calibration_scale_limit: float = 0.05,
        calibration_bias_limit_m: float = 0.02,
        robust_depth_anchor_enabled: bool = False,
        robust_anchor_mask_threshold: float = 0.5,
        robust_anchor_bias_limit_m: float = 0.25,
        num_ray_layers: int = 0,
        ray_layer_hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        if not 0 < min_depth_m < max_depth_m:
            raise ValueError("Expected 0 < min_depth_m < max_depth_m")
        c = int(base_channels)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.log_depth_span = float(math.log(max_depth_m / min_depth_m))
        self.rgb_prior_enabled = bool(rgb_prior_enabled)
        self.separate_confidence_head = bool(separate_confidence_head)
        self.level_calibration_enabled = bool(level_calibration_enabled)
        self.calibration_scale_limit = float(calibration_scale_limit)
        self.calibration_bias_limit_m = float(calibration_bias_limit_m)
        self.robust_depth_anchor_enabled = bool(robust_depth_anchor_enabled)
        self.robust_anchor_mask_threshold = float(robust_anchor_mask_threshold)
        self.robust_anchor_bias_limit_m = float(robust_anchor_bias_limit_m)
        if not 0.0 <= self.calibration_scale_limit < 1.0:
            raise ValueError("calibration_scale_limit must be in [0, 1)")
        if self.calibration_bias_limit_m < 0.0:
            raise ValueError("calibration_bias_limit_m must be non-negative")
        if not 0.0 < self.robust_anchor_mask_threshold < 1.0:
            raise ValueError("robust_anchor_mask_threshold must be in (0, 1)")
        if self.robust_anchor_bias_limit_m < 0.0:
            raise ValueError("robust_anchor_bias_limit_m must be non-negative")
        if self.rgb_prior_enabled:
            self.rgb_prior = nn.Sequential(ConvBlock(3, c), nn.Conv2d(c, c, 1))
            self.rgb_prior_scale = nn.Parameter(torch.tensor(-2.0))
        else:
            self.rgb_prior = None
            self.register_parameter("rgb_prior_scale", None)

        self.enc1 = ConvBlock(5, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.bottleneck = ConvBlock(c * 4, c * 8)
        if self.level_calibration_enabled:
            self.level_calibration_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c * 8, c * 2),
                nn.SiLU(),
                nn.Linear(c * 2, 2),
            )
            nn.init.zeros_(self.level_calibration_head[-1].weight)
            nn.init.zeros_(self.level_calibration_head[-1].bias)
        else:
            self.level_calibration_head = None
        self.pool = nn.MaxPool2d(2)
        self.dec3 = ConvBlock(c * 8 + c * 4, c * 4)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = ConvBlock(c * 2 + c, c)
        self.mask_head = nn.Conv2d(c, 1, 1)
        self.depth_head = nn.Conv2d(c, 1, 1)
        self.normal_head = nn.Conv2d(c, 3, 1)
        self.log_variance_head = nn.Conv2d(c, 1, 1)
        self.confidence_head = nn.Conv2d(c, 1, 1) if self.separate_confidence_head else None
        self.ray_layer_head = (
            RayLayerHead(
                c,
                num_layers=num_ray_layers,
                min_depth_m=self.min_depth_m,
                max_depth_m=self.max_depth_m,
                hidden_channels=ray_layer_hidden_channels,
            )
            if num_ray_layers > 0
            else None
        )

    @staticmethod
    def _up(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != 5:
            raise ValueError("Expected BCHW normalized RGB, log-depth, validity")
        e1 = self.enc1(inputs)
        if self.rgb_prior is not None:
            missing_depth = 1.0 - inputs[:, 4:5].clamp(0.0, 1.0)
            gate = torch.sigmoid(self.rgb_prior_scale) * (0.2 + 0.8 * missing_depth)
            e1 = e1 + gate * self.rgb_prior(inputs[:, :3])
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        features = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat((self._up(features, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self._up(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self._up(d2, e1), e1), dim=1))
        depth_unit = torch.sigmoid(self.depth_head(d1))
        uncalibrated_depth_m = self.min_depth_m * torch.exp(depth_unit * self.log_depth_span)
        if self.level_calibration_head is not None:
            calibration = self.level_calibration_head(features)
            calibration_scale = 1.0 + self.calibration_scale_limit * torch.tanh(calibration[:, :1])
            calibration_bias_m = self.calibration_bias_limit_m * torch.tanh(calibration[:, 1:2])
        else:
            calibration_scale = torch.ones(inputs.shape[0], 1, device=inputs.device, dtype=inputs.dtype)
            calibration_bias_m = torch.zeros_like(calibration_scale)
        depth_m = (
            uncalibrated_depth_m * calibration_scale[:, :, None, None] + calibration_bias_m[:, :, None, None]
        ).clamp(self.min_depth_m, self.max_depth_m)
        mask_logits = self.mask_head(d1)
        if self.robust_depth_anchor_enabled:
            predicted_surface = torch.sigmoid(mask_logits) >= self.robust_anchor_mask_threshold
            raw_valid = inputs[:, 4:5] > 0.5
            anchor_support = predicted_surface & raw_valid
            raw_depth_m = self.min_depth_m * torch.exp(inputs[:, 3:4].clamp(0.0, 1.0) * self.log_depth_span)
            nan = torch.full_like(raw_depth_m, float("nan"))
            raw_anchor_m = torch.nanmedian(
                torch.where(anchor_support, raw_depth_m, nan).flatten(1), dim=1
            ).values[:, None]
            predicted_anchor_m = torch.nanmedian(
                torch.where(anchor_support, depth_m, nan).flatten(1), dim=1
            ).values[:, None]
            robust_anchor_support_ratio = anchor_support.flatten(1).sum(dim=1, keepdim=True).to(depth_m) / (
                predicted_surface.flatten(1).sum(dim=1, keepdim=True).to(depth_m).clamp_min(1.0)
            )
            robust_anchor_bias_m = robust_anchor_support_ratio * (raw_anchor_m - predicted_anchor_m)
            robust_anchor_bias_m = torch.nan_to_num(robust_anchor_bias_m).clamp(
                -self.robust_anchor_bias_limit_m, self.robust_anchor_bias_limit_m
            )
            depth_m = (depth_m + robust_anchor_bias_m[:, :, None, None]).clamp(
                self.min_depth_m, self.max_depth_m
            )
        else:
            robust_anchor_support_ratio = torch.zeros_like(calibration_scale)
            robust_anchor_bias_m = torch.zeros_like(calibration_bias_m)
        log_variance = self.log_variance_head(d1).clamp(-7.0, 5.0)
        confidence_logits = (
            self.confidence_head(d1.detach()) if self.confidence_head is not None else -log_variance
        )
        result = {
            "mask_logits": mask_logits,
            "depth_m": depth_m,
            "normal": F.normalize(self.normal_head(d1), dim=1, eps=1e-6),
            "log_variance": log_variance,
            "confidence_logits": confidence_logits,
            "confidence": torch.sigmoid(confidence_logits),
            "calibration_scale": calibration_scale,
            "calibration_bias_m": calibration_bias_m,
            "robust_anchor_support_ratio": robust_anchor_support_ratio,
            "robust_anchor_bias_m": robust_anchor_bias_m,
        }
        if self.ray_layer_head is not None:
            result.update(self.ray_layer_head(d1, metric_prior_m=depth_m))
        return result


class UniversalMultiTaskLoss(nn.Module):
    def __init__(
        self,
        *,
        relative_weight: float = 1.0,
        tolerance_weight: float = 0.5,
        uncertainty_weight: float = 0.2,
        surface_level_weight: float = 0.5,
        surface_absolute_weight: float = 0.0,
        surface_tolerance_weight: float = 0.0,
        surface_quantile_weight: float = 0.0,
        surface_quantile: float = 0.90,
        ordinary_loss_boost: float = 0.0,
        calibration_regularization_weight: float = 0.0,
        confidence_relative_tolerance: float = 0.02,
        confidence_absolute_floor_m: float = 0.005,
    ) -> None:
        super().__init__()
        self.base = MultiTaskLoss(tolerance_weight=tolerance_weight)
        self.relative_weight = float(relative_weight)

        self.uncertainty_weight = float(uncertainty_weight)
        self.surface_level_weight = float(surface_level_weight)
        self.surface_absolute_weight = float(surface_absolute_weight)
        self.surface_tolerance_weight = float(surface_tolerance_weight)
        self.surface_quantile_weight = float(surface_quantile_weight)
        self.surface_quantile = float(surface_quantile)
        self.ordinary_loss_boost = float(ordinary_loss_boost)
        self.calibration_regularization_weight = float(calibration_regularization_weight)
        if not 0.0 <= self.surface_quantile < 1.0:
            raise ValueError("surface_quantile must be in [0, 1)")

        self.confidence_relative_tolerance = float(confidence_relative_tolerance)
        self.confidence_absolute_floor_m = float(confidence_absolute_floor_m)

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        losses = self.base(prediction, target)
        valid = target["valid"].float()
        log_error = (
            torch.log(prediction["depth_m"].clamp_min(1e-4)) - torch.log(target["depth_m"].clamp_min(1e-4))
        ).abs()
        relative = (log_error * valid).sum() / valid.sum().clamp_min(1.0)
        losses["relative_log"] = relative
        depth_error = (prediction["depth_m"] - target["depth_m"]).abs()
        tolerance = torch.maximum(
            target["depth_m"] * self.confidence_relative_tolerance,
            torch.full_like(target["depth_m"], self.confidence_absolute_floor_m),
        )
        reliable = (depth_error.detach() <= tolerance).float()
        calibration_map = F.binary_cross_entropy_with_logits(
            prediction["confidence_logits"], reliable, reduction="none"
        )
        calibration = (calibration_map * valid).sum() / valid.sum().clamp_min(1.0)
        losses["uncertainty_calibration"] = calibration
        flattened_valid = valid.flatten(1)
        per_sample_count = flattened_valid.sum(dim=1).clamp_min(1.0)
        predicted_level = (torch.log(prediction["depth_m"].clamp_min(1e-4)).flatten(1) * flattened_valid).sum(
            dim=1
        ) / per_sample_count
        target_level = (torch.log(target["depth_m"].clamp_min(1e-4)).flatten(1) * flattened_valid).sum(
            dim=1
        ) / per_sample_count
        has_support = (flattened_valid.sum(dim=1) > 0).float()
        surface_level = (
            (predicted_level - target_level).abs() * has_support
        ).sum() / has_support.sum().clamp_min(1.0)
        losses["surface_level"] = surface_level
        predicted_linear_level = (prediction["depth_m"].flatten(1) * flattened_valid).sum(
            dim=1
        ) / per_sample_count
        target_linear_level = (target["depth_m"].flatten(1) * flattened_valid).sum(dim=1) / per_sample_count
        surface_absolute = (
            (predicted_linear_level - target_linear_level).abs() * has_support
        ).sum() / has_support.sum().clamp_min(1.0)
        losses["surface_absolute_m"] = surface_absolute
        surface_tolerance_m = torch.maximum(
            target_linear_level * self.confidence_relative_tolerance,
            torch.full_like(target_linear_level, self.confidence_absolute_floor_m),
        )
        normalized_surface_error = (
            predicted_linear_level - target_linear_level
        ).abs() / surface_tolerance_m.clamp_min(1e-6)
        ordinary = target.get("ordinary")
        if ordinary is None:
            ordinary = torch.zeros_like(normalized_surface_error)
        ordinary = ordinary.reshape(-1).to(normalized_surface_error)
        sample_priority = 1.0 + self.ordinary_loss_boost * ordinary
        supported = has_support > 0
        if bool(supported.any()):
            priority_denominator = sample_priority[supported].sum().clamp_min(1.0)
            surface_tolerance = (
                F.relu(normalized_surface_error[supported] - 1.0) * sample_priority[supported]
            ).sum() / priority_denominator
            prioritized_tail = normalized_surface_error[supported] * sample_priority[supported]
            tail_count = max(
                1,
                math.ceil((1.0 - self.surface_quantile) * prioritized_tail.numel()),
            )
            surface_quantile_cvar = torch.topk(prioritized_tail, k=tail_count, largest=True).values.mean()
        else:
            surface_tolerance = normalized_surface_error.new_zeros(())
            surface_quantile_cvar = normalized_surface_error.new_zeros(())
        losses["surface_tolerance"] = surface_tolerance
        losses["surface_quantile_cvar"] = surface_quantile_cvar
        calibration_scale = prediction.get("calibration_scale")
        calibration_bias_m = prediction.get("calibration_bias_m")
        if calibration_scale is None or calibration_bias_m is None:
            calibration_regularization = normalized_surface_error.new_zeros(())
        else:
            calibration_regularization = (calibration_scale - 1.0).square().mean() + (
                calibration_bias_m / max(self.confidence_absolute_floor_m, 1e-6)
            ).square().mean()
        losses["calibration_regularization"] = calibration_regularization
        losses["total"] = (
            losses["total"]
            + self.relative_weight * relative
            + self.uncertainty_weight * calibration
            + self.surface_level_weight * surface_level
            + self.surface_absolute_weight * surface_absolute
            + self.surface_tolerance_weight * surface_tolerance
            + self.surface_quantile_weight * surface_quantile_cvar
            + self.calibration_regularization_weight * calibration_regularization
        )
        return losses
