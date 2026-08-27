from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from liquid_depth.models.multitask import ConvBlock


def sample_cubic_bezier(
    control_points: torch.Tensor,
    samples: int = 64,
) -> torch.Tensor:
    """Differentiably sample normalized cubic Bezier control points."""
    if control_points.ndim != 3 or control_points.shape[1:] != (4, 2):
        raise ValueError("control_points must have shape (batch, 4, 2)")
    parameter = torch.linspace(
        0.0,
        1.0,
        samples,
        device=control_points.device,
        dtype=control_points.dtype,
    )
    basis = torch.stack(
        (
            (1.0 - parameter) ** 3,
            3.0 * (1.0 - parameter) ** 2 * parameter,
            3.0 * (1.0 - parameter) * parameter**2,
            parameter**3,
        ),
        dim=1,
    )
    return torch.einsum("sk,bkd->bsd", basis, control_points)


def sample_curve_support(
    contact_logits: torch.Tensor,
    normalized_curve: torch.Tensor,
) -> torch.Tensor:
    """Read independent contact-map support at each predicted curve point."""
    if contact_logits.ndim != 4 or contact_logits.shape[1] != 1:
        raise ValueError("contact_logits must have shape (batch, 1, height, width)")
    if normalized_curve.ndim != 3 or normalized_curve.shape[2] != 2:
        raise ValueError("normalized_curve must have shape (batch, samples, 2)")
    if len(contact_logits) != len(normalized_curve):
        raise ValueError("contact logits and curve batch sizes differ")
    sampling_grid = normalized_curve.mul(2.0).sub(1.0)[:, None]
    return F.grid_sample(contact_logits.sigmoid(), sampling_grid, align_corners=True).squeeze(1).squeeze(1)


class ColorRectificationModule(nn.Module):
    """Predict a bounded RGB residual before contact-line feature extraction."""

    def __init__(self, channels: int = 12, max_residual: float = 0.5) -> None:
        super().__init__()
        self.max_residual = float(max_residual)
        self.enc1 = ConvBlock(3, channels)
        self.enc2 = ConvBlock(channels, channels * 2)
        self.bottleneck = ConvBlock(channels * 2, channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.dec2 = ConvBlock(channels * 6, channels * 2)
        self.dec1 = ConvBlock(channels * 3, channels)
        self.head = nn.Conv2d(channels, 3, 1)

    @staticmethod
    def _up(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, rgb_unit: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(rgb_unit)
        e2 = self.enc2(self.pool(e1))
        features = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat((self._up(features, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self._up(d2, e1), e1), dim=1))
        return torch.tanh(self.head(d1)) * self.max_residual


class DTLDContactGeometryNet(nn.Module):
    """CRM + contact segmentation + cubic Bezier perception for explicit geometry."""

    def __init__(
        self,
        base_channels: int = 24,
        geometry_conditioning: bool = False,
        object_experts: bool = False,
    ) -> None:
        super().__init__()
        channels = int(base_channels)
        self.geometry_conditioning = bool(geometry_conditioning)
        self.object_experts = bool(object_experts)
        self.crm = ColorRectificationModule(max(channels // 2, 8))
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        self.enc1 = ConvBlock(5, channels)
        self.enc2 = ConvBlock(channels, channels * 2)
        self.enc3 = ConvBlock(channels * 2, channels * 4)
        self.bottleneck = ConvBlock(channels * 4, channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = ConvBlock(channels * 12, channels * 4)
        self.dec2 = ConvBlock(channels * 6, channels * 2)
        self.dec1 = ConvBlock(channels * 3, channels)
        spatial_objects = 4 if self.object_experts else 1
        self.contact_head = nn.Conv2d(channels, spatial_objects, 1)
        self.control_heatmap_head = nn.Conv2d(channels, spatial_objects * 4, 1)
        self.object_embedding = nn.Embedding(4, 8)
        if self.geometry_conditioning:
            self.geometry_film = nn.Sequential(
                nn.Linear(20, channels * 4),
                nn.SiLU(),
                nn.Linear(channels * 4, channels * 16),
            )
            nn.init.zeros_(self.geometry_film[-1].weight)
            nn.init.zeros_(self.geometry_film[-1].bias)
        self.curve_quality_head = nn.Sequential(
            nn.Linear(channels * 8 + 12 + 8, channels * 4),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(channels * 4, 2),
        )

    @staticmethod
    def _up(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _soft_argmax(logits: torch.Tensor, temperature: float = 0.25) -> torch.Tensor:
        batch, controls, height, width = logits.shape
        probability = (logits.flatten(2) / temperature).softmax(dim=2)
        x = torch.linspace(0.0, 1.0, width, device=logits.device, dtype=logits.dtype)
        y = torch.linspace(0.0, 1.0, height, device=logits.device, dtype=logits.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        location_x = torch.sum(probability * grid_x.flatten(), dim=2)
        location_y = torch.sum(probability * grid_y.flatten(), dim=2)
        return torch.stack((location_x, location_y), dim=2).reshape(batch, controls, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        object_index: torch.Tensor,
        pose: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != 5:
            raise ValueError("Expected normalized RGB, metric-depth fraction, and validity channels")
        rgb_unit = inputs[:, :3] * self.rgb_std + self.rgb_mean
        color_residual = self.crm(rgb_unit)
        rectified_rgb = (rgb_unit + color_residual).clamp(0.0, 1.0)
        rectified_input = torch.cat(
            ((rectified_rgb - self.rgb_mean) / self.rgb_std, inputs[:, 3:]),
            dim=1,
        )

        e1 = self.enc1(rectified_input)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        features = self.bottleneck(self.pool(e3))
        object_features = self.object_embedding(object_index)
        if self.geometry_conditioning:
            film = self.geometry_film(torch.cat((pose, object_features), dim=1))
            scale, bias = film.chunk(2, dim=1)
            features = features * (1.0 + 0.1 * torch.tanh(scale[:, :, None, None]))
            features = features + 0.1 * torch.tanh(bias[:, :, None, None])
        d3 = self.dec3(torch.cat((self._up(features, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self._up(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self._up(d2, e1), e1), dim=1))
        contact_logits = self.contact_head(d1)
        control_heatmap_logits = self.control_heatmap_head(d1)
        if self.object_experts:
            batch_index = torch.arange(len(d1), device=d1.device)
            contact_logits = contact_logits[batch_index, object_index][:, None]
            control_heatmap_logits = control_heatmap_logits.reshape(len(d1), 4, 4, *d1.shape[-2:])[
                batch_index, object_index
            ]
        control_points = self._soft_argmax(control_heatmap_logits)

        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        curve_output = self.curve_quality_head(torch.cat((pooled, pose, object_features), dim=1))
        presence_logit = curve_output[:, 0]
        curve_log_variance = curve_output[:, 1].clamp(-7.0, 5.0)
        contact_curve = sample_cubic_bezier(control_points)
        curve_confidence = presence_logit.sigmoid() * torch.sigmoid(-curve_log_variance)
        return {
            "color_residual": color_residual,
            "rectified_rgb": rectified_rgb,
            "contact_logits": contact_logits,
            "control_heatmap_logits": control_heatmap_logits,
            "bezier_control_points": control_points,
            "contact_curve": contact_curve,
            "curve_presence_logit": presence_logit,
            "curve_log_variance": curve_log_variance,
            "curve_confidence": curve_confidence,
            "contact_curve_point_confidence": sample_curve_support(contact_logits, contact_curve),
        }


class DTLDResNet34BezierNet(nn.Module):
    """ImageNet ResNet34 encoder with a full-resolution Bezier heatmap decoder."""

    def __init__(self, pretrained_backbone: bool = False) -> None:
        super().__init__()
        from torchvision.models import ResNet34_Weights, resnet34

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        encoder = resnet34(weights=weights)
        rgb_stem = encoder.conv1
        self.stem_conv = nn.Conv2d(
            5,
            rgb_stem.out_channels,
            kernel_size=rgb_stem.kernel_size,
            stride=rgb_stem.stride,
            padding=rgb_stem.padding,
            bias=False,
        )
        with torch.no_grad():
            self.stem_conv.weight[:, :3].copy_(rgb_stem.weight)
            auxiliary = rgb_stem.weight.mean(dim=1, keepdim=True) * 0.1
            self.stem_conv.weight[:, 3:].copy_(auxiliary.repeat(1, 2, 1, 1))
        self.stem_bn = encoder.bn1
        self.stem_relu = encoder.relu
        self.stem_pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.crm = ColorRectificationModule(12)
        nn.init.zeros_(self.crm.head.weight)
        nn.init.zeros_(self.crm.head.bias)
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        self.dec3 = ConvBlock(512 + 256, 256)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.dec1 = ConvBlock(128 + 64, 64)
        self.dec0 = ConvBlock(64 + 64, 64)
        self.full_resolution = ConvBlock(64, 64)
        self.contact_head = nn.Conv2d(64, 1, 1)
        self.control_heatmap_head = nn.Conv2d(64, 4, 1)
        self.object_embedding = nn.Embedding(4, 8)
        self.curve_quality_head = nn.Sequential(
            nn.Linear(512 + 12 + 8, 256),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 2),
        )

    @staticmethod
    def _up(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        object_index: torch.Tensor,
        pose: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != 5:
            raise ValueError("Expected normalized RGB, metric-depth fraction, and validity channels")
        output_size = inputs.shape[-2:]
        rgb_unit = inputs[:, :3] * self.rgb_std + self.rgb_mean
        color_residual = self.crm(rgb_unit)
        rectified_rgb = (rgb_unit + color_residual).clamp(0.0, 1.0)
        rectified_input = torch.cat(
            ((rectified_rgb - self.rgb_mean) / self.rgb_std, inputs[:, 3:]),
            dim=1,
        )

        stem = self.stem_relu(self.stem_bn(self.stem_conv(rectified_input)))
        e1 = self.layer1(self.stem_pool(stem))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        features = self.layer4(e3)
        d3 = self.dec3(torch.cat((self._up(features, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self._up(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self._up(d2, e1), e1), dim=1))
        d0 = self.dec0(torch.cat((self._up(d1, stem), stem), dim=1))
        decoded = self.full_resolution(
            F.interpolate(d0, size=output_size, mode="bilinear", align_corners=False)
        )
        contact_logits = self.contact_head(decoded)
        control_heatmap_logits = self.control_heatmap_head(decoded)
        control_points = DTLDContactGeometryNet._soft_argmax(control_heatmap_logits)
        object_features = self.object_embedding(object_index)
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        curve_output = self.curve_quality_head(torch.cat((pooled, pose, object_features), dim=1))
        presence_logit = curve_output[:, 0]
        curve_log_variance = curve_output[:, 1].clamp(-7.0, 5.0)
        contact_curve = sample_cubic_bezier(control_points)
        curve_confidence = presence_logit.sigmoid() * torch.sigmoid(-curve_log_variance)
        return {
            "color_residual": color_residual,
            "rectified_rgb": rectified_rgb,
            "contact_logits": contact_logits,
            "control_heatmap_logits": control_heatmap_logits,
            "bezier_control_points": control_points,
            "contact_curve": contact_curve,
            "curve_presence_logit": presence_logit,
            "curve_log_variance": curve_log_variance,
            "curve_confidence": curve_confidence,
            "contact_curve_point_confidence": sample_curve_support(contact_logits, contact_curve),
        }


def build_dtld_contact_model(
    backbone: str = "unet",
    base_channels: int = 24,
    pretrained_backbone: bool = False,
    geometry_conditioning: bool = False,
    object_experts: bool = False,
) -> nn.Module:
    if backbone == "unet":
        return DTLDContactGeometryNet(
            base_channels,
            geometry_conditioning=geometry_conditioning,
            object_experts=object_experts,
        )
    if backbone == "resnet34":
        if geometry_conditioning or object_experts:
            raise ValueError("ResNet34 comparison does not combine rejected FiLM/expert ablations")
        return DTLDResNet34BezierNet(pretrained_backbone=pretrained_backbone)
    raise ValueError(f"Unsupported DTLD contact backbone: {backbone!r}")


class DTLDContactGeometryLoss(nn.Module):
    """TCLD-style multitask loss with heteroscedastic curve uncertainty."""

    def __init__(
        self,
        curve_weight: float = 1.0,
        presence_weight: float = 0.1,
        localization_weight: float = 0.25,
        segmentation_weight: float = 0.75,
        residual_weight: float = 1.0,
        consistency_weight: float = 0.0,
        decoupled_uncertainty: bool = False,
        uncertainty_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if uncertainty_weight < 0.0:
            raise ValueError("uncertainty_weight must be non-negative")
        self.weights = (
            float(curve_weight),
            float(presence_weight),
            float(localization_weight),
            float(segmentation_weight),
            float(residual_weight),
            float(consistency_weight),
        )
        self.decoupled_uncertainty = bool(decoupled_uncertainty)
        self.uncertainty_weight = float(uncertainty_weight)

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        contact = target["contact"]
        logits = prediction["contact_logits"]
        positive_weight = torch.as_tensor(20.0, device=logits.device, dtype=logits.dtype)
        segmentation_bce = F.binary_cross_entropy_with_logits(
            logits,
            contact,
            pos_weight=positive_weight,
        )
        probability = logits.sigmoid()
        segmentation_dice = 1.0 - (2.0 * (probability * contact).sum() + 1.0) / (
            probability.sum() + contact.sum() + 1.0
        )
        segmentation = segmentation_bce + segmentation_dice

        target_control = target["bezier_control_points"]
        predicted_control = prediction["bezier_control_points"]
        control_logits = prediction["control_heatmap_logits"]
        _, controls, height, width = control_logits.shape
        x = torch.linspace(0.0, 1.0, width, device=logits.device, dtype=logits.dtype)
        y = torch.linspace(0.0, 1.0, height, device=logits.device, dtype=logits.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        squared_distance = (grid_x[None, None] - target_control[:, :, 0, None, None]).square() + (
            grid_y[None, None] - target_control[:, :, 1, None, None]
        ).square()
        target_distribution = (-squared_distance / (2.0 * 0.02**2)).flatten(2).softmax(dim=2)
        localization = (
            F.kl_div(
                control_logits.flatten(2).log_softmax(dim=2),
                target_distribution,
                reduction="batchmean",
            )
            / controls
        )
        control_error = torch.linalg.vector_norm(
            predicted_control - target_control,
            dim=2,
        ).mean(dim=1)
        predicted_curve = sample_cubic_bezier(predicted_control)
        target_curve = sample_cubic_bezier(target_control)
        sampled_error = torch.linalg.vector_norm(
            predicted_curve - target_curve,
            dim=2,
        ).mean(dim=1)
        curve_error = control_error + sampled_error
        target_grid = target_curve.mul(2.0).sub(1.0)[:, None]
        predicted_grid = predicted_curve.mul(2.0).sub(1.0)[:, None]
        predicted_on_target = (
            F.grid_sample(probability, target_grid, align_corners=True).squeeze(1).squeeze(1)
        )
        target_on_prediction = (
            F.grid_sample(contact, predicted_grid, align_corners=True).squeeze(1).squeeze(1)
        )
        consistency = (
            -torch.log(predicted_on_target.clamp_min(1e-5)).mean() + (1.0 - target_on_prediction).mean()
        )
        log_variance = prediction["curve_log_variance"]
        uncertainty = (curve_error.detach() * torch.exp(-log_variance) + 0.5 * log_variance).mean()
        if self.decoupled_uncertainty:
            curve = curve_error.mean()
        else:
            curve = (curve_error * torch.exp(-log_variance) + 0.5 * log_variance).mean()
            uncertainty = uncertainty.detach().new_zeros(())

        presence = F.binary_cross_entropy_with_logits(
            prediction["curve_presence_logit"],
            torch.ones_like(prediction["curve_presence_logit"]),
        )
        residual_difference = prediction["color_residual"] - target["color_residual"]
        residual_importance = 1.0 + 20.0 * contact
        residual = (residual_difference.square() * residual_importance).sum() / (
            3.0 * residual_importance.sum().clamp_min(1.0)
        )

        wc, wp, wl, ws, wr, wcons = self.weights
        total = (
            wc * curve
            + self.uncertainty_weight * uncertainty
            + wp * presence
            + wl * localization
            + ws * segmentation
            + wr * residual
            + wcons * consistency
        )
        return {
            "total": total,
            "curve": curve,
            "uncertainty": uncertainty,
            "presence": presence,
            "localization": localization,
            "segmentation": segmentation,
            "residual": residual,
            "consistency": consistency,
        }
