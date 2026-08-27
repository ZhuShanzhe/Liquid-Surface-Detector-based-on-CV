from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RayLayerHead(nn.Module):
    """Lightweight SeeGroup-inspired set prediction for surfaces on one ray.

    It attaches to the existing RGB-D decoder. The head predicts several metric
    depth candidates, Laplace scales, and existence probabilities without making
    any one candidate the production liquid depth.
    """

    def __init__(
        self,
        input_channels: int,
        num_layers: int = 4,
        max_depth_m: float = 10.0,
        hidden_channels: int = 32,
        min_scale_m: float = 0.002,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least two")
        self.num_layers = int(num_layers)
        self.max_depth_m = float(max_depth_m)
        self.min_scale_m = float(min_scale_m)
        self.shared = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.center_head = nn.Conv2d(hidden_channels, num_layers, 1)
        self.scale_head = nn.Conv2d(hidden_channels, num_layers, 1)
        self.presence_head = nn.Conv2d(hidden_channels, num_layers, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(features)
        centers = torch.sigmoid(self.center_head(shared)) * self.max_depth_m
        centers, order = torch.sort(centers, dim=1)
        scales = F.softplus(self.scale_head(shared)) + self.min_scale_m
        presence_logits = self.presence_head(shared)
        scales = torch.gather(scales, 1, order)
        presence_logits = torch.gather(presence_logits, 1, order)
        return {
            "layer_depths_m": centers,
            "layer_scales_m": scales,
            "layer_presence_logits": presence_logits,
            "layer_confidence": presence_logits.sigmoid() / (1.0 + scales),
        }


class PermutationInvariantLayerLoss(nn.Module):
    """Point-process-style likelihood over an unordered set of ray depths."""

    def __init__(
        self,
        likelihood_weight: float = 1.0,
        count_weight: float = 0.2,
        separation_weight: float = 0.05,
        min_separation_m: float = 0.003,
    ) -> None:
        super().__init__()
        self.likelihood_weight = float(likelihood_weight)
        self.count_weight = float(count_weight)
        self.separation_weight = float(separation_weight)
        self.min_separation_m = float(min_separation_m)

    @staticmethod
    def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return (values * valid).sum() / valid.sum().clamp_min(1.0)

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        target_depths_m: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        centers = prediction["layer_depths_m"]
        scales = prediction["layer_scales_m"].clamp_min(1e-5)
        presence_logits = prediction["layer_presence_logits"]
        if target_depths_m.ndim != 4 or target_valid.shape != target_depths_m.shape:
            raise ValueError("targets must have shape BxTxHxW")
        if centers.shape[0] != target_depths_m.shape[0] or centers.shape[-2:] != target_depths_m.shape[-2:]:
            raise ValueError("prediction and target spatial shapes do not match")

        errors = (target_depths_m[:, :, None] - centers[:, None]).abs()
        log_laplace = -errors / scales[:, None] - torch.log(2.0 * scales[:, None])
        log_weights = F.logsigmoid(presence_logits)[:, None]
        log_intensity = torch.logsumexp(log_weights + log_laplace, dim=2)
        likelihood = self._masked_mean(-log_intensity, target_valid.float())

        target_count = target_valid.sum(dim=1, keepdim=True)
        layer_index = torch.arange(centers.shape[1], device=centers.device, dtype=target_count.dtype).view(
            1, -1, 1, 1
        )
        target_presence = (layer_index < target_count).to(presence_logits.dtype)
        count = F.binary_cross_entropy_with_logits(presence_logits, target_presence)

        gaps = centers[:, 1:] - centers[:, :-1]
        separation = F.relu(self.min_separation_m - gaps).mean()
        total = (
            self.likelihood_weight * likelihood
            + self.count_weight * count
            + self.separation_weight * separation
        )
        return {
            "total": total,
            "set_likelihood": likelihood,
            "count": count,
            "separation": separation,
        }


def select_layer_by_metric_prior(
    layer_depths_m: torch.Tensor,
    layer_confidence: torch.Tensor,
    metric_prior_m: torch.Tensor,
    *,
    maximum_deviation_m: float,
) -> dict[str, torch.Tensor]:
    """Select an interface only when a calibrated geometric prior supports it."""

    if layer_depths_m.shape != layer_confidence.shape:
        raise ValueError("layer depths and confidence must have identical shapes")
    if metric_prior_m.ndim == layer_depths_m.ndim - 1:
        metric_prior_m = metric_prior_m.unsqueeze(1)
    deviation = (layer_depths_m - metric_prior_m).abs()
    normalized = deviation / max(float(maximum_deviation_m), 1e-6)
    score = torch.log(layer_confidence.clamp_min(1e-6)) - normalized
    index = score.argmax(dim=1, keepdim=True)
    selected_depth = torch.gather(layer_depths_m, 1, index)
    selected_confidence = torch.gather(layer_confidence, 1, index)
    selected_deviation = torch.gather(deviation, 1, index)
    accepted = selected_deviation <= maximum_deviation_m
    return {
        "depth_m": selected_depth,
        "confidence": selected_confidence * accepted,
        "layer_index": index,
        "deviation_m": selected_deviation,
        "accepted": accepted,
    }
