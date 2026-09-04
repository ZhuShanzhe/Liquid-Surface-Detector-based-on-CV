from __future__ import annotations

import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F


def _gather_layers(values: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 1, order)


def canonicalize_layer_set(
    depths_m: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Sort only for reporting/auxiliary losses, never for component assignment."""

    sortable = depths_m
    if valid is not None:
        sortable = torch.where(
            valid > 0,
            depths_m,
            torch.full_like(depths_m, float("inf")),
        )
    order = sortable.argsort(dim=1)
    sorted_depths = _gather_layers(depths_m, order)
    sorted_valid = _gather_layers(valid, order) if valid is not None else None
    return sorted_depths, sorted_valid, order


class RayLayerHead(nn.Module):
    """Compact recurrent, self-grouping ray decomposition for metric RGB-D.

    Each recurrence extracts one Laplace component and erases its evidence from
    the latent feature map. Component identity is deliberately left unordered;
    callers sort only when a physical downstream operation requires near-to-far
    layers. This follows SeeGroup's central representation while remaining small
    enough for the industrial specialist route.
    """

    def __init__(
        self,
        input_channels: int,
        num_layers: int = 4,
        min_depth_m: float = 0.1,
        max_depth_m: float = 10.0,
        hidden_channels: int = 32,
        min_scale_m: float = 0.002,
        max_depth_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least two")
        if not 0 < min_depth_m < max_depth_m:
            raise ValueError("expected 0 < min_depth_m < max_depth_m")
        self.num_layers = int(num_layers)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.log_depth_span = float(math.log(max_depth_m / min_depth_m))
        self.min_scale_m = float(min_scale_m)
        self.max_log_offset = float(math.log(max_depth_ratio))
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.component_decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.step_embedding = nn.Parameter(torch.zeros(num_layers, hidden_channels, 1, 1))
        nn.init.normal_(self.step_embedding, std=0.02)
        self.step_depth_bias = nn.Parameter(torch.linspace(-0.20, 0.35, num_layers))
        self.depth_head = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.depth_head.weight)
        nn.init.zeros_(self.depth_head.bias)
        self.scale_head = nn.Conv2d(hidden_channels, 1, 1)
        self.presence_head = nn.Conv2d(hidden_channels, 1, 1)
        self.interface_head = nn.Conv2d(hidden_channels, 1, 1)
        self.erase_gate = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.erase_projection = nn.Conv2d(hidden_channels, hidden_channels, 1)

    def forward(
        self,
        features: torch.Tensor,
        metric_prior_m: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        source = self.input_projection(features)
        residual = source
        if metric_prior_m is None:
            metric_prior_m = torch.full(
                (features.shape[0], 1, *features.shape[-2:]),
                math.sqrt(self.min_depth_m * self.max_depth_m),
                device=features.device,
                dtype=features.dtype,
            )
        elif metric_prior_m.shape[-2:] != features.shape[-2:]:
            metric_prior_m = F.interpolate(
                metric_prior_m,
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        metric_prior_m = metric_prior_m.detach().clamp(
            self.min_depth_m,
            self.max_depth_m,
        )
        depths, scales, presence, interfaces = [], [], [], []
        for layer_index in range(self.num_layers):
            component = self.component_decoder(residual + self.step_embedding[layer_index])
            log_offset = (
                torch.tanh(self.depth_head(component) + self.step_depth_bias[layer_index])
                * self.max_log_offset
            )
            depths.append(
                (metric_prior_m * torch.exp(log_offset)).clamp(
                    self.min_depth_m,
                    self.max_depth_m,
                )
            )
            scales.append(F.softplus(self.scale_head(component)) + self.min_scale_m)
            presence.append(self.presence_head(component))
            interfaces.append(self.interface_head(component))
            erased = torch.sigmoid(self.erase_gate(component)) * torch.tanh(self.erase_projection(component))
            residual = F.relu(residual - erased) + 0.05 * source

        layer_depths = torch.cat(depths, dim=1)
        layer_scales = torch.cat(scales, dim=1)
        presence_logits = torch.cat(presence, dim=1)
        interface_logits = torch.cat(interfaces, dim=1)
        presence_probability = presence_logits.sigmoid()
        relative_scale = layer_scales / layer_depths.clamp_min(self.min_depth_m)
        layer_confidence = presence_probability * torch.exp(-relative_scale)
        interface_probability = F.softmax(interface_logits, dim=1) * presence_probability
        sorted_depths, _, order = canonicalize_layer_set(layer_depths)
        return {
            "layer_depths_m": layer_depths,
            "layer_scales_m": layer_scales,
            "layer_presence_logits": presence_logits,
            "layer_interface_logits": interface_logits,
            "layer_confidence": layer_confidence,
            "liquid_interface_probability": interface_probability,
            "layer_depths_sorted_m": sorted_depths,
            "layer_sort_order": order,
        }


class PermutationInvariantLayerLoss(nn.Module):
    """Bidirectional maximum-intensity loss over unordered ray depth sets."""

    def __init__(
        self,
        intensity_weight: float = 1.0,
        count_weight: float = 0.50,
        separation_weight: float = 0.10,
        gradient_weight: float = 0.25,
        interface_weight: float = 0.20,
        assignment_weight: float = 0.75,
        gamma: float = 0.8,
        min_separation_m: float = 0.003,
        relative_separation: float = 0.01,
        multilayer_pixel_boost: float = 3.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        self.intensity_weight = float(intensity_weight)
        self.count_weight = float(count_weight)
        self.separation_weight = float(separation_weight)
        self.gradient_weight = float(gradient_weight)
        self.interface_weight = float(interface_weight)
        self.assignment_weight = float(assignment_weight)
        self.gamma = float(gamma)
        self.min_separation_m = float(min_separation_m)
        self.relative_separation = float(relative_separation)
        self.multilayer_pixel_boost = float(multilayer_pixel_boost)

    @staticmethod
    def _weighted_mean(
        values: torch.Tensor,
        valid: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        combined = valid.to(values) * weight.to(values)
        return (values * combined).sum() / combined.sum().clamp_min(1.0)

    @staticmethod
    def _gradient_matching(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        difference = prediction - target
        total = prediction.sum() * 0.0
        terms = 0
        for step in (1, 2, 4):
            if prediction.shape[-1] > step:
                pair_valid = valid[..., step:] & valid[..., :-step]
                gradient = (difference[..., step:] - difference[..., :-step]).abs()
                total = total + ((gradient * pair_valid).sum() / pair_valid.sum().clamp_min(1))
                terms += 1
            if prediction.shape[-2] > step:
                pair_valid = valid[..., step:, :] & valid[..., :-step, :]
                gradient = (difference[..., step:, :] - difference[..., :-step, :]).abs()
                total = total + ((gradient * pair_valid).sum() / pair_valid.sum().clamp_min(1))
                terms += 1
        return total / max(terms, 1)

    @staticmethod
    def _distinct_assignment(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        pixel_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Minimum one-to-one log-depth assignment for each unordered set."""

        target, valid, _ = canonicalize_layer_set(target, valid)
        target_count = valid.sum(dim=1)
        component_count = prediction.shape[1]
        maximum = min(component_count, target.shape[1])
        total = prediction.sum() * 0.0
        denominator = prediction.sum() * 0.0
        log_prediction = torch.log(prediction.clamp_min(1e-5))
        log_target = torch.log(target.clamp_min(1e-5))
        for count in range(1, maximum + 1):
            selected_pixels = target_count == count
            if not bool(selected_pixels.any()):
                continue
            permutations = torch.tensor(
                list(itertools.permutations(range(component_count), count)),
                device=prediction.device,
            )
            assigned = log_prediction[:, permutations]
            costs = (assigned - log_target[:, None, :count]).abs().mean(dim=2)
            best = costs.min(dim=1).values
            weights = pixel_weight[:, 0] * selected_pixels.to(prediction)
            total = total + (best * weights).sum()
            denominator = denominator + weights.sum()
        return total / denominator.clamp_min(1.0)

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        target_depths_m: torch.Tensor,
        target_valid: torch.Tensor,
        liquid_depth_m: torch.Tensor | None = None,
        liquid_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        centers = prediction["layer_depths_m"]
        scales = prediction["layer_scales_m"].clamp_min(1e-5)
        presence_logits = prediction["layer_presence_logits"]
        presence_probability = presence_logits.sigmoid()
        if target_depths_m.ndim != 4 or target_valid.shape != target_depths_m.shape:
            raise ValueError("targets must have shape BxTxHxW")
        if centers.shape[0] != target_depths_m.shape[0] or centers.shape[-2:] != target_depths_m.shape[-2:]:
            raise ValueError("prediction and target spatial shapes do not match")

        valid = target_valid > 0
        target_count = valid.sum(dim=1, keepdim=True).to(centers)
        pixel_weight = 1.0 + self.multilayer_pixel_boost * (target_count - 1.0).clamp_min(0.0)
        absolute_error = (target_depths_m[:, :, None] - centers[:, None]).abs()
        log_likelihood = (
            -absolute_error / scales[:, None]
            - torch.log(2.0 * scales[:, None])
            + F.logsigmoid(presence_logits)[:, None]
        )

        target_best = log_likelihood.max(dim=2).values
        target_to_prediction = self._weighted_mean(
            -target_best,
            valid,
            pixel_weight.expand_as(target_best),
        )
        prediction_best = (
            log_likelihood.masked_fill(
                ~valid[:, :, None],
                -torch.inf,
            )
            .max(dim=1)
            .values
        )
        any_target = valid.any(dim=1, keepdim=True)
        prediction_best = torch.where(
            any_target,
            prediction_best,
            torch.zeros_like(prediction_best),
        )
        prediction_weights = presence_probability * any_target.to(centers) * pixel_weight
        prediction_to_target = (
            -prediction_best * prediction_weights
        ).sum() / prediction_weights.sum().clamp_min(1.0)
        intensity = (1.0 - self.gamma) * target_to_prediction + self.gamma * prediction_to_target

        predicted_count = presence_probability.sum(dim=1, keepdim=True)
        if bool(any_target.any()):
            count = F.smooth_l1_loss(
                predicted_count[any_target],
                target_count[any_target],
            )
        else:
            count = predicted_count.sum() * 0.0

        pair_difference = (centers[:, :, None] - centers[:, None, :]).abs()
        pair_minimum_depth = torch.minimum(
            centers[:, :, None],
            centers[:, None, :],
        )
        minimum_pair_separation = torch.maximum(
            torch.full_like(pair_difference, self.min_separation_m),
            self.relative_separation * pair_minimum_depth,
        )
        pair_presence = presence_probability[:, :, None] * presence_probability[:, None, :]
        triangle = torch.triu(
            torch.ones(
                centers.shape[1],
                centers.shape[1],
                device=centers.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )[None, :, :, None, None]
        separation_terms = F.relu(minimum_pair_separation - pair_difference) * pair_presence
        separation_mask = triangle & any_target[:, None]
        if bool(separation_mask.any()):
            separation = separation_terms[separation_mask.expand_as(separation_terms)].mean()
        else:
            separation = separation_terms.sum() * 0.0

        sorted_prediction, _, _ = canonicalize_layer_set(centers)
        sorted_target, sorted_valid, _ = canonicalize_layer_set(
            target_depths_m,
            valid,
        )
        gradient = self._gradient_matching(
            sorted_prediction[:, : sorted_target.shape[1]],
            sorted_target,
            sorted_valid.bool(),
        )

        assignment = self._distinct_assignment(
            centers,
            target_depths_m,
            valid,
            pixel_weight,
        )

        interface = centers.sum() * 0.0
        if liquid_depth_m is not None and liquid_valid is not None:
            liquid_error = (centers - liquid_depth_m).abs()
            matched_component = liquid_error.detach().argmin(dim=1)
            interface_map = F.cross_entropy(
                prediction["layer_interface_logits"],
                matched_component,
                reduction="none",
            )
            liquid_mask = liquid_valid[:, 0] > 0
            if bool(liquid_mask.any()):
                interface = interface_map[liquid_mask].mean()

        total = (
            self.intensity_weight * intensity
            + self.count_weight * count
            + self.separation_weight * separation
            + self.gradient_weight * gradient
            + self.interface_weight * interface
            + self.assignment_weight * assignment
        )
        return {
            "total": total,
            "set_intensity": intensity,
            "target_to_prediction": target_to_prediction,
            "prediction_to_target": prediction_to_target,
            "count": count,
            "separation": separation,
            "gradient": gradient,
            "interface": interface,
            "distinct_assignment": assignment,
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


def select_liquid_interface(
    prediction: dict[str, torch.Tensor],
    metric_prior_m: torch.Tensor,
    *,
    relative_tolerance: float = 0.08,
    absolute_floor_m: float = 0.02,
    rejection_multiplier: float = 3.0,
    confidence_threshold: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Fuse learned interface identity with the calibrated single-depth prior."""

    depths = prediction["layer_depths_m"]
    if metric_prior_m.ndim == depths.ndim - 1:
        metric_prior_m = metric_prior_m.unsqueeze(1)
    tolerance = torch.maximum(
        metric_prior_m.abs() * float(relative_tolerance),
        torch.full_like(metric_prior_m, float(absolute_floor_m)),
    )
    deviation = (depths - metric_prior_m).abs()
    interface = prediction["liquid_interface_probability"].clamp_min(1e-7)
    confidence = prediction["layer_confidence"].clamp_min(1e-7)
    score = torch.log(interface) + 0.5 * torch.log(confidence) - deviation / tolerance.clamp_min(1e-6)
    index = score.argmax(dim=1, keepdim=True)
    selected_depth = torch.gather(depths, 1, index)
    selected_deviation = torch.gather(deviation, 1, index)
    selected_interface = torch.gather(interface, 1, index)
    selected_layer_confidence = torch.gather(confidence, 1, index)
    prior_supported = selected_deviation <= rejection_multiplier * tolerance
    selected_confidence = torch.sqrt(selected_interface * selected_layer_confidence) * torch.exp(
        -selected_deviation / tolerance.clamp_min(1e-6)
    )
    confidence_supported = selected_confidence >= float(confidence_threshold)
    accepted = prior_supported & confidence_supported
    rejection_code = torch.zeros_like(index)
    rejection_code = torch.where(
        ~prior_supported,
        torch.ones_like(rejection_code),
        rejection_code,
    )
    rejection_code = torch.where(
        prior_supported & ~confidence_supported,
        torch.full_like(rejection_code, 2),
        rejection_code,
    )
    return {
        "depth_m": selected_depth,
        "confidence": selected_confidence * prior_supported,
        "layer_index": index,
        "deviation_m": selected_deviation,
        "accepted": accepted,
        "rejection_code": rejection_code,
        "tolerance_m": tolerance,
    }
