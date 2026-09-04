from __future__ import annotations

import math

import torch

from .dataset import MultiTaskDataset


class UniversalMultiTaskDataset(MultiTaskDataset):
    """Canonical dataset adapter with log-uniform 0.1-10 m depth input."""

    def __init__(
        self,
        *args,
        min_depth_m: float = 0.1,
        max_depth_m: float = 10.0,
        **kwargs,
    ) -> None:
        if not 0 < min_depth_m < max_depth_m:
            raise ValueError("Expected 0 < min_depth_m < max_depth_m")
        self.universal_min_depth_m = float(min_depth_m)
        self.universal_max_depth_m = float(max_depth_m)
        super().__init__(*args, max_depth_m=max_depth_m, **kwargs)

    def __getitem__(self, index: int):
        inputs, target = super().__getitem__(index)
        raw_depth_m = inputs[3].clamp(0.0, 1.0) * self.universal_max_depth_m
        validity = inputs[4] > 0
        encoded = torch.log(
            raw_depth_m.clamp(self.universal_min_depth_m, self.universal_max_depth_m)
            / self.universal_min_depth_m
        ) / math.log(self.universal_max_depth_m / self.universal_min_depth_m)
        inputs[3] = torch.where(validity, encoded, torch.zeros_like(encoded))
        target["valid"] *= (target["depth_m"] >= self.universal_min_depth_m).float()
        target["normal_valid"] *= target["valid"]
        if "layer_depths_m" in target:
            layer_range_valid = (
                (target["layer_depths_m"] >= self.universal_min_depth_m)
                & (target["layer_depths_m"] <= self.universal_max_depth_m)
            ).float()
            target["layer_valid"] *= layer_range_valid
        return inputs, target
