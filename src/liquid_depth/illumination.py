from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class IlluminationMetrics:
    """Task-facing exposure diagnostics measured on a small image preview."""

    luma_mean: float
    luma_p10: float
    luma_p50: float
    luma_p90: float
    dark_pixel_ratio: float
    saturated_pixel_ratio: float
    dynamic_range: float

    def to_dict(self) -> dict[str, float]:
        return {
            "luma_mean": self.luma_mean,
            "luma_p10": self.luma_p10,
            "luma_p50": self.luma_p50,
            "luma_p90": self.luma_p90,
            "dark_pixel_ratio": self.dark_pixel_ratio,
            "saturated_pixel_ratio": self.saturated_pixel_ratio,
            "dynamic_range": self.dynamic_range,
        }


@dataclass(frozen=True)
class ExposureCorrection:
    image_bgr: np.ndarray
    before: IlluminationMetrics
    after: IlluminationMetrics
    applied: bool
    gamma: float
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "gamma": self.gamma,
            "reason": self.reason,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


def _preview(image_bgr: np.ndarray, max_side: int = 320) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape HxWx3; got {image_bgr.shape}")
    height, width = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image_bgr
    return cv2.resize(
        image_bgr,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def measure_illumination(image_bgr: np.ndarray, max_side: int = 320) -> IlluminationMetrics:
    preview = _preview(image_bgr, max_side)
    luma = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    p10, p50, p90 = np.percentile(luma, (10.0, 50.0, 90.0))
    return IlluminationMetrics(
        luma_mean=float(luma.mean()),
        luma_p10=float(p10),
        luma_p50=float(p50),
        luma_p90=float(p90),
        dark_pixel_ratio=float((luma <= 20.0 / 255.0).mean()),
        saturated_pixel_ratio=float((luma >= 250.0 / 255.0).mean()),
        dynamic_range=float(p90 - p10),
    )


def adaptive_exposure_correction(
    image_bgr: np.ndarray,
    config: dict[str, Any] | None = None,
) -> ExposureCorrection:
    """Conditionally brighten dark frames with a bounded, real-time transform.

    This is deliberately an optional front end. It never invents metric depth and
    is skipped for normally exposed frames so the validated color distribution is
    preserved.
    """

    settings = config or {}
    before = measure_illumination(image_bgr, int(settings.get("preview_max_side", 320)))
    if not bool(settings.get("enabled", False)):
        return ExposureCorrection(image_bgr, before, before, False, 1.0, "disabled")
    trigger = float(settings.get("trigger_luma_p50", 0.22))
    if before.luma_p50 >= trigger:
        return ExposureCorrection(image_bgr, before, before, False, 1.0, "exposure_acceptable")

    target = float(settings.get("target_luma_p50", 0.42))
    minimum_gamma = float(settings.get("min_gamma", 0.45))
    median = max(before.luma_p50, 1.0 / 255.0)
    gamma = float(np.clip(np.log(target) / np.log(median), minimum_gamma, 1.0))
    lookup = np.clip(
        255.0 * np.power(np.arange(256, dtype=np.float32) / 255.0, gamma),
        0.0,
        255.0,
    ).astype(np.uint8)
    corrected = cv2.LUT(image_bgr, lookup)
    clip_limit = float(settings.get("clahe_clip_limit", 0.0))
    if clip_limit > 0.0:
        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(8, 8),
        ).apply(lab[..., 0])
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    after = measure_illumination(corrected, int(settings.get("preview_max_side", 320)))
    return ExposureCorrection(corrected, before, after, True, gamma, "low_light")
