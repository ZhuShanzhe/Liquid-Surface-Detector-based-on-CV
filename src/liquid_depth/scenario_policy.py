from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import depth_to_meters
from .illumination import measure_illumination

SUPPORTED_MODES = {"off", "auto", "always"}
SUPPORTED_MISSING_MODEL_POLICIES = {"reject", "warn_and_use_standard"}


@dataclass(frozen=True)
class SceneSignals:
    raw_depth_valid_ratio: float
    luma_p50: float
    dark_pixel_ratio: float
    saturated_pixel_ratio: float
    dynamic_range: float

    def to_dict(self) -> dict[str, float]:
        return {
            "raw_depth_valid_ratio": self.raw_depth_valid_ratio,
            "luma_p50": self.luma_p50,
            "dark_pixel_ratio": self.dark_pixel_ratio,
            "saturated_pixel_ratio": self.saturated_pixel_ratio,
            "dynamic_range": self.dynamic_range,
        }


@dataclass(frozen=True)
class ComplexSceneDecision:
    mode: str
    requested: bool
    activated: bool
    model_variant: str | None
    triggers: tuple[str, ...]
    held_by_hysteresis: bool
    model_available: bool
    result_allowed: bool
    missing_model_policy: str
    latency_budget_ms: float
    signals: SceneSignals

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "requested": self.requested,
            "activated": self.activated,
            "model_variant": self.model_variant,
            "triggers": list(self.triggers),
            "held_by_hysteresis": self.held_by_hysteresis,
            "model_available": self.model_available,
            "result_allowed": self.result_allowed,
            "missing_model_policy": self.missing_model_policy,
            "latency_budget_ms": self.latency_budget_ms,
            "signals": self.signals.to_dict(),
        }


def load_scene_context(frame_dir: str | Path) -> dict[str, Any]:
    """Load optional operator/runtime scene hints without changing RGB-D format."""

    source = Path(frame_dir)
    for name in ("scene_context.json", "scene_context.yaml", "scene_context.yml"):
        path = source / name
        if not path.is_file():
            continue
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            import yaml

            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"{path} must contain an object")
        return value
    return {}


def measure_scene_signals(
    rgb_bgr: np.ndarray,
    raw_depth: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    *,
    depth_scale_to_m: float | None = None,
    max_depth_m: float = 10.0,
) -> SceneSignals:
    if roi is None:
        image = rgb_bgr
        depth = raw_depth
    else:
        x0, y0, x1, y1 = roi
        image = rgb_bgr[y0:y1, x0:x1]
        depth = raw_depth[y0:y1, x0:x1]
    if image.size == 0 or depth.size == 0:
        raise ValueError("Scene diagnostics ROI is empty")
    illumination = measure_illumination(image)
    if depth_scale_to_m is None:
        depth_m = depth_to_meters(depth)
    else:
        depth_m = depth.astype(np.float32) * float(depth_scale_to_m)
    valid = np.isfinite(depth_m) & (depth_m > 0) & (depth_m <= float(max_depth_m))
    return SceneSignals(
        raw_depth_valid_ratio=float(valid.mean()),
        luma_p50=illumination.luma_p50,
        dark_pixel_ratio=illumination.dark_pixel_ratio,
        saturated_pixel_ratio=illumination.saturated_pixel_ratio,
        dynamic_range=illumination.dynamic_range,
    )


def _truthy(context: dict[str, Any], keys: Iterable[str]) -> bool:
    return any(bool(context.get(key, False)) for key in keys)


class ComplexScenePolicy:
    """Route expensive scene specialists only when configured or diagnostically useful.

    The controller is stateful so video inference does not oscillate between
    models around a threshold. Create one instance per camera stream.
    """

    def __init__(
        self,
        config: dict[str, Any] | None,
        *,
        available_variants: Iterable[str] = (),
    ) -> None:
        self.config = config or {}
        self.mode = str(self.config.get("mode", "off")).lower()
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"complex_scene.mode must be one of {sorted(SUPPORTED_MODES)}")
        self.missing_model_policy = str(self.config.get("missing_model_policy", "reject")).lower()
        if self.missing_model_policy not in SUPPORTED_MISSING_MODEL_POLICIES:
            raise ValueError("complex_scene.missing_model_policy must be reject or warn_and_use_standard")
        self.available_variants = set(available_variants)
        self.hold_frames = max(0, int(self.config.get("hold_frames", 8)))
        self._remaining_hold_frames = 0
        self._held_variant: str | None = None

    @property
    def latency_budget_ms(self) -> float:
        return float(self.config.get("latency_budget_ms", 500.0))

    def _automatic_triggers(
        self,
        signals: SceneSignals,
        context: dict[str, Any],
    ) -> list[str]:
        thresholds = self.config.get("auto", {})
        triggers: list[str] = []
        if _truthy(
            context,
            (
                "transparent_container",
                "transparent_liquid",
                "translucent_liquid",
                "multi_layer_expected",
            ),
        ):
            triggers.append("operator_transparent_or_multilayer_scene")
        if _truthy(context, ("glare_expected", "specular_surface")):
            triggers.append("operator_glare_scene")
        if _truthy(context, ("low_light_expected",)):
            triggers.append("operator_low_light_scene")
        if signals.raw_depth_valid_ratio < float(thresholds.get("raw_depth_valid_ratio_below", 0.45)):
            triggers.append("raw_depth_valid_ratio_low")
        if signals.saturated_pixel_ratio > float(thresholds.get("saturated_pixel_ratio_above", 0.10)):
            triggers.append("saturated_highlight")
        if signals.luma_p50 < float(thresholds.get("luma_p50_below", 0.18)):
            triggers.append("low_light")
        if signals.dark_pixel_ratio > float(thresholds.get("dark_pixel_ratio_above", 0.70)):
            triggers.append("large_dark_region")
        if signals.dynamic_range < float(thresholds.get("dynamic_range_below", 0.06)):
            triggers.append("low_dynamic_range")
        return triggers

    @staticmethod
    def _variant(triggers: list[str], context: dict[str, Any]) -> str:
        requested = context.get("model_variant")
        if requested:
            return str(requested)
        if "operator_transparent_or_multilayer_scene" in triggers:
            return "transparent_multilayer"
        if "operator_glare_scene" in triggers or "saturated_highlight" in triggers:
            return "glare"
        if any(item in triggers for item in ("operator_low_light_scene", "low_light", "large_dark_region")):
            return "low_light"
        return "depth_failure"

    def decide(
        self,
        signals: SceneSignals,
        context: dict[str, Any] | None = None,
    ) -> ComplexSceneDecision:
        context = context or {}
        force = context.get("force_complex_model")
        triggers: list[str]
        held = False
        if force is False:
            requested = False
            triggers = ["operator_forced_off"]
        elif force is True:
            requested = True
            triggers = ["operator_forced_on"]
        elif self.mode == "off":
            requested = False
            triggers = ["policy_off"]
        elif self.mode == "always":
            requested = True
            triggers = ["policy_always"]
        else:
            triggers = self._automatic_triggers(signals, context)
            requested = bool(triggers)

        variant = self._variant(triggers, context) if requested else None
        if requested:
            self._remaining_hold_frames = self.hold_frames
            self._held_variant = variant
        elif force is not False and self.mode == "auto" and self._remaining_hold_frames > 0:
            requested = True
            held = True
            triggers = ["hysteresis_hold"]
            variant = self._held_variant
            self._remaining_hold_frames -= 1

        available = bool(variant and variant in self.available_variants)
        activated = requested and available
        allowed = not (requested and not available and self.missing_model_policy == "reject")
        return ComplexSceneDecision(
            mode=self.mode,
            requested=requested,
            activated=activated,
            model_variant=variant,
            triggers=tuple(triggers),
            held_by_hysteresis=held,
            model_available=available,
            result_allowed=allowed,
            missing_model_policy=self.missing_model_policy,
            latency_budget_ms=self.latency_budget_ms,
            signals=signals,
        )
