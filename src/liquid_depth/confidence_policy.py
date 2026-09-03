from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConfidenceGate:
    enabled: bool
    scenario: str
    threshold: float
    qualified: bool
    require_qualified: bool
    source: str

    @property
    def result_allowed(self) -> bool:
        return not self.enabled or self.qualified or not self.require_qualified

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "scenario": self.scenario,
            "threshold": self.threshold,
            "qualified": self.qualified,
            "require_qualified": self.require_qualified,
            "result_allowed": self.result_allowed,
            "source": self.source,
        }


@lru_cache(maxsize=8)
def _load_report(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("policy"), dict):
        raise TypeError(f"Confidence calibration report has no policy object: {path}")
    return value


def depth_failure_confidence_group(scenario: str, valid_ratio: float | None) -> str:
    if scenario not in {"depth_failure", "compound"} or valid_ratio is None:
        return scenario
    if valid_ratio < 0.10:
        severity = "extreme"
    elif valid_ratio < 0.25:
        severity = "severe"
    elif valid_ratio < 0.45:
        severity = "large"
    else:
        severity = "partial"
    return f"{scenario}:{severity}"


def infer_confidence_scenario(
    model_variant: str | None,
    triggers: tuple[str, ...] | list[str],
    context: dict[str, Any] | None = None,
    route_map: dict[str, str] | None = None,
    raw_depth_valid_ratio: float | None = None,
) -> str:
    context = context or {}
    explicit = context.get("confidence_scenario") or context.get("scenario")
    if explicit:
        return str(explicit)
    trigger_set = set(triggers)
    complex_groups = sum(
        (
            bool(trigger_set & {"operator_transparent_or_multilayer_scene"}),
            bool(trigger_set & {"operator_glare_scene", "saturated_highlight"}),
            bool(trigger_set & {"operator_low_light_scene", "low_light", "large_dark_region"}),
            bool(trigger_set & {"raw_depth_valid_ratio_low"}),
        )
    )
    if complex_groups >= 2:
        return depth_failure_confidence_group("compound", raw_depth_valid_ratio)
    mapping = {
        "transparent_multilayer": "multilayer",
        "glare": "glare",
        "low_light": "low_light",
        "depth_failure": "depth_failure",
    }
    mapping.update(route_map or {})
    scenario = mapping.get(str(model_variant), "ordinary")
    return depth_failure_confidence_group(scenario, raw_depth_valid_ratio)


def select_confidence_gate(
    config: dict[str, Any] | None,
    *,
    model_variant: str | None,
    triggers: tuple[str, ...] | list[str] = (),
    context: dict[str, Any] | None = None,
    raw_depth_valid_ratio: float | None = None,
) -> ConfidenceGate:
    config = config or {}
    enabled = bool(config.get("enabled", False))
    default = float(config.get("default_threshold", 0.0))
    if not 0.0 <= default <= 1.0:
        raise ValueError("confidence_policy.default_threshold must be within [0, 1]")
    route_map = {str(key): str(value) for key, value in config.get("route_to_scenario", {}).items()}
    scenario = infer_confidence_scenario(model_variant, triggers, context, route_map, raw_depth_valid_ratio)
    require_qualified = bool(config.get("require_qualified", True))
    source = "config"
    policy: dict[str, Any] = config.get("thresholds", {})
    report_path = config.get("calibration_report")
    if report_path:
        resolved = str(Path(report_path).expanduser().resolve())
        policy = _load_report(resolved)["policy"]
        source = str(Path(report_path))
    entry = policy.get(scenario) if isinstance(policy, dict) else None
    if entry is None and ":" in scenario and isinstance(policy, dict):
        entry = policy.get(scenario.split(":", maxsplit=1)[0])
    if entry is None:
        entry = {}
    if isinstance(entry, (int, float)):
        threshold = float(entry)
        qualified = True
    elif isinstance(entry, dict):
        threshold = float(entry.get("threshold", default))
        qualified = bool(entry.get("qualified", False))
    else:
        threshold = default
        qualified = False
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Confidence threshold for {scenario} must be within [0, 1]")
    return ConfidenceGate(
        enabled=enabled,
        scenario=scenario,
        threshold=threshold,
        qualified=qualified,
        require_qualified=require_qualified,
        source=source,
    )
