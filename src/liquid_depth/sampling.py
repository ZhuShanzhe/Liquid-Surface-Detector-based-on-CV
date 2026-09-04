from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

DIFFICULTY_TAGS = (
    "transparent",
    "translucent",
    "glare",
    "saturated_highlight",
    "container_edge",
    "depth_failure",
    "multilayer",
    "low_light",
    "nonplanar_surface",
    "compound",
)


def parse_tags(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.replace(";", ",").split(",")
    else:
        values = value
    return tuple(sorted({str(item).strip().lower() for item in values if str(item).strip()}))


def balanced_sample_weights(
    rows: Iterable[Mapping[str, object]],
    tag_column: str = "difficulty_tags",
    priority_multipliers: Mapping[str, float] | None = None,
) -> list[float]:
    rows = list(rows)
    parsed = [parse_tags(str(row.get(tag_column, "ordinary"))) or ("ordinary",) for row in rows]
    counts = Counter(tag for tags in parsed for tag in tags)
    if not rows:
        return []
    multipliers = {
        str(tag).strip().lower(): max(float(value), 0.0)
        for tag, value in (priority_multipliers or {}).items()
    }
    raw = [
        max(1.0 / counts[tag] for tag in tags) * max((multipliers.get(tag, 1.0) for tag in tags), default=1.0)
        for tags in parsed
    ]
    mean = sum(raw) / len(raw)
    return [weight / mean for weight in raw]


def scenario_severity_sample_weights(
    rows: Iterable[Mapping[str, object]],
    *,
    priority_multipliers: Mapping[str, float] | None = None,
) -> list[float]:
    """Balance scene families, then corruption severity within each family."""

    rows = list(rows)
    if not rows:
        return []
    parsed = [parse_tags(str(row.get("difficulty_tags", ""))) for row in rows]
    scenarios = [str(row.get("scenario", "ordinary")).strip().lower() for row in rows]
    severities = [
        next((tag for tag in tags if tag.startswith("severity_")), "severity_unknown") for tags in parsed
    ]
    groups = list(zip(scenarios, severities, strict=True))
    group_counts = Counter(groups)
    severity_counts = Counter()
    for scenario, _ in set(groups):
        severity_counts[scenario] += 1
    multipliers = {
        str(tag).strip().lower(): max(float(value), 0.0)
        for tag, value in (priority_multipliers or {}).items()
    }
    raw = []
    for scenario, severity, tags in zip(scenarios, severities, parsed, strict=True):
        scenario_priority = multipliers.get(scenario, 1.0)
        tag_priority = max(
            (multipliers.get(tag, 1.0) for tag in tags if tag != scenario),
            default=1.0,
        )
        denominator = severity_counts[scenario] * group_counts[(scenario, severity)]
        raw.append(scenario_priority * tag_priority / denominator)
    mean = sum(raw) / len(raw)
    return [weight / mean for weight in raw]


DEFAULT_RANGE_BINS = (
    ("0.1-0.3m", 0.1, 0.3),
    ("0.3-1m", 0.3, 1.0),
    ("1-3m", 1.0, 3.0),
    ("3-10m", 3.0, 10.0001),
)


def _row_reference_distance_m(row: Mapping[str, object], manifest_root: Path) -> float | None:
    for key in ("reference_depth_m", "camera_distance_m", "surface_depth_m"):
        value = str(row.get(key, "")).strip()
        if value:
            return float(value)
    metadata_value = str(row.get("metadata_path", "")).strip()
    if not metadata_value:
        return None
    metadata_path = Path(metadata_value).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = manifest_root / metadata_path
    try:
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        position = metadata.get("camera_position_m")
        target = metadata.get("camera_target_m")
        if len(position) == 3 and len(target) == 3:
            return math.dist(tuple(map(float, position)), tuple(map(float, target)))
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None
    return None


def range_balanced_sample_weights(
    rows: Iterable[Mapping[str, object]],
    base_weights: Iterable[float],
    *,
    manifest_root: str | Path,
    strength: float = 1.0,
    maximum_factor: float = 4.0,
    range_bins: tuple[tuple[str, float, float], ...] = DEFAULT_RANGE_BINS,
) -> list[float]:
    """Combine existing weights with inverse-frequency camera-range factors."""

    rows = list(rows)
    base_weights = [float(value) for value in base_weights]
    if len(rows) != len(base_weights):
        raise ValueError("rows and base_weights must have the same length")
    if not rows:
        return []
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    root = Path(manifest_root)
    labels: list[str] = []
    for row in rows:
        distance = _row_reference_distance_m(row, root)
        label = "unknown"
        if distance is not None:
            label = next(
                (name for name, lower, upper in range_bins if lower <= distance < upper),
                "unknown",
            )
        labels.append(label)
    counts = Counter(labels)
    inverse = [1.0 / counts[label] for label in labels]
    inverse_mean = sum(inverse) / len(inverse)
    factors = [
        min(maximum_factor, max(1.0 / maximum_factor, value / inverse_mean)) ** strength for value in inverse
    ]
    combined = [base * factor for base, factor in zip(base_weights, factors, strict=True)]
    mean = sum(combined) / len(combined)
    return [value / mean for value in combined]
