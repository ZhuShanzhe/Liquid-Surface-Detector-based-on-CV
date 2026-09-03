from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

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
        max(1.0 / counts[tag] for tag in tags)
        * max((multipliers.get(tag, 1.0) for tag in tags), default=1.0)
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
        next((tag for tag in tags if tag.startswith("severity_")), "severity_unknown")
        for tags in parsed
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
    for scenario, severity, tags in zip(
        scenarios, severities, parsed, strict=True
    ):
        scenario_priority = multipliers.get(scenario, 1.0)
        tag_priority = max(
            (multipliers.get(tag, 1.0) for tag in tags if tag != scenario),
            default=1.0,
        )
        denominator = severity_counts[scenario] * group_counts[(scenario, severity)]
        raw.append(scenario_priority * tag_priority / denominator)
    mean = sum(raw) / len(raw)
    return [weight / mean for weight in raw]
