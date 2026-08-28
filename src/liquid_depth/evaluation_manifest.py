from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .sampling import parse_tags
from .scenario_policy import SceneSignals


@dataclass(frozen=True)
class EvaluationThresholds:
    raw_depth_valid_ratio_below: float = 0.45
    saturated_pixel_ratio_above: float = 0.10
    luma_p50_below: float = 0.18
    dark_pixel_ratio_above: float = 0.70
    dynamic_range_below: float = 0.06


def difficulty_buckets(
    signals: SceneSignals,
    difficulty_tags: str | Iterable[str] = (),
    *,
    multi_layer: bool = False,
    thresholds: EvaluationThresholds | None = None,
) -> tuple[str, ...]:
    """Assign independent evaluation buckets without forcing one winning route."""

    limits = thresholds or EvaluationThresholds()
    tags = set(parse_tags(difficulty_tags))
    buckets: set[str] = set()
    if multi_layer or "multi_layer" in tags:
        buckets.add("transparent_multilayer")
    if tags.intersection({"transparent", "translucent", "non_lambertian"}):
        buckets.add("transparent_general")
    if (
        tags.intersection({"glare", "saturated_highlight", "specular"})
        or signals.saturated_pixel_ratio > limits.saturated_pixel_ratio_above
    ):
        buckets.add("glare")
    if (
        "low_light" in tags
        or signals.luma_p50 < limits.luma_p50_below
        or signals.dark_pixel_ratio > limits.dark_pixel_ratio_above
        or signals.dynamic_range < limits.dynamic_range_below
    ):
        buckets.add("low_light")
    if (
        tags.intersection({"depth_failure", "depth_hole", "depth_dropout"})
        or signals.raw_depth_valid_ratio < limits.raw_depth_valid_ratio_below
    ):
        buckets.add("depth_failure")
    return tuple(sorted(buckets))


def stable_fraction(key: str, seed: int = 2026) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def cap_records_per_source_bucket(
    records: Iterable[Mapping[str, Any]],
    maximum: int,
    *,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Cap each dataset/bucket so large synthetic sets cannot dominate."""

    if maximum < 1:
        raise ValueError("maximum must be positive")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        record = dict(item)
        groups[(str(record["dataset"]), str(record["bucket"]))].append(record)
    selected = []
    for group in sorted(groups):
        ranked = sorted(
            groups[group],
            key=lambda row: stable_fraction(str(row["record_id"]), seed),
        )
        selected.extend(ranked[:maximum])
    return sorted(
        selected,
        key=lambda row: (
            str(row["bucket"]),
            str(row["dataset"]),
            str(row["record_id"]),
        ),
    )
