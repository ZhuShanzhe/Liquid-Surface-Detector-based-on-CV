from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

DIFFICULTY_TAGS = (
    "transparent",
    "translucent",
    "glare",
    "saturated_highlight",
    "container_edge",
)


def parse_tags(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.replace(";", ",").split(",")
    else:
        values = value
    return tuple(sorted({str(item).strip().lower() for item in values if str(item).strip()}))


def balanced_sample_weights(
    rows: Iterable[Mapping[str, object]], tag_column: str = "difficulty_tags"
) -> list[float]:
    rows = list(rows)
    parsed = [parse_tags(str(row.get(tag_column, "ordinary"))) or ("ordinary",) for row in rows]
    counts = Counter(tag for tags in parsed for tag in tags)
    if not rows:
        return []
    raw = [max(1.0 / counts[tag] for tag in tags) for tags in parsed]
    mean = sum(raw) / len(raw)
    return [weight / mean for weight in raw]
