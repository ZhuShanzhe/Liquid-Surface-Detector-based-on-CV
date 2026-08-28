from __future__ import annotations

from collections.abc import Iterable, Mapping

from .evaluation_manifest import stable_fraction
from .sampling import parse_tags

SUPPORTED_AUGMENTATION_PROFILES = (
    "standard",
    "glare",
    "depth_failure",
    "low_light",
)


def assign_augmentation_profile(
    frame_id: str,
    profiles: Iterable[str],
    *,
    seed: int = 2026,
) -> str:
    choices = tuple(str(value).strip().lower() for value in profiles)
    if not choices:
        raise ValueError("at least one augmentation profile is required")
    unknown = set(choices) - set(SUPPORTED_AUGMENTATION_PROFILES)
    if unknown:
        raise ValueError(
            f"unsupported augmentation profiles: {sorted(unknown)}"
        )
    index = min(
        int(stable_fraction(frame_id, seed) * len(choices)),
        len(choices) - 1,
    )
    return choices[index]


def specialize_rows(
    rows: Iterable[Mapping[str, object]],
    profiles: Iterable[str],
    *,
    seed: int = 2026,
) -> list[dict[str, str]]:
    choices = tuple(profiles)
    output = []
    for source in rows:
        row = {key: str(value) for key, value in source.items()}
        if row.get("split", "").strip() == "train":
            profile = assign_augmentation_profile(
                row.get("frame_id", ""),
                choices,
                seed=seed,
            )
            tags = set(parse_tags(row.get("difficulty_tags", "")))
            if profile != "standard":
                tags.add(profile)
            row["difficulty_tags"] = ";".join(sorted(tags))
        else:
            profile = "standard"
        row["augmentation_profile"] = profile
        output.append(row)
    return output
