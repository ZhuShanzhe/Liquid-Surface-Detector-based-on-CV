import pytest

from liquid_depth.evaluation_manifest import (
    EvaluationThresholds,
    cap_records_per_source_bucket,
    difficulty_buckets,
)
from liquid_depth.scenario_policy import SceneSignals


def _signals(**overrides):
    values = {
        "raw_depth_valid_ratio": 0.95,
        "luma_p50": 0.5,
        "dark_pixel_ratio": 0.01,
        "saturated_pixel_ratio": 0.01,
        "dynamic_range": 0.4,
    }
    values.update(overrides)
    return SceneSignals(**values)


def test_difficulty_buckets_are_independent_and_contract_aware():
    buckets = difficulty_buckets(
        _signals(
            raw_depth_valid_ratio=0.1,
            saturated_pixel_ratio=0.3,
        ),
        "transparent;container_edge",
    )
    assert buckets == (
        "depth_failure",
        "glare",
        "transparent_general",
    )
    assert difficulty_buckets(
        _signals(),
        multi_layer=True,
    ) == ("transparent_multilayer",)


def test_thresholds_match_runtime_boundary_semantics():
    limits = EvaluationThresholds()
    assert "low_light" in difficulty_buckets(
        _signals(luma_p50=limits.luma_p50_below - 0.01)
    )
    assert "glare" not in difficulty_buckets(
        _signals(
            saturated_pixel_ratio=(
                limits.saturated_pixel_ratio_above
            )
        )
    )


def test_cap_is_deterministic_and_per_source_bucket():
    records = [
        {
            "record_id": f"a-{index}",
            "dataset": "a",
            "bucket": "glare",
        }
        for index in range(10)
    ] + [
        {
            "record_id": f"b-{index}",
            "dataset": "b",
            "bucket": "glare",
        }
        for index in range(10)
    ]
    first = cap_records_per_source_bucket(records, 3, seed=7)
    second = cap_records_per_source_bucket(
        reversed(records),
        3,
        seed=7,
    )
    assert first == second
    assert len(first) == 6
    assert {row["dataset"] for row in first} == {"a", "b"}
    with pytest.raises(ValueError):
        cap_records_per_source_bucket(records, 0)
