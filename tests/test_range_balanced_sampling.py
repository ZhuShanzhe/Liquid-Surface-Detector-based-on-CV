from __future__ import annotations

from liquid_depth.sampling import range_balanced_sample_weights


def test_range_balancing_increases_rare_range_weight(tmp_path):
    rows = [
        {"reference_depth_m": "0.2"},
        {"reference_depth_m": "0.25"},
        {"reference_depth_m": "0.28"},
        {"reference_depth_m": "5.0"},
    ]
    weights = range_balanced_sample_weights(
        rows,
        [1.0] * len(rows),
        manifest_root=tmp_path,
        strength=1.0,
    )
    assert weights[-1] > weights[0]
    assert abs(sum(weights) / len(weights) - 1.0) < 1e-9


def test_range_balancing_zero_strength_preserves_base_ratios(tmp_path):
    rows = [
        {"reference_depth_m": "0.2"},
        {"reference_depth_m": "5.0"},
    ]
    weights = range_balanced_sample_weights(
        rows,
        [1.0, 2.0],
        manifest_root=tmp_path,
        strength=0.0,
    )
    assert weights[1] / weights[0] == 2.0
