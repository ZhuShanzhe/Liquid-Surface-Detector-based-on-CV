import json

from liquid_depth.confidence_policy import (
    infer_confidence_scenario,
    select_confidence_gate,
)


def test_infers_compound_scene_from_multiple_signal_groups():
    scenario = infer_confidence_scenario(
        "low_light",
        ("low_light", "raw_depth_valid_ratio_low"),
    )
    assert scenario == "compound"


def test_explicit_scene_context_overrides_route():
    scenario = infer_confidence_scenario(
        "transparent_multilayer",
        ("operator_transparent_or_multilayer_scene",),
        {"confidence_scenario": "transparent"},
    )
    assert scenario == "transparent"


def test_embedded_threshold_and_qualification_gate():
    config = {
        "enabled": True,
        "require_qualified": True,
        "default_threshold": 0.95,
        "thresholds": {
            "glare": {"threshold": 0.91, "qualified": True},
            "depth_failure": {"threshold": 0.98, "qualified": False},
        },
    }
    glare = select_confidence_gate(
        config,
        model_variant="glare",
        triggers=("saturated_highlight",),
    )
    assert glare.threshold == 0.91
    assert glare.result_allowed

    failed = select_confidence_gate(
        config,
        model_variant="depth_failure",
        triggers=("raw_depth_valid_ratio_low",),
    )
    assert failed.threshold == 0.98
    assert not failed.result_allowed


def test_depth_failure_uses_observed_validity_band_and_base_fallback():
    config = {
        "enabled": True,
        "require_qualified": True,
        "thresholds": {
            "depth_failure": {"threshold": 0.70, "qualified": False},
            "depth_failure:extreme": {"threshold": 0.55, "qualified": True},
        },
    }
    extreme = select_confidence_gate(
        config,
        model_variant="depth_failure",
        triggers=("raw_depth_valid_ratio_low",),
        raw_depth_valid_ratio=0.04,
    )
    assert extreme.scenario == "depth_failure:extreme"
    assert extreme.threshold == 0.55
    assert extreme.result_allowed

    severe = select_confidence_gate(
        config,
        model_variant="depth_failure",
        triggers=("raw_depth_valid_ratio_low",),
        raw_depth_valid_ratio=0.20,
    )
    assert severe.scenario == "depth_failure:severe"
    assert severe.threshold == 0.70
    assert not severe.result_allowed


def test_loads_calibration_report(tmp_path):
    report = tmp_path / "confidence.json"
    report.write_text(
        json.dumps(
            {
                "policy": {
                    "low_light": {
                        "threshold": 0.87,
                        "qualified": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    gate = select_confidence_gate(
        {
            "enabled": True,
            "calibration_report": str(report),
            "require_qualified": True,
        },
        model_variant="low_light",
        triggers=("low_light",),
    )
    assert gate.scenario == "low_light"
    assert gate.threshold == 0.87
    assert gate.qualified
    assert gate.result_allowed
