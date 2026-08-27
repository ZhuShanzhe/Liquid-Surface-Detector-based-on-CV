import json

import numpy as np
import pytest

from liquid_depth.scenario_policy import (
    ComplexScenePolicy,
    SceneSignals,
    load_scene_context,
    measure_scene_signals,
)


def _normal_signals() -> SceneSignals:
    return SceneSignals(
        raw_depth_valid_ratio=0.95,
        luma_p50=0.5,
        dark_pixel_ratio=0.01,
        saturated_pixel_ratio=0.01,
        dynamic_range=0.4,
    )


def test_policy_modes_and_missing_specialist_are_explicit():
    off = ComplexScenePolicy({"mode": "off"})
    assert not off.decide(_normal_signals()).requested

    always = ComplexScenePolicy(
        {
            "mode": "always",
            "missing_model_policy": "reject",
        }
    )
    unavailable = always.decide(_normal_signals(), {"model_variant": "transparent_multilayer"})
    assert unavailable.requested
    assert not unavailable.activated
    assert not unavailable.result_allowed

    available = ComplexScenePolicy(
        {"mode": "always"},
        available_variants={"transparent_multilayer"},
    ).decide(_normal_signals(), {"model_variant": "transparent_multilayer"})
    assert available.activated
    assert available.result_allowed


def test_auto_policy_selects_variants_and_holds_without_overriding_forced_off():
    policy = ComplexScenePolicy(
        {
            "mode": "auto",
            "hold_frames": 2,
            "auto": {"raw_depth_valid_ratio_below": 0.5},
        },
        available_variants={"depth_failure"},
    )
    poor_depth = SceneSignals(
        raw_depth_valid_ratio=0.1,
        luma_p50=0.5,
        dark_pixel_ratio=0.0,
        saturated_pixel_ratio=0.0,
        dynamic_range=0.4,
    )
    first = policy.decide(poor_depth)
    assert first.activated
    assert first.model_variant == "depth_failure"

    held = policy.decide(_normal_signals())
    assert held.activated
    assert held.held_by_hysteresis

    disabled = policy.decide(_normal_signals(), {"force_complex_model": False})
    assert not disabled.requested
    assert not disabled.activated


def test_operator_hints_have_priority_over_optical_fallback():
    policy = ComplexScenePolicy(
        {"mode": "auto"},
        available_variants={"transparent_multilayer", "glare", "low_light"},
    )
    transparent = policy.decide(
        _normal_signals(),
        {
            "transparent_container": True,
            "transparent_liquid": True,
        },
    )
    assert transparent.model_variant == "transparent_multilayer"
    assert transparent.activated


def test_scene_signal_measurement_and_frame_context(tmp_path):
    rgb = np.full((20, 30, 3), 128, dtype=np.uint8)
    depth = np.full((20, 30), 1000, dtype=np.uint16)
    depth[:, :3] = 0
    signals = measure_scene_signals(
        rgb,
        depth,
        depth_scale_to_m=0.001,
        max_depth_m=2.0,
    )
    assert signals.raw_depth_valid_ratio == pytest.approx(0.9)
    assert signals.luma_p50 == pytest.approx(128 / 255, abs=1e-6)

    context = {"transparent_container": True, "model_variant": "transparent_multilayer"}
    (tmp_path / "scene_context.json").write_text(json.dumps(context), encoding="utf-8")
    assert load_scene_context(tmp_path) == context
