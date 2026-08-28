from __future__ import annotations

import json

import numpy as np

from liquid_depth.simulation import (
    SCENARIOS,
    build_manifest,
    render_geometric_labels,
    sample_scene,
    scene_metadata,
    simulate_raw_depth,
    write_sample_arrays,
)


def test_scene_sampling_is_deterministic_and_log_range_aware():
    first = sample_scene(11, seed=7, width=96, height=54)
    second = sample_scene(11, seed=7, width=96, height=54)
    assert first == second
    scenes = [sample_scene(index, seed=7, width=96, height=54) for index in range(100)]
    distances = np.asarray([np.linalg.norm(scene.camera_position_m) for scene in scenes])
    assert distances.min() >= 0.1
    assert distances.max() <= 10.0
    assert {scene.scenario for scene in scenes} == set(SCENARIOS)
    assert {scene.split for scene in scenes} == {"train", "val", "test"}


def test_geometric_labels_and_sensor_failures_are_consistent():
    coverage = {}
    for index, scenario in enumerate(SCENARIOS):
        scene = sample_scene(index, seed=2026, width=128, height=72)
        assert scene.scenario == scenario
        labels = render_geometric_labels(scene)
        inside = labels["mask"] > 0
        assert inside.sum() > 50
        assert np.all(labels["target_depth_m"][inside] > 0)
        norms = np.linalg.norm(labels["normal_camera"][inside], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=2e-3)
        depths, valid = labels["layer_depths_m"], labels["layer_valid"] > 0
        adjacent = valid[1:] & valid[:-1]
        assert np.all((depths[1:] - depths[:-1])[adjacent] >= -1e-5)
        sensor = simulate_raw_depth(scene, labels)
        coverage[scenario] = float(((sensor["raw_depth_m"] > 0) & inside).sum() / inside.sum())
        assert sensor["uncertainty"].shape == labels["target_depth_m"].shape
        assert sensor["simulated_error_type"].dtype == np.uint8
        assert sensor["raw_reliable_mask"].shape == labels["target_depth_m"].shape
        assert sensor["incidence_cosine"].shape == labels["target_depth_m"].shape
    assert coverage["depth_failure"] < coverage["ordinary"]
    assert coverage["compound"] < coverage["ordinary"]


def test_sample_contract_builds_canonical_manifest(tmp_path):
    root = tmp_path / "synthetic"
    sample_dir = root / "samples" / "00000000"
    scene = sample_scene(0, seed=3, width=64, height=36)
    labels = render_geometric_labels(scene)
    sensor = simulate_raw_depth(scene, labels)
    write_sample_arrays(sample_dir, labels, sensor)
    (sample_dir / "rgb.png").write_bytes(b"placeholder")
    metadata = scene_metadata(scene, sample_dir)
    (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    manifest = root / "manifest.csv"
    assert build_manifest(root, manifest) == 1
    text = manifest.read_text(encoding="utf-8")
    assert "rgb_path,raw_depth_path,target_depth_path" in text
    assert "active_stereo_proxy_v2" in text
