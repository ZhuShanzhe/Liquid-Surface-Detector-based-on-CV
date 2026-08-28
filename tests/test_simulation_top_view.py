from __future__ import annotations

import numpy as np

from liquid_depth.simulation import CONTAINER_SHAPES, SENSOR_FAMILIES, sample_scene


def _elevation_deg(scene) -> float:
    position = np.asarray(scene.camera_position_m)
    return float(np.degrees(np.arcsin(position[2] / np.linalg.norm(position))))


def test_industrial_top_sampling_covers_top_views_and_hardware_variants():
    scenes = [
        sample_scene(index, seed=9, width=96, height=54, camera_profile="industrial_top")
        for index in range(105)
    ]
    elevations = np.asarray([_elevation_deg(scene) for scene in scenes])
    assert elevations.min() >= 42.0
    assert float(np.mean(elevations >= 60.0)) >= 0.80
    assert elevations.max() <= 87.0
    assert {scene.container_shape for scene in scenes} == set(CONTAINER_SHAPES)
    assert {scene.sensor_family for scene in scenes} == set(SENSOR_FAMILIES)
    near_vertical = sample_scene(3, seed=9, camera_profile="near_vertical")
    assert 72.0 <= _elevation_deg(near_vertical) <= 88.5
    assert near_vertical.camera_profile == "near_vertical"
