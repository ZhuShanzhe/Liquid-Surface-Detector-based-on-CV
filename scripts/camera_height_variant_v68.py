"""Held-out top-view camera-distance perturbation for simulation only."""
from dataclasses import replace

import numpy as np


def scale_camera_height(scene, extra: dict, scale: float):
    if not .8 <= scale <= 1.:
        raise ValueError("camera_height_scale_must_be_in_0p8_to_1")
    if scale == 1.:
        return scene, extra
    bottom = float(extra["interior_bottom_z_m"])
    x, y, z = scene.camera_position_m
    new_z = bottom + scale * (z - bottom)
    updated = replace(scene,
                      camera_position_m=(x * scale, y * scale, new_z),
                      surface_radius_x_m=scene.surface_radius_x_m * scale,
                      surface_radius_y_m=scene.surface_radius_y_m * scale)
    revised = {**extra, "camera_height_scale": float(scale),
               "camera_to_bottom_m": float(np.linalg.norm(
                   [x * scale, y * scale, new_z - bottom]))}
    return updated, revised
