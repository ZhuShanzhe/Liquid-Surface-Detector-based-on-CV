#!/usr/bin/env python3
"""Range audit and loss/reacquisition sequences; no severe-hole recovery tuning."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_synthetic_liquid as gen

CASES = [
    (0.1, 0.1),
    (0.2, 0.1),
    (0.5, 0.3),
    (1.0, 0.3),
    (2.0, 1.0),
    (4.0, 3.0),
    (6.0, 5.0),
    (8.0, 8.0),
    (10.0, 10.0),
    (1.0, 3.0),
    (3.0, 0.1),
    (6.0, 0.3),
]


def render(folder, base, shift, seed, index):
    position = np.array(base.camera_position_m)
    position[2] -= shift
    scene = replace(
        base,
        camera_position_m=tuple(position),
        camera_target_m=(0.0, 0.0, -shift),
        container_bottom_z_m=base.container_bottom_z_m - shift,
        container_rim_z_m=base.container_rim_z_m - shift,
    )
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "frame.npz").exists():
        gen.clear_scene()
        rng = random.Random(seed)
        gen.create_environment(base, rng)
        gen.create_container(scene)
        gen.create_liquid(scene)
        gen.create_camera(scene)
        gen.create_lighting(base, rng)
        for obj in bpy.context.scene.objects:
            if obj.name not in ("transparent_container", "liquid_volume") and obj.type != "CAMERA":
                obj.location.z -= shift
        gen.configure_render(scene, argparse.Namespace(engine="eevee", render_samples=16))
        bpy.context.scene.render.filepath = str(folder / "rgb.png")
        bpy.ops.render.render(write_still=True)
        labels = gen.render_geometric_labels(scene)
        sensor = gen.simulate_raw_depth(replace(scene, index=scene.index + index * 16), labels)
        transform = gen.camera_to_world(scene)
        transform[2, 3] += shift
        np.savez_compressed(
            folder / "frame.npz",
            depth=sensor["raw_depth_m"],
            truth_depth=labels["target_depth_m"],
            truth_mask=labels["mask"],
            camera_to_world=transform,
            intrinsics=gen._simulation.camera_intrinsics(scene),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[10607, 10709])
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    rows = []
    for seed in args.seeds:
        for case, (distance, level) in enumerate(CASES):
            for sensor in ("active_stereo", "structured_light", "tof"):
                template = gen.sample_scene(
                    3,
                    seed=seed,
                    width=320,
                    height=180,
                    min_distance_m=1.0,
                    max_distance_m=1.1,
                    camera_profile="industrial_top",
                )
                direction = np.array([0.10, -0.35, 1.10])
                direction /= np.linalg.norm(direction)
                radius = 0.23 * distance
                base = replace(
                    template,
                    scenario="ordinary",
                    surface_radius_x_m=radius,
                    surface_radius_y_m=radius * 0.8,
                    container_bottom_z_m=-level - 0.005,
                    container_rim_z_m=0.12 * distance,
                    wall_thickness_m=0.005,
                    camera_position_m=tuple(direction * distance),
                    camera_target_m=(0.0, 0.0, 0.0),
                    tilt_x=0.0,
                    tilt_y=0.0,
                    wave_amplitude_m=0.0,
                    floating_object_count=0,
                    corruption_severity=0.1,
                    liquid_turbidity=0.7,
                    container_taper_ratio=1.0,
                    liquid_color_rgb=(0.12, 0.35, 0.20),
                    sensor_family=sensor,
                )
                sequence = f"{seed}_d{distance:g}_h{level:g}_{sensor}"
                for index in range(16):
                    folder = args.output / sequence / f"{index:04d}"
                    render(folder, base, 0.0, seed, index)
                    rows.append(
                        {
                            "path": str(folder),
                            "sequence": sequence,
                            "seed": seed,
                            "index": index,
                            "motion": "static",
                            "sensor": sensor,
                            "standoff_m": distance,
                            "truth_m": level,
                            "bottom_world_m": -level,
                            "radius_x_m": radius,
                            "radius_y_m": radius * 0.8,
                            "fps": 10.0,
                        }
                    )
                print(json.dumps({"sequence": sequence, "done": True}), flush=True)
        # Six seconds without depth; level changes during the outage. Only
        # post-outage reacquisition is under test, not filling missing frames.
        distance, level = 1.0, 0.3
        base = replace(
            base,
            sensor_family="tof",
            surface_radius_x_m=0.23,
            surface_radius_y_m=0.184,
            container_bottom_z_m=-0.305,
            container_rim_z_m=0.12,
            camera_position_m=tuple(direction),
            camera_target_m=(0.0, 0.0, 0.0),
        )
        for motion in ("recovery_static", "recovery_changed"):
            sequence = f"{seed}_{motion}"
            for index in range(80):
                shift = 0.04 if motion == "recovery_changed" and index >= 25 else 0.0
                folder = args.output / sequence / f"{index:04d}"
                render(folder, base, shift, seed, index)
                rows.append(
                    {
                        "path": str(folder),
                        "sequence": sequence,
                        "seed": seed,
                        "index": index,
                        "motion": motion,
                        "sensor": "tof",
                        "standoff_m": distance,
                        "truth_m": level + shift,
                        "bottom_world_m": -level,
                        "radius_x_m": 0.23,
                        "radius_y_m": 0.184,
                        "fps": 10.0,
                    }
                )
            print(json.dumps({"sequence": sequence, "done": True}), flush=True)
    (args.output / "sequences.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
