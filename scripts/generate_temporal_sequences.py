#!/usr/bin/env python3
"""Render physically continuous top-view RGB-D sequences in Blender."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9107, 9212, 9323])
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        for motion in ["static", "rising", "step", "moving", "long_static"]:
            base = gen.sample_scene(
                3 if seed % 2 else 0,
                seed=seed,
                width=320,
                height=180,
                min_distance_m=1.0,
                max_distance_m=1.1,
                camera_profile="industrial_top",
            )
            base = replace(
                base,
                scenario="ordinary",
                surface_radius_x_m=0.25,
                surface_radius_y_m=0.20,
                container_bottom_z_m=-0.305,
                container_rim_z_m=0.15,
                wall_thickness_m=0.005,
                camera_position_m=(0.10, -0.35, 1.10),
                camera_target_m=(0.0, 0.0, 0.0),
                tilt_x=0.0,
                tilt_y=0.0,
                wave_amplitude_m=0.0,
                floating_object_count=0,
                corruption_severity=0.1,
                liquid_turbidity=0.7,
                liquid_color_rgb=(0.12, 0.35, 0.20),
            )
            for i in range(120 if motion == "long_static" else args.frames):
                shift = 0.0
                if motion == "rising":
                    shift = max(i - 7, 0) * 0.002
                if motion == "step" and i >= 12:
                    shift = 0.06
                dx = 0.10 * np.sin(i / 8.0) if motion == "moving" else 0.0
                # Translate the coordinate origin with the liquid surface. The
                # physical container, light and environment remain fixed.
                scene = replace(
                    base,
                    camera_position_m=(0.10 + dx, -0.35, 1.10 - shift),
                    camera_target_m=(0.0, 0.0, -shift),
                    container_bottom_z_m=base.container_bottom_z_m - shift,
                    container_rim_z_m=base.container_rim_z_m - shift,
                )
                folder = args.output / f"{seed}_{motion}" / f"{i:04d}"
                folder.mkdir(parents=True, exist_ok=True)
                if not (folder / "frame.npz").exists():
                    gen.clear_scene()
                    rng = random.Random(seed)
                    gen.create_environment(base, rng)
                    gen.create_container(scene)
                    gen.create_liquid(scene)
                    gen.create_camera(scene)
                    gen.create_lighting(base, rng)
                    for obj in list(bpy.context.scene.objects):
                        if (
                            obj.name not in ("transparent_container", "liquid_volume")
                            and obj.type != "CAMERA"
                        ):
                            obj.location.z -= shift
                    gen.configure_render(scene, argparse.Namespace(engine="eevee", render_samples=16))
                    bpy.context.scene.render.filepath = str(folder / "rgb.png")
                    bpy.ops.render.render(write_still=True)
                    labels = gen.render_geometric_labels(scene)
                    sensor = gen.simulate_raw_depth(scene, labels)
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
                records.append(
                    {
                        "path": str(folder),
                        "sequence": f"{seed}_{motion}",
                        "seed": seed,
                        "motion": motion,
                        "index": i,
                        "truth_m": 0.3 + shift,
                        "bottom_world_m": -0.3,
                        "fps": 10.0,
                    }
                )
                print(json.dumps({"sequence": f"{seed}_{motion}", "frame": i}), flush=True)
    (args.output / "sequences.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
