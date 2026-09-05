#!/usr/bin/env python3
"""Long RGB-D sequences with analytic ground truth and parametrically moving waves."""

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
from generate_range_sequences import CASES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[11017, 11119])
    p.add_argument("--frames", type=int, default=120)
    args = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.frames < 40:
        raise ValueError("Long sequence requires at least 40 frames")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        rng = random.Random(seed)
        direction = np.array([rng.uniform(0.04, 0.14), rng.uniform(-0.38, -0.22), 1.1])
        direction /= np.linalg.norm(direction)
        radius_ratio = rng.uniform(0.21, 0.25)
        color = (rng.uniform(0.08, 0.16), rng.uniform(0.30, 0.40), rng.uniform(0.16, 0.24))
        cases = [(d, h, "static") for d, h in CASES] + [
            (0.5, 0.3, "waves"),
            (1.0, 0.3, "waves"),
            (2.0, 1.0, "waves"),
            (6.0, 0.3, "waves"),
        ]
        for case, (distance, level, motion) in enumerate(cases):
            template = gen.sample_scene(
                3,
                seed=seed,
                width=320,
                height=180,
                min_distance_m=1,
                max_distance_m=1.1,
                camera_profile="industrial_top",
            )
            base = replace(
                template,
                scenario="ordinary",
                surface_radius_x_m=radius_ratio * distance,
                surface_radius_y_m=radius_ratio * distance * 0.8,
                container_bottom_z_m=-level - 0.005,
                container_rim_z_m=max(0.12 * distance, 0.12),
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
                liquid_color_rgb=color,
            )
            states = {}
            # Flat first 20 frames initialize memory. Subsequent wave states
            # are a smooth periodic parameterization, not a CFD fluid solver.
            for frame in range(args.frames):
                phase_index = (frame - 20) % 20 if motion == "waves" and frame >= 20 else -1
                if phase_index not in states:
                    wave_phase = 1.6 * np.sin(2 * np.pi * phase_index / 20) if phase_index >= 0 else 0.0
                    amplitude = min(0.08, 0.2 * level) if phase_index >= 0 else 0.0
                    scene = replace(
                        base,
                        wave_amplitude_m=amplitude,
                        wave_frequency_x=2.2,
                        wave_frequency_y=1.7,
                        wave_phase=float(wave_phase),
                    )
                    path = args.output / "states" / f"{seed}_{case:02d}_{phase_index + 1:02d}"
                    path.mkdir(parents=True, exist_ok=True)
                    labels = gen.render_geometric_labels(scene)
                    if not (path / "rgb.png").exists():
                        gen.clear_scene()
                        gen.create_environment(scene, random.Random(seed))
                        gen.create_container(scene)
                        gen.create_liquid(scene)
                        gen.create_camera(scene)
                        gen.create_lighting(scene, random.Random(seed))
                        gen.configure_render(scene, argparse.Namespace(engine="eevee", render_samples=16))
                        bpy.context.scene.render.filepath = str(path / "rgb.png")
                        bpy.ops.render.render(write_still=True)
                    gx, gy = np.meshgrid(
                        np.linspace(-scene.surface_radius_x_m, scene.surface_radius_x_m, 81),
                        np.linspace(-scene.surface_radius_y_m, scene.surface_radius_y_m, 81),
                    )
                    inside = (gx / scene.surface_radius_x_m) ** 2 + (gy / scene.surface_radius_y_m) ** 2 <= 1
                    xy = np.column_stack((gx[inside], gy[inside]))
                    truth = gen.surface_height_and_gradient(scene, xy[:, 0], xy[:, 1])[0] + level
                    np.savez_compressed(
                        path / "geometry.npz",
                        **labels,
                        intrinsics=gen._simulation.camera_intrinsics(scene),
                        camera_to_world=gen.camera_to_world(scene),
                        area_xy=xy,
                        area_truth_m=truth,
                    )
                    states[phase_index] = (scene, labels, path)
                scene, labels, path = states[phase_index]
                for si, sensor in enumerate(("active_stereo", "structured_light", "tof")):
                    name = f"{seed}_d{distance:g}_h{level:g}_{motion}_{sensor}"
                    folder = args.output / "depth" / name
                    folder.mkdir(parents=True, exist_ok=True)
                    output = folder / f"{frame:04d}.npz"
                    if not output.exists():
                        raw = gen.simulate_raw_depth(
                            replace(scene, sensor_family=sensor, index=case * 10000 + frame * 17 + si * 1000),
                            labels,
                        )
                        np.savez_compressed(output, depth=raw["raw_depth_m"])
                    rows.append(
                        {
                            "sequence": name,
                            "seed": seed,
                            "index": frame,
                            "sensor": sensor,
                            "standoff_m": distance,
                            "truth_m": level,
                            "motion": motion,
                            "state": str(path),
                            "depth_path": str(output),
                            "bottom_world_m": -level,
                            "radius_x_m": scene.surface_radius_x_m,
                            "radius_y_m": scene.surface_radius_y_m,
                            "fps": 10,
                        }
                    )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "distance": distance,
                        "level": level,
                        "motion": motion,
                        "frames": args.frames,
                        "states": len(states),
                    }
                ),
                flush=True,
            )
    (args.output / "sequences.json").write_text(json.dumps(rows, indent=2))
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "frames_per_sequence": args.frames,
                "calibration_frames": 20,
                "wave_truth": "uniform container footprint 81x81 grid; parametric smooth surface, not CFD",
                "new_sensor_noise_every_frame": True,
                "rendered_states_reused_when_exactly_identical": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
