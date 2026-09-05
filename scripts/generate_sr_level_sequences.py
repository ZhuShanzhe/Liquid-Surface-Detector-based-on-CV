#!/usr/bin/env python3
"""Native HR RGB at changed liquid levels for metrological SR evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_synthetic_liquid as gen
from generate_range_sequences import render


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in (11231, 11333):
        for distance, level in ((1.0, 0.3), (3.0, 0.1), (6.0, 0.3)):
            base = gen.sample_scene(
                3,
                seed=seed,
                width=1280,
                height=720,
                min_distance_m=1,
                max_distance_m=1.1,
                camera_profile="industrial_top",
            )
            direction = np.array([0.1, -0.35, 1.1])
            direction /= np.linalg.norm(direction)
            radius = 0.23 * distance
            base = replace(
                base,
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
                liquid_color_rgb=(0.12, 0.35, 0.2),
                sensor_family="tof",
            )
            for i, shift in enumerate((0.0, 0.005, -0.005, 0.01, -0.01, 0.02, -0.02)):
                path = args.output / f"{seed}_d{distance:g}" / f"{i:02d}"
                render(path, base, shift, seed, i)
                rows.append(
                    {
                        "sequence": f"{seed}_d{distance:g}",
                        "index": i,
                        "path": str(path),
                        "standoff_m": distance,
                        "known_level_m": level,
                        "truth_m": level + shift,
                        "bottom_world_m": -level,
                        "radius_x_m": radius,
                        "radius_y_m": radius * 0.8,
                    }
                )
            print(json.dumps({"seed": seed, "distance": distance, "done": True}), flush=True)
    (args.output / "sequences.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
