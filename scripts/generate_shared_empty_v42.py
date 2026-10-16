"""Render one independent dry RGB-D observation for each V34 scene geometry.

The wet scene files remain untouched. This uses the existing full-frame ray-cast
sensor proxy, with no liquid object, so later estimators receive only a dry frame.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_synthetic_liquid as gen
from camera_height_variant_v68 import scale_camera_height
from generate_fullframe_v16 import context_proxy
from generate_other_scenes_v34 import make_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=202610120341)
    parser.add_argument('--cases', type=int, default=99)
    parser.add_argument('--camera-height-scale', type=float, default=1.0)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    for case in range(args.worker, args.cases, args.workers):
        scene, extra = make_scene(case * 6, args.seed)
        scene, _ = scale_camera_height(scene, extra, args.camera_height_scale)
        folder = args.output / f'{case:04d}'
        folder.mkdir(parents=True, exist_ok=False)
        rng = random.Random(args.seed + case * 104729 + 47)
        gen.clear_scene()
        gen.create_environment(scene, rng)
        gen.create_container(scene)
        gen.create_camera(scene)
        gen.create_lighting(scene, rng)
        dry, _, _, _, _ = context_proxy(
            scene,
            np.zeros((scene.height, scene.width), np.float32),
            np.zeros((scene.height, scene.width), np.float32),
            np.zeros((scene.height, scene.width), np.uint8),
        )
        np.save(folder / 'dry_depth_m.npy', dry, allow_pickle=False)
        (folder / 'metadata.json').write_text(json.dumps({
            'case': case, 'source_slot': case * 6, 'one_dry_observation': True,
            'contains_liquid': False, 'sensor_proxy': 'v16_fullframe_ray_cast',
        }, indent=2))
        print(json.dumps({'case': case, 'valid_pixels': int((dry > 0).sum())}), flush=True)


if __name__ == '__main__':
    main()
