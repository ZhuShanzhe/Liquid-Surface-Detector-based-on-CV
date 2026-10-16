"""No-USB public entry for the fixed all-scene facade."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from liquid_depth.determined_scene_meter import SCENES, DeterminedSceneMeter


def resolve(root, value):
    p = Path(value)
    return p if p.is_absolute() else root/p


def image(path, gray=False):
    path = Path(path)
    a = np.load(path, allow_pickle=False) if path.suffix == '.npy' else cv2.imread(
        str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if a is None:
        raise ValueError(f'image_unreadable:{path}')
    return a


def initialize(path, scene):
    spec = json.loads(path.read_text())
    root = path.parent
    if spec['kind'] == 'empty_geometry':
        return DeterminedSceneMeter.from_empty_geometry(
            json.loads(resolve(root, spec['bundle']).read_text()),
            np.load(resolve(root, spec['dry_depth']), allow_pickle=False),
            max_return_depth_m=spec.get('max_return_depth_m', 10.5))
    if spec['kind'] == 'five_priors':
        with np.load(resolve(root, spec['masks']), allow_pickle=False) as masks_file:
            masks = {ch: masks_file[ch].copy() for ch in ('left', 'right')}
        prior = [{ch: image(resolve(root, p[ch]), True) for ch in ('left', 'right')}
                 for p in spec['prior_images']]
        return DeterminedSceneMeter.from_five_priors(
            json.loads(resolve(root, spec['calibration']).read_text()), masks,
            prior, spec['known_heights_m'], scene=scene)
    raise ValueError('unknown_initialization_kind')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', choices=sorted(SCENES), required=True)
    p.add_argument('--initialization', type=Path, required=True)
    p.add_argument('--depth', type=Path)
    p.add_argument('--rgb', type=Path)
    p.add_argument('--left', type=Path)
    p.add_argument('--right', type=Path)
    p.add_argument('--enable-floating-peak-vision', action='store_true')
    a = p.parse_args()
    if a.scene == 'floating_objects' and a.enable_floating_peak_vision:
        import torch
        torch.set_num_threads(4)
    meter = initialize(a.initialization, a.scene)
    depth = np.load(a.depth, allow_pickle=False) if a.depth else None
    # Non-floating RGB is deliberately not decoded or passed to a model.
    rgb = image(a.rgb) if a.scene == 'floating_objects' and a.enable_floating_peak_vision and a.rgb else None
    stereo = ({'left': image(a.left, True), 'right': image(a.right, True)}
              if a.left and a.right else None)
    print(json.dumps(meter.measure(a.scene, depth_m=depth, rgb_bgr=rgb,
                                  stereo_images=stereo, visual_enabled=a.enable_floating_peak_vision),
                     allow_nan=False))


if __name__ == '__main__':
    main()
