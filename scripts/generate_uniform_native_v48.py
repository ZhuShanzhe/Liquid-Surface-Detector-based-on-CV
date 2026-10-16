"""Blender entry: native noisy depth returns with only scenario dropout disabled.

This generates no truth-filled depth. Range limits, wrong echoes, flying pixels,
full-frame background and front-wall contamination remain in the sensor proxy.
Later evaluation may ONLY delete returns from these arrays to set actual loss.
"""
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
from camera_height_variant_v68 import scale_camera_height
from generate_fullframe_v16 import context_proxy
from generate_other_scenes_v34 import make_scene


def generate_case(root, case, seed, repeats, surface_only_returns=False,
                  sensor_components=None, calibration_fraction=None,
                  camera_height_scale=1.):
    ordinary, ordinary_extra = make_scene(case * 6, seed)
    ordinary, ordinary_extra = scale_camera_height(
        ordinary, ordinary_extra, camera_height_scale)
    ordinary_folder = root / f'{case * 6:05d}'
    ordinary_folder.mkdir(parents=True, exist_ok=False)
    ordinary_meta = gen.scene_metadata(ordinary, ordinary_folder)
    ordinary_meta.update(ordinary_extra, slot=case * 6,
                         empty=ordinary_extra['nominal_liquid_height_m'] == 0)
    (ordinary_folder / 'metadata.json').write_text(json.dumps(ordinary_meta, indent=2))

    scene, extra = make_scene(case * 6 + 5, seed)
    scene, extra = scale_camera_height(scene, extra, camera_height_scale)
    if calibration_fraction is not None:
        if not 0 < calibration_fraction < 1:
            raise ValueError('calibration_fraction_must_be_between_zero_and_one')
        original_height = float(extra['nominal_liquid_height_m'])
        capacity = (scene.container_rim_z_m - scene.container_bottom_z_m -
                    scene.wall_thickness_m)
        known = float(calibration_fraction * capacity)
        shift = known - original_height
        scene = replace(
            scene, seed=scene.seed + int(calibration_fraction * 100000) * 97,
            container_bottom_z_m=scene.container_bottom_z_m - shift,
            container_rim_z_m=scene.container_rim_z_m - shift,
            camera_position_m=(scene.camera_position_m[0],
                               scene.camera_position_m[1],
                               scene.camera_position_m[2] - shift),
            camera_target_m=(scene.camera_target_m[0],
                             scene.camera_target_m[1],
                             scene.camera_target_m[2] - shift))
        extra = {**extra, 'nominal_liquid_height_m': known,
                 'interior_bottom_z_m': -known,
                 'calibration_fraction': calibration_fraction,
                 'calibration_capacity_m': capacity}
    folder = root / f'{case * 6 + 5:05d}'
    folder.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed + (case * 6 + 5) * 104729)
    gen.clear_scene()
    gen.create_environment(scene, rng)
    gen.create_container(scene)
    if extra['nominal_liquid_height_m'] > 0:
        obj = gen.create_liquid(scene)
        bsdf = obj.data.materials[0].node_tree.nodes.get('Principled BSDF')
        gen._set_input(bsdf, 'Transmission Weight', 0.)
        gen._set_input(bsdf, 'Transmission', 0.)
        gen._set_input(bsdf, 'Alpha', 1.)
    gen.create_floating_objects(scene, rng)
    gen.create_camera(scene)
    gen.create_lighting(scene, rng)
    gen.configure_render(scene, argparse.Namespace(engine='eevee', render_samples=16))
    bpy.context.scene.render.filepath = str(folder / 'rgb.png')
    bpy.ops.render.render(write_still=True)
    labels = gen.render_geometric_labels(scene)
    components = {'scenario_dropout': False,
                  'wrong_return': not surface_only_returns,
                  'flying': not surface_only_returns}
    components.update(sensor_components or {})
    first = gen.simulate_raw_depth(scene, labels, components=components)
    full, target, visible, _, _ = context_proxy(
        scene, first['raw_depth_m'], labels['target_depth_m'], labels['mask'],
        front_wall_returns=not surface_only_returns)
    clean_surface = visible.astype(bool) & np.isclose(
        full, first['raw_depth_m'], rtol=0, atol=.00051)
    np.savez_compressed(folder / 'geometry.npz', target=target, mask=visible,
                        analytic_mask=labels['mask'])
    meta = gen.scene_metadata(scene, folder)
    meta.update(extra, slot=case * 6 + 5,
                empty=extra['nominal_liquid_height_m'] == 0,
                opaque_surface=True, render_engine='eevee',
                background_depth_present=True,
                frontwall_sensor_return_mode=('surface_only' if surface_only_returns
                                              else 'mixed_frontwall'),
                wrong_return_component_enabled=not surface_only_returns,
                flying_component_enabled=not surface_only_returns,
                sensor_limitations='native parametric sensor proxy; scenario dropout disabled for controlled <=50% loss audit',
                effective_max_return_depth_m=10.0,
                sensor_component_switches=components)
    (folder / 'metadata.json').write_text(json.dumps(meta, indent=2))
    records = []
    for repeat in range(repeats):
        sensor = gen.simulate_raw_depth(replace(scene, seed=scene.seed + repeat * 79907),
            labels, components=components)
        raw = np.where(clean_surface, sensor['raw_depth_m'], full).astype(np.float32)
        # Apply the same output-unit quantization and 10 m SDK-style cap to
        # every repeat; the first context frame already uses this contract.
        raw = (np.round(raw / .001) * .001).astype(np.float32)
        raw = np.where(np.isfinite(raw) & (raw >= .1) & (raw <= 10.),
                       raw, 0.).astype(np.float32)
        filename = f'raw_native_{repeat:02d}.npy'
        np.save(folder / filename, raw, allow_pickle=False)
        records.append({'folder': str(folder), 'raw_path': str(folder / filename),
                        'case': case, 'repeat': repeat})
    (folder / 'frames.json').write_text(json.dumps(records, indent=2))
    surface_pixels = int((visible > 0).sum())
    return {'case': case, 'surface_pixels': surface_pixels,
            'base_loss_pct': [100 * (1 - np.count_nonzero(
                (visible > 0) & (np.load(r['raw_path']) > 0)) / max(surface_pixels, 1))
                              for r in records]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--cases', type=int, default=99)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--surface-only-returns', action='store_true')
    parser.add_argument('--case-ids', type=str)
    parser.add_argument('--disable-depth-noise', action='store_true')
    parser.add_argument('--disable-disparity-noise', action='store_true')
    parser.add_argument('--disable-quantization', action='store_true')
    parser.add_argument('--calibration-fraction', type=float)
    parser.add_argument('--camera-height-scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=202610124901)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.root.mkdir(parents=True, exist_ok=True)
    selected = ([int(x) for x in args.case_ids.split(',')] if args.case_ids
                else list(range(args.cases)))
    if len(selected) != len(set(selected)) or any(x < 0 or x >= args.cases for x in selected):
        raise ValueError('unique_case_ids_within_cases_required')
    switches = {name: False for name, disabled in (
        ('depth_noise', args.disable_depth_noise),
        ('disparity_noise', args.disable_disparity_noise),
        ('quantization', args.disable_quantization)) if disabled}
    for case in selected[args.worker::args.workers]:
        print(json.dumps(generate_case(args.root, case, args.seed, args.repeats,
                                       args.surface_only_returns, switches,
                                       args.calibration_fraction,
                                       args.camera_height_scale)), flush=True)


if __name__ == '__main__':
    main()
