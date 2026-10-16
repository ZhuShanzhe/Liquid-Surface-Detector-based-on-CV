#!/usr/bin/env python3
"""Offline event/proximity audit; truth never enters deployed inference."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import fields, replace
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
# Blender's bundled NumPy reproduces the original quantization decisions.
# Load these two dependency-free modules without the application __init__,
# which imports OpenCV. This does not alter the normal application process.
if '--' in sys.argv:
    import types
    package = types.ModuleType('liquid_depth')
    package.__path__ = [str(Path(__file__).resolve().parents[1]/'src'/'liquid_depth')]
    sys.modules['liquid_depth'] = package
from liquid_depth.simulation import SyntheticScene, render_geometric_labels, simulate_raw_depth
from liquid_depth.liquid_height_metrics import HeightCalibration, world_points, reference_statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows = list(csv.DictReader(open(args.manifest)))
    records = []
    for row in rows:
        p = Path(row['raw_depth_path']).parent
        meta = json.loads((p/'provenance.json').read_text())['scene']
        scene = SyntheticScene(**{f.name: meta[f.name] for f in fields(SyntheticScene)})
        labels = render_geometric_labels(scene)
        replay = simulate_raw_depth(scene, labels)
        roi = np.load(row['roi_raw_path'])
        if not np.allclose(roi, replay['raw_depth_m'], rtol=1e-5, atol=1e-7):
            raise AssertionError(f'Sensor replay mismatch: {p}')
        # Changing liquid IOR while preserving geometry and noise is a causal
        # simulator ablation, NOT a material estimate from observations.
        changed = simulate_raw_depth(replace(scene, liquid_ior=1.48), labels)['raw_depth_m']
        cal = HeightCalibration.from_simulation(meta)
        target = np.load(row['target_depth_path'])
        mask = np.load(row['mask_path']) > 0
        raw = np.load(row['raw_depth_path'])
        event = replay['simulated_error_type']
        valid = mask & (raw > 0) & np.isfinite(raw)
        xyz = world_points(raw, cal)
        truth_xyz = world_points(target, cal)
        dh = xyz[..., 2] - truth_xyz[..., 2]
        h = reference_statistics(meta, cal)['mean_m']
        tol = max(.005, .02*h)
        near = valid & (np.abs(dh) <= tol)
        base = mask & (event == 0) & (roi > 0)
        final_base = base & valid & (np.abs(raw-roi) <= .000501)
        n = max(int(mask.sum()), 1)
        def fraction(m):
            return float(np.count_nonzero(m & mask)/n)
        def median(a):
            return float(np.median(a)) if len(a) else None
        records.append(dict(sequence_id=row['sequence_id'], scenario=row['scenario'],
            sensor=scene.sensor_family, liquid_height_m=h,
            camera_depth_m=median(target[mask]), target_pixels=int(mask.sum()),
            valid_fraction=fraction(valid), invalid_fraction=1-fraction(valid),
            precontext_base_event_fraction=fraction(base),
            precontext_wrong_layer_fraction=fraction(event == 2),
            precontext_flying_fraction=fraction(event == 3),
            precontext_drop_fraction=fraction((event == 1) | (event == 4) | (event == 5)),
            final_preserved_base_fraction=fraction(final_base),
            within_height_tolerance_fraction=fraction(near),
            within_height_tolerance_among_valid=float(near.sum()/max(valid.sum(), 1)),
            base_event_vertical_bias_mm=median(dh[final_base]*1000),
            base_event_vertical_abs_error_mm=median(np.abs(dh[final_base])*1000),
            raw_unchanged_when_liquid_ior_changed=bool(np.array_equal(changed, replay['raw_depth_m']))))
    aggregate = {}
    for key in ('scenario', 'sensor'):
        groups = defaultdict(list)
        for r in records:
            groups[r[key]].append(r)
        aggregate[key] = {}
        for name, group in groups.items():
            values = {}
            for metric in ('valid_fraction', 'precontext_base_event_fraction',
                'precontext_wrong_layer_fraction', 'precontext_flying_fraction',
                'precontext_drop_fraction', 'final_preserved_base_fraction',
                'within_height_tolerance_fraction', 'within_height_tolerance_among_valid',
                'base_event_vertical_bias_mm', 'base_event_vertical_abs_error_mm'):
                a = [r[metric] for r in group if r[metric] is not None]
                values[metric] = float(np.mean(a)) if a else None
            aggregate[key][name] = dict(frames=len(group), frame_unweighted_mean=values)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(
        offline_ground_truth_audit_only=True, surface_origin_is_simulator_event_not_photon_truth=True,
        base_event_includes_systematic_bias_and_tof_multipath=True,
        synthetic_ior_ablation_identical_count=sum(r['raw_unchanged_when_liquid_ior_changed'] for r in records),
        frames=len(records), aggregate=aggregate, records=records), indent=2))
    print(json.dumps(aggregate['scenario'], indent=2))


if __name__ == '__main__':
    main()
