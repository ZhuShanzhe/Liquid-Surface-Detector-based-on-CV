#!/usr/bin/env python3
"""Research calibration CLI: extract measurements, fit priors, infer unknown H.

Not connected to the production GUI. A photo alone is insufficient: preserve
its synchronized raw depth/correspondences, intrinsics and empty reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from liquid_depth.stereo_refraction_v22 import StereoMaterialPrior, fit_stereo_prior, infer_stereo_height
from liquid_depth.homogeneous_refraction_v22 import MaterialPrior, fit_material_prior, infer_height


def load_observation(path, stereo):
    with np.load(path, allow_pickle=False) as a:
        expected = {'value_m', 'qx' if stereo else 'q2'}
        if set(a.files) != expected:
            raise ValueError('Observation must contain only ' + ', '.join(sorted(expected)))
        return {k: a[k].copy() for k in expected}


def extract(raw, empty, roi, intrinsics, stereo):
    """Static perpendicular camera, flat bottom, central epipolar band for stereo.

    ROI is a human-calibrated bottom region in an empty reference, NOT a current
    ground-truth liquid mask. Vendor depth must already be metric axial depth.
    Regions are spatial bins, not guaranteed independent statistical samples.
    """
    if raw.ndim != 2 or empty.shape != raw.shape or roi.shape != raw.shape:
        raise ValueError('Aligned metric raw/empty depth and calibrated ROI required')
    k = np.asarray(intrinsics, float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Metric pinhole intrinsics required')
    valid_empty = roi & np.isfinite(empty) & (empty > 0)
    if valid_empty.sum() < 256:
        raise ValueError('Insufficient empty-reference bottom support')
    bottom = float(np.median(empty[valid_empty]))
    if np.percentile(abs(empty[valid_empty]-bottom), 90) > max(.005, .005*bottom):
        raise ValueError('Bottom not perpendicular/flat enough for this prototype')
    yy, xx = np.indices(raw.shape)
    qx = (xx-k[0, 2])/k[0, 0]
    qy = (yy-k[1, 2])/k[1, 1]
    good = valid_empty & np.isfinite(raw) & (raw > 0)
    values, rays = [], []
    if stereo:
        good &= abs(qy) <= .01
        for ids in np.array_split(np.arange(raw.shape[1]), 64):
            mask = good[:, ids]
            if mask.sum() >= 4:
                values.append(float(np.median((bottom-raw[:, ids])[mask])))
                rays.append(float(np.median(qx[:, ids][mask])))
    else:
        for row_ids in np.array_split(np.arange(raw.shape[0]), 8):
            for col_ids in np.array_split(np.arange(raw.shape[1]), 8):
                region = np.ix_(row_ids, col_ids)
                mask = good[region]
                if mask.sum() >= 4:
                    values.append(float(np.median((raw[region]-bottom)[mask])))
                    rays.append(float(np.median((qx[region]**2+qy[region]**2)[mask])))
    if len(values) < 16:
        raise ValueError('Insufficient measured regions, not enough bottom evidence')
    return {'value_m': np.array(values), 'qx' if stereo else 'q2': np.array(rays)}, bottom


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    subs = ap.add_subparsers(dest='command', required=True)
    e = subs.add_parser('extract')
    for name in ('raw', 'empty', 'roi', 'intrinsics', 'output'):
        e.add_argument('--'+name, required=True)
    e.add_argument('--sensor-family', required=True, choices=('active_stereo', 'structured_light', 'tof'))
    c = subs.add_parser('calibrate')
    c.add_argument('--spec', required=True)
    c.add_argument('--output', required=True)
    i = subs.add_parser('infer')
    for name in ('prior', 'observations', 'material-id', 'rig-id', 'output'):
        i.add_argument('--'+name, required=True)
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.command == 'extract':
        if output.suffix != '.npz':
            raise ValueError('Output must end in .npz')
        obs, bottom = extract(np.load(args.raw), np.load(args.empty), np.load(args.roi)>0,
                              json.loads(Path(args.intrinsics).read_text()), args.sensor_family != 'tof')
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, **obs)
        print(json.dumps(dict(bottom_distance_m=bottom, regions=len(obs['value_m']),
            research_only=True, bottom_path_identity_verified=False)))
        return
    if args.command == 'calibrate':
        spec_path = Path(args.spec)
        spec = json.loads(spec_path.read_text())
        if spec['geometry_model'] != 'fixed_camera_horizontal_bottom_flat_liquid':
            raise ValueError('This prototype does not support arbitrary moving/tilted geometry')
        family = spec['sensor_family']
        if family not in ('active_stereo', 'structured_light', 'tof'):
            raise ValueError('Unsupported sensor family')
        stereo = family != 'tof'
        frames, heights, provenance = [], [], []
        for entry in spec['frames']:
            path = (spec_path.parent/entry['observations_path']).resolve()
            frames.append(load_observation(path, stereo))
            heights.append(float(entry['known_liquid_height_m']))
            provenance.append(dict(observations=str(path), sha256=digest(path),
                                   known_liquid_height_m=heights[-1]))
        if stereo:
            prior = fit_stereo_prior(frames, heights, sensor_family=family,
                bottom_m=spec['bottom_distance_m'], baseline_m=spec['baseline_m'],
                sigma_m=spec['observation_sigma_m'])
        else:
            if spec.get('forward_model') != 'nondispersive_coaxial_tof_proxy':
                raise ValueError('ToF must explicitly opt into the nondispersive research proxy')
            prior = fit_material_prior(frames, heights, sensor_model=spec['forward_model'],
                bottom_distance_m=spec['bottom_distance_m'], observation_sigma_m=spec['observation_sigma_m'])
        result = dict(schema_version=1, research_only=True, production_enabled=False,
            hardware_qualified=False, sensor_family=family, material_id=spec['material_id'],
            rig_id=spec['rig_id'], geometry_model=spec['geometry_model'],
            prior=prior.to_dict(), calibration_provenance=provenance)
    else:
        stored = json.loads(Path(args.prior).read_text())
        if stored['material_id'] != args.material_id or stored['rig_id'] != args.rig_id:
            raise ValueError('Material or rig changed: recalibration required')
        stereo = stored['sensor_family'] != 'tof'
        obs = load_observation(args.observations, stereo)
        if stereo:
            result = infer_stereo_height(obs, StereoMaterialPrior(**stored['prior']))
        else:
            result = infer_height(obs, MaterialPrior(**stored['prior']))
        result.update(prior_sha256=digest(args.prior), material_id=args.material_id,
            rig_id=args.rig_id, production_enabled=False, bottom_path_identity_verified=False,
            current_ground_truth_height_used=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
