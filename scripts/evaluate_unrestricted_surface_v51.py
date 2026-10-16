"""Diagnostic candidate-free depth estimates on every V50 native frame.

Truth is used only for controlled loss, counterfactual attribution, and scoring.
The deployable-input estimates use fitted empty geometry, dry depth, current raw
depth, and no surface mask, model probabilities, height clipping, or acceptance
gate. Records from failed geometry installations are retained separately.
"""
import argparse
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from evaluate_shared_empty_v45 import install
from evaluate_uniform_loss_v48 import LEVELS, delete_to_actual_loss
from liquid_depth.liquid_height_metrics import HeightCalibration, reference_statistics, world_points
from liquid_depth.ordinary_surface_v35 import interior_ray_roi


def valid(z):
    return np.isfinite(z) & (z > 0)


def median_height(z, cal, pixels):
    good = pixels & valid(z)
    if not good.any():
        return None, 0
    h = world_points(z, cal)[..., 2] - cal.bottom_z_m
    return float(np.median(h[good])), int(good.sum())


def trimmed_disparity_height(raw, cal, pixels, sensor):
    """Exploratory no-mask stereo estimator; no ground truth or zero clipping."""
    if sensor == 'tof':
        return None
    y,x=np.nonzero(pixels&valid(raw))
    if len(x)<2:
        return None
    k=cal.intrinsics
    rays=np.column_stack(((x-k[0,2])/k[0,0],(y-k[1,2])/k[1,1],
                          np.ones(len(x))))
    cosine=-(rays@cal.camera_to_world_cv[:3,:3].T)[:,2]
    keep=cosine>.1
    if keep.sum()<2:
        return None
    baseline={'active_stereo':.055,'structured_light':.035}[sensor]
    fb=k[0,0]*baseline
    inverse_clearance=(fb/raw[y[keep],x[keep]])/cosine[keep]
    values=np.sort(inverse_clearance[np.isfinite(inverse_clearance)])
    if len(values)<2:
        return None
    trim=int(.1*len(values))
    slope=float(values[trim:len(values)-trim].mean())
    if not np.isfinite(slope) or slope<=0:
        return None
    return float(cal.camera_to_world_cv[2,3]-fb/slope-cal.bottom_z_m)


def make_masks(raw, cal, rim):
    opening_plane = replace(cal, bottom_z_m=rim - 1e-5)
    opening = interior_ray_roi(opening_plane, rim, raw.shape, .90)
    central = interior_ray_roi(cal, rim, raw.shape, .65)
    xyz = world_points(raw, cal)
    normalized = xyz[..., :2] / np.asarray(cal.radii_m)
    inside = (abs(normalized) ** cal.footprint_exponent).sum(axis=-1) <= 1.05 ** cal.footprint_exponent
    margin = max(.03, .02 * rim)
    plausible = (xyz[..., 2] >= cal.bottom_z_m - margin) & (xyz[..., 2] <= rim + margin)
    container = opening & inside & plausible
    return opening, central, container


def stat(rows, field):
    n = len(rows)
    q = [r for r in rows if r[field] is not None]
    ae = np.asarray([abs(r[field]-r['true_height_m']) for r in q])
    rel = np.asarray([abs(r[field]-r['true_height_m'])/r['true_height_m']
                      for r in q if r['true_height_m'] > 0])
    passed = sum(abs(r[field]-r['true_height_m']) <= max(.005,.02*r['true_height_m']) for r in q)
    return {'frames':n,'numeric':len(q),'coverage_pct':100*len(q)/n if n else None,
            'all_frame_pass_pct':100*passed/n if n else None,
            'conditional_mae_mm':1000*float(ae.mean()) if len(ae) else None,
            'conditional_p95_mm':1000*float(np.percentile(ae,95)) if len(ae) else None,
            'conditional_absrel_pct':100*float(rel.mean()) if len(rel) else None,
            'absolute_bins_all_pct':(100*np.histogram(ae,[0,.005,.01,.02,.05,.1,.2,np.inf])[0]/n).tolist()
                  if n else None,
            'relative_bins_numeric_pct':(100*np.histogram(rel,[0,.005,.01,.02,.03,.05,.1,np.inf])[0]/len(rel)).tolist()
                  if len(rel) else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wet-root',type=Path,required=True)
    p.add_argument('--dry-root',type=Path,required=True)
    p.add_argument('--baseline-records',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--cases',type=int,default=99)
    p.add_argument('--repeats',type=int,default=4)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    baseline={(r['case'],r['repeat'],r['target_loss_pct']):r
              for r in map(json.loads,a.baseline_records.open())}
    rows=[]
    failed=[]
    for case in range(a.cases):
        folder=a.wet_root/f'{case*6+5:05d}'
        meta=json.loads((folder/'metadata.json').read_text())
        cal_true=replace(HeightCalibration.from_simulation(meta),
                         bottom_z_m=float(meta['interior_bottom_z_m']))
        truth=float(reference_statistics(meta,cal_true)['mean_m'])
        with np.load(folder/'geometry.npz') as z:
            oracle=z['mask'].astype(bool)
        frames=[f for f in json.loads((folder/'frames.json').read_text())
                if f['repeat'] < a.repeats]
        try:
            item=install(case,a.wet_root,a.dry_root,a.seed)
            bundle=item['bundle']
            cal=item['fitted']
            dry=item['dry']
            rim=float(bundle['rim_z_m'])
            opening, central, container=make_masks(dry,cal,rim)
            dry_med,dry_count=median_height(dry,cal,central)
        except (ValueError,FileNotFoundError) as exc:
            for frame in frames:
                failed.append({'case':case,'repeat':frame['repeat'],
                               'sensor':meta['sensor_family'],'band':meta['band'],
                               'true_height_m':truth,'reason':str(exc)})
            continue
        for frame in frames:
            rep=frame['repeat']
            native=np.load(frame['raw_path'],allow_pickle=False)
            native_loss=100*(1-int((oracle&valid(native)).sum())/max(1,int(oracle.sum())))
            eligible=native_loss<=.5
            for level in LEVELS:
                if level and native_loss>level:
                    continue
                raw=native if level==0 else delete_to_actual_loss(
                    native,oracle,level,a.seed+case*10007+rep*509)
                actual_loss=100*(1-int((oracle&valid(raw)).sum())/max(1,int(oracle.sum())))
                # The central raw median has no model-mask or minimum-point gate.
                wet_med,center_count=median_height(raw,cal,central)
                paired=None if wet_med is None or dry_med is None else wet_med-dry_med
                broad,broad_count=median_height(raw,cal,container)
                stereo_trimmed=trimmed_disparity_height(raw,cal,container,
                                                        meta['sensor_family'])
                broad_surface=int((container&oracle&valid(raw)).sum())
                broad_oracle_fit,broad_oracle_count=median_height(raw,cal,container&oracle)
                broad_oracle_true,_=median_height(raw,cal_true,container&oracle)
                broad_non_surface,_=median_height(raw,cal,container&~oracle)
                oracle_fit,oracle_count=median_height(raw,cal,central&oracle)
                oracle_true,_=median_height(raw,cal_true,central&oracle)
                center_surface=int((central&oracle&valid(raw)).sum())
                previous=baseline.get((case,rep,level))
                rows.append({'case':case,'repeat':rep,'sensor':meta['sensor_family'],
                             'band':meta['band'],'target_loss_pct':level,
                             'native_surface_loss_pct':native_loss,
                             'actual_surface_loss_pct':actual_loss,
                             'paired_eligible':eligible,
                             'geometry_source':bundle['source'],
                             'true_height_m':truth,
                             'baseline_m':None if previous is None else previous['prediction_m'],
                             'broad_container_median_m':broad,
                             'broad_disparity_trimmed_m':stereo_trimmed,
                             'broad_surface_pixels_installed_median_m':broad_oracle_fit,
                             'broad_surface_pixels_true_geometry_median_m':broad_oracle_true,
                             'broad_non_surface_median_m':broad_non_surface,
                             'central_raw_median_m':wet_med,
                             'central_paired_median_m':paired,
                             'oracle_pixels_installed_median_m':oracle_fit,
                             'oracle_pixels_true_geometry_median_m':oracle_true,
                             'full_valid':int(valid(raw).sum()),
                             'opening_valid':int((opening&valid(raw)).sum()),
                             'container_valid':broad_count,
                             'container_true_surface_valid':broad_surface,
                             'container_true_surface_fraction':broad_surface/max(broad_count,1),
                             'container_oracle_valid':broad_oracle_count,
                             'central_valid':center_count,
                             'central_true_surface_valid':center_surface,
                             'central_true_surface_fraction':center_surface/max(center_count,1),
                             'dry_center_valid':dry_count,
                             'oracle_center_valid':oracle_count})
        print('case_done',case+1,'rows',len(rows),'failed',len(failed),flush=True)
    with (a.output/'records.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row,allow_nan=False)+'\n')
    (a.output/'geometry_failures.json').write_text(json.dumps(failed,indent=2))
    groups=defaultdict(list)
    for row in rows:
        if row['paired_eligible']:
            groups[str(row['target_loss_pct'])].append(row)
            groups[f"{row['target_loss_pct']}/{row['sensor']}"].append(row)
            groups[f"{row['target_loss_pct']}/band_{row['band']}"].append(row)
    fields=('baseline_m','broad_container_median_m','broad_disparity_trimmed_m',
            'broad_surface_pixels_installed_median_m',
            'broad_surface_pixels_true_geometry_median_m',
            'central_raw_median_m',
            'central_paired_median_m','oracle_pixels_installed_median_m',
            'oracle_pixels_true_geometry_median_m')
    summary={key:{field:stat(group,field) for field in fields}
             for key,group in sorted(groups.items())}
    (a.output/'summary_paired.json').write_text(json.dumps(summary,indent=2))
    (a.output/'protocol.json').write_text(json.dumps({
        'measurand':'actual liquid height relative to interior bottom',
        'truth_use':'controlled deletion and offline scoring/attribution only',
        'primary_candidate_free_route':'broad container-footprint raw height median using fitted empty geometry; no learned surface mask, probability/point-count acceptance gate, clipping, or zero snap',
        'auxiliary_candidate_free_route':'central dry-paired raw height median; diagnostic comparator only',
        'baseline_records':str(a.baseline_records),'seed':a.seed,
        'attempted_native_frames':a.cases*a.repeats,
        'geometry_failed_frames':len(failed),
        'paired_eligible_per_level':len(groups['0']),
        'native_over_0.5pct_frames':sum(1 for r in rows if r['target_loss_pct']==0 and not r['paired_eligible']),
        'all_installed_frames_native_level':sum(1 for r in rows if r['target_loss_pct']==0),
        'important':'unrestricted estimates are diagnostics, not production-qualified outputs'},indent=2))
    print('COMPLETE',len(rows),len(failed),flush=True)


if __name__=='__main__':
    main()
