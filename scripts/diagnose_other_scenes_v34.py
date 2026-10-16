"""Post-test paired attribution and camera-range strata; does not tune a model."""
import argparse,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from summarize_other_scenes_v34 import summarize,band

def paired(rows,left,right):
    q=[r for r in rows if left in r['methods'] and right in r['methods'] and r['methods'][left]['numeric'] and r['methods'][right]['numeric']]
    def errors(m):return np.array([r['methods'][m]['absolute_error_m'] for r in q])*1000
    a,b=errors(left),errors(right)
    return dict(paired_frames=len(q),left_mae_mm=float(a.mean()) if len(q) else None,
        right_mae_mm=float(b.mean()) if len(q) else None,left_p95_mm=float(np.percentile(a,95)) if len(q) else None,
        right_p95_mm=float(np.percentile(b,95)) if len(q) else None,
        right_better_fraction=float((b<a).mean()) if len(q) else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records=json.loads((a.output/'test-raw.json').read_text())['records'];groups=defaultdict(list)
    for r in records:
        scene=r['scenario']+('/'+r['frame']['variant'] if r['scenario']=='depth_failure' else '')
        h=band(r['true_height_m'],r['nominal_height_m']);z=r['camera_to_surface_m']
        zb='0-1' if z<1 else '1-3' if z<3 else '3-6' if z<6 else '6-10'
        invalid=1-r['raw_surface_valid_fraction']
        loss='0-25' if invalid<.25 else '25-50' if invalid<.5 else '50-75' if invalid<.75 else '75-90' if invalid<.9 else '90-95' if invalid<.95 else '95-100'
        for key in (scene,scene+'/H='+h,scene+'/Z='+zb,scene+'/loss='+loss,scene+'/H='+h+'/Z='+zb):groups[key].append(r)
    report={}
    for key,rows in groups.items():
        ps=[r['methods']['refined'] for r in rows if r['methods']['refined']['numeric']]
        bad=[r for r in rows if r['methods']['refined']['numeric'] and not r['methods']['refined']['within_tolerance']]
        good=[r for r in rows if r['methods']['refined']['numeric'] and r['methods']['refined']['within_tolerance']]
        avg=lambda rs,k:float(np.mean([r[k] for r in rs])) if rs else None
        report[key]=dict(refined=summarize(rows,'refined'),paired_mask_oracle=paired(rows,'refined','oracle_mask_refined'),
            paired_raw_grid_refined=paired(rows,'raw_grid','refined'),paired_neural_refined=paired(rows,'v8_neural','refined'),
            mean_mask_iou_bad=avg(bad,'mask_iou'),mean_mask_iou_good=avg(good,'mask_iou'),
            raw_valid_fraction_bad=avg(bad,'raw_surface_valid_fraction'),raw_valid_fraction_good=avg(good,'raw_surface_valid_fraction'),
            camera_surface_distance_bad=avg(bad,'camera_to_surface_m'),camera_surface_distance_good=avg(good,'camera_to_surface_m'))
        if rows[0]['scenario']=='low_light':report[key]['paired_exposure']=paired(rows,'refined','exposure_refined')
    # Cluster bootstrap respects four correlated depth repeats and paired scene geometry.
    bootstrap={};rng=np.random.default_rng(202610120349)
    for scene in ('ordinary','low_light','glare','floating_objects','uneven_surface'):
        rows=groups[scene];by_cluster=defaultdict(list)
        for r in rows:by_cluster[r['cluster']].append(r)
        clusters=list(by_cluster);maes=[];passes=[]
        for _ in range(2000):
            sample=[r for i in rng.integers(0,len(clusters),len(clusters)) for r in by_cluster[clusters[i]]]
            q=[r['methods']['refined'] for r in sample if r['methods']['refined']['numeric']]
            maes.append(np.mean([p['absolute_error_m']*1000 for p in q]) if q else np.nan)
            passes.append(100*sum(p['within_tolerance'] for p in q)/len(sample))
        bootstrap[scene]=dict(mae_mm_cluster_ci95=np.nanpercentile(maes,[2.5,97.5]).tolist(),
            all_frame_pass_percent_cluster_ci95=np.percentile(passes,[2.5,97.5]).tolist(),clusters=len(clusters))
    geometry=[]
    for r in records:
        if r['oracle_geometry']['statistics']:
            geometry.append(abs(r['oracle_geometry']['statistics']['mean_m']-r['true_height_m'])*1000)
    output=dict(groups=report,bootstrap=bootstrap,coordinate_sanity=dict(frames=len(geometry),
        mae_mm=float(np.mean(geometry)),p95_mm=float(np.percentile(geometry,95)),max_mm=float(max(geometry))),
        camera_to_bottom_max_m=max(r['camera_to_bottom_m'] for r in records),
        actual_height_min_m=min(r['true_height_m'] for r in records),actual_height_max_m=max(r['true_height_m'] for r in records))
    (a.output/'diagnosis.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in output.items() if k not in ('groups','bootstrap')},indent=2))
if __name__=='__main__':main()
