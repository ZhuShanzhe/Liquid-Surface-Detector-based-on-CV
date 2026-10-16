"""All-attempted-frame evaluation of the frozen facade, with clustered CIs."""
import argparse
import hashlib
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from evaluate_shared_empty_v45 import install
from special_scene_protocol_v1 import (SEED, REPEATS, SCENES, SENSORS, LEVELS,
                                       BAND_NAMES, tolerance, design)
from liquid_depth.determined_scene_meter import DeterminedSceneMeter, VERSION
from special_scene_input_evidence_v1 import assess, assess_capture_origin, POLICY


def default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=default, allow_nan=False))


def calibrate(a):
    rows=[]
    for gid in range(a.worker,a.count,a.workers):
        for si,sensor in enumerate(SENSORS):
            case=gid*2+si
            out=a.out/'calibrations'/f'{case:05d}.json'
            if out.exists():
                rows.append(json.loads(out.read_text()));continue
            try:
                # Paired operator perturbations across the two camera mechanisms.
                item=install(case,a.root/'installations',a.root/'dry',SEED-case*8071+gid*8071,
                             max_annotation_attempts=6,source_slot_stride=1)
                value={'case':case,'physical_id':gid,'sensor':sensor,'success':True,
                       'bundle':item['bundle'],'audit':item['audit']}
            except ValueError as exc:
                value={'case':case,'physical_id':gid,'sensor':sensor,'success':False,'reason':str(exc)}
            dump(out,value);rows.append(value)
            print('CALIBRATED',case,value['success'],flush=True)
    dump(a.out/f'calibration-worker-{a.worker}.json',rows)


def evaluate(a):
    selections=json.loads((a.out/'selected-input-repeats.json').read_text()) if (a.out/'selected-input-repeats.json').exists() else None
    with (a.out/f'records-worker-{a.worker}.jsonl').open('w') as stream:
        for gid in range(a.worker,a.count,a.workers):
            for si,sensor in enumerate(SENSORS):
                case=gid*2+si
                cal=json.loads((a.out/'calibrations'/f'{case:05d}.json').read_text())
                dry=np.load(a.root/'dry'/f'{case:04d}'/'dry_depth_m.npy',allow_pickle=False)
                meter=DeterminedSceneMeter.from_empty_geometry(cal['bundle'],dry) if cal['success'] else None
                for scene in SCENES:
                    for level in LEVELS:
                        folder=a.root/'observations'/f'{gid:04d}'/f'{scene}_{level}'
                        meta=json.loads((folder/'metadata.json').read_text())
                        rgb=cv2.imread(str(folder/'rgb.png'))
                        assert rgb is not None and rgb.shape[:2]==dry.shape
                        with np.load(folder/'labels_offline.npz') as f:
                            semantic=f['semantic'];surface=semantic==2
                        gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
                        # Offline RGB quality is descriptive only, never an inclusion gate.
                        roi=gray[surface] if surface.any() else gray.ravel()
                        for rep in selections[f'{gid}|{scene}|{level}'] if selections is not None else range(REPEATS):
                            raw=np.load(folder/f'{sensor}_raw_{rep:02d}.npy',allow_pickle=False)
                            err=np.load(folder/f'{sensor}_error_offline_{rep:02d}.npy',allow_pickle=False)
                            row={k:meta[k] for k in ('physical_id','band','band_name','scenario','level','truth','camera_to_bottom_m','nominal_liquid_height_m')}
                            row.update(case=case,sensor=sensor,repeat=rep,
                                rgb_mean_gray=float(roi.mean()),rgb_dark_fraction=float((roi<16).mean()),
                                rgb_saturated_fraction=float((rgb.max(axis=-1)[surface]>=250).mean()) if surface.any() else None,
                                surface_pixels=int(surface.sum()),
                                valid_surface_pixels=int((surface&(raw>0)).sum()),
                                actual_liquid_missing_fraction_offline=float((raw[surface]<=0).mean()) if surface.any() else None,
                                # Legacy error labels can be overwritten by the range-cutoff label;
                                # this is NOT the true generated highlight-mask area.
                                retained_highlight_error_label_fraction_offline=float((err[surface]==4).mean()) if surface.any() else None,
                                calibration_success=cal['success'],inference_failure=None)
                            result={}
                            evidence=assess(raw,cal['bundle']) if cal['success'] else {'eligible':False,'reasons':['invalid_calibration']}
                            capture=assess_capture_origin(raw,semantic,err,contains_liquid=meta['nominal_liquid_height_m']>0)
                            evidence['capture_origin']=capture
                            evidence['eligible']&=capture['eligible']
                            evidence['reasons']+=capture['reasons']
                            row['input_evidence']=evidence
                            row['scope_included']=evidence['eligible']
                            if not evidence['eligible']:
                                row['inference_failure']='not_tested_input_evidence:'+'|'.join(evidence['reasons'])
                            elif meter is not None:
                                try:
                                    start=time.perf_counter()
                                    result=meter.measure(scene,depth_m=raw,rgb_bgr=rgb,visual_enabled=False)
                                    row['algorithm_ms']=1000*(time.perf_counter()-start)
                                    assert result['determined_version']==VERSION
                                    assert not result['visual_module_executed']
                                    assert not result['liquid_visual_guidance_used']
                                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                                    row['inference_failure']=type(exc).__name__+':'+str(exc)
                            else:
                                row['inference_failure']='input_calibration:'+cal['reason']
                            for key in ('liquid_depth_m','highest_height_m'):
                                value=result.get(key)
                                row[key]=float(value) if value is not None and np.isfinite(value) else None
                            row['diagnostics']=result.get('diagnostics',{})
                            if row['liquid_depth_m'] is None and not row['inference_failure']:
                                row['inference_failure']='|'.join(row['diagnostics'].get('flags',[])) or 'backend_no_numeric_value'
                            row['visual_module_executed']=bool(result.get('visual_module_executed',False))
                            stream.write(json.dumps(row,default=default,allow_nan=False)+'\n');stream.flush()
                print('EVALUATED',case,sensor,flush=True)


def metric(rows,key,peak=False):
    truth_key='highest_height_m' if peak else 'liquid_height_m'
    n=len(rows);numeric=[r for r in rows if r.get(key) is not None]
    if not n:
        return dict(frames=0,numeric_frames=0,numeric_coverage_pct=None,all_frame_pass_pct=None,
                    mae_mm=None,p95_mm=None,max_mm=None,absrel_pct=None,no_output_frames=0,
                    absolute_bins_all_frame_pct=[],relative_bins_nonzero_numeric_pct=[],
                    reference_20mm_2pct_all_frame_pass_pct=None,cluster_bootstrap_95ci={})
    e=np.array([abs(r[key]-r['truth'][truth_key]) for r in numeric])
    bias=np.array([r[key]-r['truth'][truth_key] for r in numeric])
    rel=np.array([abs(r[key]-r['truth'][truth_key])/r['truth'][truth_key] for r in numeric if r['truth'][truth_key]>0])
    passing=[abs(r[key]-r['truth'][truth_key]) <= (max(.05,.05*r['truth'][truth_key]) if peak else tolerance(r['scenario'],r['truth'][truth_key])) for r in numeric]
    legacy=sum(abs(r[key]-r['truth'][truth_key])<=max(.02,.02*r['truth'][truth_key]) for r in numeric)
    result=dict(frames=n,physical_installations=len({r['physical_id'] for r in rows}),
        sensor_installations=len({r['case'] for r in rows}),numeric_frames=len(numeric),
        numeric_coverage_pct=100*len(numeric)/n,all_frame_pass_pct=100*sum(passing)/n,
        conditional_numeric_pass_pct=100*sum(passing)/len(numeric) if numeric else None,
        mae_mm=float(e.mean()*1000) if len(e) else None,
        median_mm=float(np.median(e)*1000) if len(e) else None,
        p95_mm=float(np.percentile(e,95)*1000) if len(e) else None,
        p99_mm=float(np.percentile(e,99)*1000) if len(e) else None,
        max_mm=float(e.max()*1000) if len(e) else None,
        signed_bias_mm=float(bias.mean()*1000) if len(e) else None,
        absrel_pct=float(rel.mean()*100) if len(rel) else None,relative_numeric_frames=len(rel),
        absolute_bins_all_frame_pct=(np.histogram(e,[0,.005,.02,.05,.1,np.inf])[0]*100/n).tolist(),
        relative_bins_nonzero_numeric_pct=(np.histogram(rel,[0,.01,.02,.05,.10,np.inf])[0]*100/len(rel)).tolist() if len(rel) else [],
        no_output_frames=n-len(numeric),
        no_output_reasons=dict(Counter(r['inference_failure'] or 'backend_no_numeric_value' for r in rows if r.get(key) is None)),
        reference_20mm_2pct_all_frame_pass_pct=100*legacy/n)
    # Stratify cluster resampling by actual-depth band; preserve paired cameras/conditions/repeats.
    clusters=defaultdict(list)
    for r in rows:
        clusters[(r['band'],r['physical_id'])].append(r)
    draws=np.random.default_rng(SEED+37)
    sums=np.zeros((1000,4))
    for band in sorted({k[0] for k in clusters}):
        matrix=[]
        for (b,gid),q in clusters.items():
            if b!=band:continue
            good=[r for r in q if r.get(key) is not None]
            ae=sum(abs(r[key]-r['truth'][truth_key]) for r in good)
            ok=sum(abs(r[key]-r['truth'][truth_key]) <= (max(.05,.05*r['truth'][truth_key]) if peak else tolerance(r['scenario'],r['truth'][truth_key])) for r in good)
            matrix.append([len(q),len(good),ae,ok])
        m=np.asarray(matrix,float)
        sums+=m[draws.integers(0,len(m),(1000,len(m)))].sum(axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        ci_values={'mae_mm':1000*sums[:,2]/sums[:,1],
                   'all_frame_pass_pct':100*sums[:,3]/sums[:,0],
                   'numeric_coverage_pct':100*sums[:,1]/sums[:,0]}
    result['cluster_bootstrap_95ci']={k:np.percentile(v[np.isfinite(v)],[2.5,97.5]).tolist() if np.isfinite(v).any() else None for k,v in ci_values.items()}
    return result


def summarize(a):
    rows=[]
    for worker in range(a.workers):
        rows.extend(json.loads(s) for s in (a.out/f'records-worker-{worker}.jsonl').read_text().splitlines())
    assert len(rows)==a.count*2*3*3*REPEATS
    ids=[(r['physical_id'],r['sensor'],r['scenario'],r['level'],r['repeat']) for r in rows]
    assert len(ids)==len(set(ids))
    rows.sort(key=lambda r:(r['physical_id'],r['sensor'],r['scenario'],r['level'],r['repeat']))
    groups=defaultdict(list)
    for r in rows:
        for sensor,level,band in itertools.product(('all',r['sensor']),('all',r['level']),('all',r['band_name'])):
            groups[f"{r['scenario']}|{sensor}|{level}|{band}"].append(r)
        h=r['truth']['liquid_height_m'];distance=r['camera_to_bottom_m']
        subset='far_shallow_wet' if distance>=6 and 0<h<=1 else 'far_dry' if distance>=6 and h==0 else 'other'
        for sensor in ('all',r['sensor']):
            groups[f"{r['scenario']}|{sensor}|all|subset={subset}"].append(r)
            groups[f"{r['scenario']}|{sensor}|all|{'wet' if h>0 else 'dry_control'}"].append(r)
    summaries={}
    for key,attempted in sorted(groups.items()):
        q=[r for r in attempted if r['scope_included']]
        summaries[key]={'liquid':metric(q,'liquid_depth_m')}
        summaries[key].update(attempted_frames=len(attempted),excluded_input_frames=len(attempted)-len(q),
            input_qualification_pct=100*len(q)/len(attempted),
            excluded_reasons=dict(Counter(reason for r in attempted if not r['scope_included'] for reason in r['input_evidence']['reasons'])))
        if attempted[0]['scenario']=='uneven_surface':
            summaries[key]['highest']=metric(q,'highest_height_m',True)
        elapsed=[r['algorithm_ms'] for r in q if 'algorithm_ms' in r]
        summaries[key]['algorithm_latency_ms']={name:float(fn(elapsed)) for name,fn in (
            ('median',np.median),('p95',lambda x:np.percentile(x,95)),('max',max))} if elapsed else None
        summaries[key]['input']={
            'mean_gray':float(np.mean([r['rgb_mean_gray'] for r in q])),
            'dark_fraction':float(np.mean([r['rgb_dark_fraction'] for r in q])),
            'mean_liquid_missing_fraction':float(np.mean([r['actual_liquid_missing_fraction_offline'] for r in q if r['actual_liquid_missing_fraction_offline'] is not None])) if any(r['actual_liquid_missing_fraction_offline'] is not None for r in q) else None} if q else None
    dump(a.out/'stratified-summary.json',summaries)
    (a.out/'records.jsonl').write_text(''.join(json.dumps(r,default=default,allow_nan=False)+'\n' for r in rows))
    cals=[json.loads(p.read_text()) for p in sorted((a.out/'calibrations').glob('*.json'))]
    dump(a.out/'input-audit.json',{'planned_frames':a.count*72,'observed_frames':len(rows),
        'planned_sensor_installations':a.count*2,'successful_calibrations':sum(c['success'] for c in cals),
        'calibrations':cals,'all_attempted_frames_in_primary_denominator':False,
        'input_evidence_policy':POLICY,'qualified_frames':sum(r['scope_included'] for r in rows),
        'excluded_input_frames':sum(not r['scope_included'] for r in rows),
        'input_qualification_pct':100*sum(r['scope_included'] for r in rows)/len(rows),
        'no_prediction_or_truth_based_exclusion':True,'visual_calls':sum(r['visual_module_executed'] for r in rows),
        'absolute_bins_mm':['[0,5)','[5,20)','[20,50)','[50,100)','[100,infinity)'],
        'relative_bins_pct':['[0,1)','[1,2)','[2,5)','[5,10)','[10,infinity)']})
    if (a.out/'capture-eligibility-summary.json').exists():
        audit=json.loads((a.out/'input-audit.json').read_text())
        audit['raw_capture_eligibility']=json.loads((a.out/'capture-eligibility-summary.json').read_text())
        dump(a.out/'input-audit.json',audit)
    print('COMPLETE',len(rows),'frames',len(summaries),'strata',flush=True)
    for scene in SCENES:
        for sensor in ('all',*SENSORS):
            m=summaries[f'{scene}|{sensor}|all|all']['liquid']
            print(scene,sensor,json.dumps(m),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=66)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--phase',choices=['calibrate','evaluate','summarize'],required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'calibrations').mkdir(exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    {'calibrate':calibrate,'evaluate':evaluate,'summarize':summarize}[a.phase](a)
