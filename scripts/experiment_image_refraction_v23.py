#!/usr/bin/env python3
"""Rendered-image experiment. Calibration labels and test scoring are separate.

Stage 1 generates actual PNG image measurements; stage 2 reads PNGs into a
GT-free matcher; stage 3 freezes gate thresholds on development data only.
All protocols are research-only; no production configuration is changed.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from liquid_depth.refraction_images_v23 import render_capture
from liquid_depth.refraction_matching_v23 import stereo_matches,decode_structured
from liquid_depth.refraction_inverse_v23 import fit_prior,infer,raw_height


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))


def generate(root,pilot=False):
    root.mkdir(parents=True,exist_ok=False)
    configs=[]
    conditions=('transparent','translucent_mild','translucent_strong')
    schedule=[('pilot',.8,1.38,91001)] if pilot else (
        [('development',b,n,93000+i) for i,(b,n) in enumerate([(.5,1.31),(2.,1.40),(6.,1.35)])]
        +[('test',b,n,97000+i) for i,(b,n) in enumerate((b,n) for b in (.3,.8,2.,6.,9.8) for n in (1.30,1.45))])
    started=time.time()
    for fi,(family,baseline) in enumerate([('active_stereo',.055),('structured_light',.035)]):
        for ci,(split,bottom,index,seed) in enumerate(schedule):
            for condition in conditions:
                config_id=f'{family}_{split}_{ci:02d}_{condition}'
                folder=root/config_id;folder.mkdir()
                captures=[]
                # H labels appear ONLY in the calibration interface / evaluator.
                heights=[('prior',x) for x in (.10,.22,.35,.50,.65)]+[('test',x) for x in (.08,.15,.28,.42,.58,.72)]
                for j,(role,ratio) in enumerate(heights):
                    sample=folder/f'{j:02d}';sample.mkdir()
                    images,cal,truth=render_capture(seed+fi*1000,bottom*ratio,index,bottom,baseline,condition,family)
                    for name,image in images.items():
                        if not cv2.imwrite(str(sample/(name+'.png')),image):raise IOError('Image write failed')
                    write_json(sample/'calibration.json',cal)
                    truth_path=sample/'evaluator_truth.npz'
                    np.savez_compressed(truth_path,**truth)
                    record=dict(folder=str(sample.resolve()),role=role,
                        evaluator_truth=str(truth_path.resolve()))
                    if role=='prior':
                        label_rng=np.random.default_rng(seed+fi*1000+j*65537)
                        label_sigma=.0005+.001*bottom*ratio
                        record['known_liquid_height_m']=float(bottom*ratio+label_rng.normal(0,label_sigma))
                        record['prior_label_sigma_m']=label_sigma
                    captures.append(record)
                configs.append(dict(id=config_id,family=family,condition=condition,split=split,
                    camera_to_bottom_m=bottom,material_seed=seed,captures=captures))
                print('GENERATED',len(configs),config_id,round(time.time()-started,1),flush=True)
    write_json(root/'manifest.json',dict(version=23,configs=configs,
        renderer='2D Snell monochrome ray tracing plus Beer-Lambert/blur/veil scattering proxy',
        monochrome_is_not_spectral_ir=True,no_glass_sidewall=True,flat_liquid=True,
        test_labels_available_to_inference=False,pilot=pilot))


def match_capture(capture,family,optimized):
    folder=Path(capture['folder'])
    cal=json.loads((folder/'calibration.json').read_text())
    # Observations come ONLY from PNG pixels and declared rig calibration.
    images={p.stem:cv2.imread(str(p),cv2.IMREAD_GRAYSCALE) for p in sorted(folder.glob('*.png'))}
    t=time.perf_counter()
    obs=(stereo_matches(images['left'],images['right'],cal,optimized) if family=='active_stereo'
         else decode_structured(images,cal,optimized))
    elapsed=1000*(time.perf_counter()-t)
    np.savez_compressed(folder/('matches_optimized.npz' if optimized else 'matches_baseline.npz'),
        left_q=obs['left_q'],right_q=obs['right_q'],weights=obs['weights'],
        left_pixels=obs['left_pixels'],right_pixels=obs['right_pixels'])
    return obs,cal,elapsed


def match_truth_error(obs,capture,family):
    # Evaluator only: truth never affects feature selection or confidence.
    if not obs['regions']:return dict(regions=0,epe_px=None,within_1px_pct=None)
    with np.load(capture['evaluator_truth']) as truth:
        u=truth['right_u' if family=='active_stereo' else 'projector_u']
        v=truth['right_v' if family=='active_stereo' else 'projector_v']
        px=obs['left_pixels'][:,0].astype(np.float32).reshape(-1,1)
        py=obs['left_pixels'][:,1].astype(np.float32).reshape(-1,1)
        reference=np.column_stack((cv2.remap(u,px,py,cv2.INTER_LINEAR).ravel(),
                                   cv2.remap(v,px,py,cv2.INTER_LINEAR).ravel()))
    error=np.linalg.norm(reference-obs['right_pixels'],axis=1)
    return dict(regions=obs['regions'],epe_px=float(np.mean(error)),
                within_1px_pct=float(100*np.mean(error<=1)))


def evaluate(root,output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    data=json.loads((root/'manifest.json').read_text());records=[];matches=[];priors=[]
    started=time.time()
    for ci,config in enumerate(data['configs']):
        for optimized in (False,True):
            captured=[match_capture(c,config['family'],optimized) for c in config['captures']]
            all_obs=[x[0] for x in captured];cal=captured[0][1]
            for c,(obs,_,elapsed) in zip(config['captures'],captured):
                matches.append(dict(config_id=config['id'],split=config['split'],family=config['family'],
                    condition=config['condition'],optimized=optimized,role=c['role'],matching_ms=elapsed,
                    **match_truth_error(obs,c,config['family'])))
            hs=[c['known_liquid_height_m'] for c in config['captures'][:5]]
            prior_maps={}
            for count,ids in [(3,[0,2,4]),(5,[0,1,2,3,4])]:
                try:
                    p=fit_prior([all_obs[i] for i in ids],[hs[i] for i in ids],cal)
                    prior_maps[count]=p
                    priors.append(dict(config_id=config['id'],optimized=optimized,prior_count=count,
                        accepted=True,parameters=p))
                except ValueError as exc:
                    prior_maps[count]=None
                    priors.append(dict(config_id=config['id'],optimized=optimized,prior_count=count,
                        accepted=False,reason=str(exc)))
            anchor=raw_height(all_obs[2],cal)
            gain=hs[2]/anchor if anchor is not None and anchor>.001 else None
            raw_priors=[raw_height(all_obs[i],cal) for i in [0,2,4]]
            affine=np.polyfit(raw_priors,[hs[i] for i in [0,2,4]],1) if all(h is not None for h in raw_priors) else None
            for j,capture in enumerate(config['captures'][5:],5):
                obs=all_obs[j]
                # All predictions are computed before opening evaluator truth.
                raw=raw_height(obs,cal)
                predictions={}
                for name,value in [('no_prior_raw_surface',raw),
                    ('one_prior_gain',raw*gain if raw is not None and gain is not None else None),
                    ('three_prior_affine',float(np.polyval(affine,raw)) if raw is not None and affine is not None else None)]:
                    predictions[name]=dict(height_m=value,observable=value is not None,
                        uncertainty_m=None,reasons=[] if value is not None else ['insufficient_matches'])
                predictions['no_prior_fixed_water']=infer(obs,dict(index=1.333,bias_px=0,index_sd=0,bias_sd=0),cal)
                for count in (3,5):
                    p=prior_maps[count]
                    predictions[f'{count}_prior_physical']=infer(obs,p,cal) if p is not None else dict(
                        height_m=None,observable=False,uncertainty_m=None,reasons=['material_prior_failed'])
                with np.load(capture['evaluator_truth']) as truth:height=float(truth['height_m'])
                for method,result in predictions.items():
                    predicted=result['height_m']
                    records.append(dict(config_id=config['id'],sample_id=Path(capture['folder']).name,
                        split=config['split'],family=config['family'],condition=config['condition'],
                        camera_to_bottom_m=config['camera_to_bottom_m'],optimized=optimized,method=method,
                        true_height_m=height,predicted_height_m=predicted,
                        absolute_error_m=abs(predicted-height) if predicted is not None else None,
                        observable=result['observable'],accepted=result['observable'],
                        tolerance_m=max(.005,.02*height),
                        predicted_tolerance_m=max(.005,.02*max(predicted or 0,0)),
                        diagnostics=result,matching_ms=captured[j][2]))
        print('EVALUATED',ci+1,len(data['configs']),round(time.time()-started,1),flush=True)
    write_json(output/'raw-evaluation.json',dict(records=records,matches=matches,priors=priors,
        elapsed_s=time.time()-started,manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        inference_uses_current_height_truth=False,acceptance_uses_current_truth=False,
        production_enabled=False,hardware_qualified=False))
    finalize(output)


def summarize(rs):
    accepted=[r for r in rs if r['accepted'] and r['predicted_height_m'] is not None]
    errors=np.array([r['absolute_error_m'] for r in accepted])
    passes=sum(r['absolute_error_m']<=r['tolerance_m'] for r in accepted)
    return dict(frames=len(rs),outputs=len(accepted),coverage_pct=100*len(accepted)/max(len(rs),1),
        mae_mm=float(errors.mean()*1000) if len(errors) else None,
        p95_mm=float(np.percentile(errors,95)*1000) if len(errors) else None,
        absrel_pct=float(np.mean([r['absolute_error_m']/r['true_height_m'] for r in accepted])*100) if len(errors) else None,
        pass_pct=100*passes/len(accepted) if accepted else None,
        effective_pass_pct=100*passes/max(len(rs),1),
        inaccurate_acceptance_pct=100*(len(accepted)-passes)/max(len(rs),1),
        evaluable_output_pct=100. if accepted else None)


def finalize(output):
    data=json.loads((output/'raw-evaluation.json').read_text());records=data['records']
    gates={}
    # Development-only gate fitting. No condition label enters the gate.
    for family in ('active_stereo','structured_light'):
        rs=[r for r in records if r['split']=='development' and r['family']==family
            and r['optimized'] and r['method']=='3_prior_physical']
        if not rs:continue
        candidates=[]
        for multiplier in (.25,.5,1.,2.,4.,8.,16.,1e6):
            trial=[dict(r,accepted=gate(r,multiplier)) for r in rs]
            metrics=summarize(trial)
            candidates.append(dict(multiplier=multiplier,**metrics))
        feasible=[c for c in candidates if c['outputs'] and c['pass_pct']>=90 and c['inaccurate_acceptance_pct']<=5]
        chosen=max(feasible,key=lambda c:(c['effective_pass_pct'],c['coverage_pct'])) if feasible else None
        gates[family]=dict(selected_multiplier=chosen['multiplier'] if chosen else None,
                           candidates=candidates,fit_split='development_only')
    write_json(output/'frozen-gates.json',gates)
    groups=defaultdict(list);ranges=defaultdict(list)
    for r in records:
        if r['split'] not in ('test','pilot'):continue
        variants=[('observable_only',r)]
        if r['optimized'] and r['method'] in ('3_prior_physical','5_prior_physical'):
            g=gates.get(r['family'],{}).get('selected_multiplier')
            variants.append(('validation_gate',dict(r,accepted=gate(r,g))))
        for gate_name,row in variants:
            key=(r['family'],r['condition'],r['optimized'],r['method'],gate_name)
            groups[key].append(row);ranges[key+(r['camera_to_bottom_m'],)].append(row)
    result=[dict(family=f,condition=c,optimized=o,method=m,gate=g,**summarize(rs))
            for (f,c,o,m,g),rs in groups.items()]
    by_range=[dict(family=f,condition=c,optimized=o,method=m,gate=g,camera_to_bottom_m=b,**summarize(rs))
              for (f,c,o,m,g,b),rs in ranges.items()]
    match_groups=defaultdict(list)
    for r in data['matches']:
        if r['split'] in ('test','pilot') and r['role']=='test':match_groups[(r['family'],r['condition'],r['optimized'])].append(r)
    matched=[]
    for (family,condition,opt),rs in match_groups.items():
        valid=[r for r in rs if r['epe_px'] is not None]
        matched.append(dict(family=family,condition=condition,optimized=opt,frames=len(rs),
            regions_mean=float(np.mean([r['regions'] for r in rs])),
            frame_mean_epe_px=float(np.mean([r['epe_px'] for r in valid])) if valid else None,
            frame_mean_within_1px_pct=float(np.mean([r['within_1px_pct'] for r in valid])) if valid else None,
            matching_ms_p50=float(np.median([r['matching_ms'] for r in rs])),
            matching_ms_p95=float(np.percentile([r['matching_ms'] for r in rs],95))))
    write_json(output/'summary.json',dict(summary=result,by_range=by_range,matching=matched,
        gate_fitting_uses_test=False,known_prior_counts=[0,1,3,5],
        flat_uniform_liquid_only=True,scattering_is_approximate=True,
        synthetic_image_matching_not_hardware=True,production_enabled=False))


def gate(record,multiplier):
    if multiplier is None or not record['observable']:return False
    d=record['diagnostics'];u=d.get('uncertainty_m')
    return bool(u is not None and np.isfinite(u) and u<=multiplier*record['predicted_tolerance_m'])


def main():
    cv2.setNumThreads(4)
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('generate','evaluate','finalize'))
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path)
    ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    if args.command=='generate':generate(args.data,args.pilot)
    elif args.command=='evaluate':evaluate(args.data,args.output)
    else:finalize(args.output)


if __name__=='__main__':main()
