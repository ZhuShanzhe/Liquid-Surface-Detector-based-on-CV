"""Sequential, development-only selection without selective-output filtering."""
import argparse,json,time
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from liquid_depth import transparent_diagnostic_v25 as v25
from liquid_depth import transparent_refinement_v26 as v26
from liquid_depth.refraction_matching_v26 import decode_structured

ROOT=Path('/root/autodl-tmp')
BASE=ROOT/'liquid-depth-artifacts/evaluation/transparent-diagnostic-v25-20261011'
OUT=ROOT/'liquid-depth-artifacts/evaluation/transparent-refinement-v26-20261011'
DEV=ROOT/'liquid-depth-data/synthetic/image-refraction-v23-final-20261011/manifest.json'
V25TEST=ROOT/'liquid-depth-data/synthetic/transparent-diagnostic-v25-20261011/manifest.json'
CACHE=ROOT/'liquid-depth-data/synthetic/transparent-refinement-v26-cache-20261011'
BASE_POLICY=json.loads((BASE/'frozen-selection.json').read_text())['methods']

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def corrections():
    yield dict(kind='none')
    for shrink in (.25,.5,.75,1.):
        for slope in (.05,.15,.3,.6):
            for tail in ('constant','taper','zero'):
                yield dict(kind='bounded',shrink=shrink,slope_cap=slope,tail=tail)

def id_for(c):return json.dumps(c,sort_keys=True,separators=(',',':'))

def summarize(rows):
    rs=[r for r in rows if r['predicted_height_m'] is not None]
    e=np.array([r['absolute_error_m'] for r in rs]);h=np.array([r['true_height_m'] for r in rs]);t=np.maximum(.005,.02*h)
    z=e/t
    return dict(frames=len(rows),outputs=len(rs),coverage_pct=100*len(rs)/max(len(rows),1),
        passed=int(np.sum(e<=t)),pass_pct=float(np.mean(e<=t)*100) if len(e) else None,
        mae_mm=float(e.mean()*1000) if len(e) else None,absrel_pct=float(np.mean(e/h)*100) if len(e) else None,
        p95_mm=float(np.percentile(e,95)*1000) if len(e) else None,max_mm=float(e.max()*1000) if len(e) else None,
        score=float(np.mean(np.minimum(z,20))+.2*np.percentile(z,90)+10*(len(rows)-len(rs))/max(len(rows),1)) if len(z) else 100.)

def observation(c,variant='legacy',config_id=''):
    if variant=='legacy':path=Path(c['folder'])/'matches_optimized.npz'
    else:
        path=CACHE/config_id/Path(c['folder']).name/(variant+'.npz')
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            sigma,block=map(float,variant.split('_')[1:])
            images={p.stem:cv2.imread(str(p),cv2.IMREAD_GRAYSCALE) for p in Path(c['folder']).glob('*.png')}
            cal=json.loads((Path(c['folder'])/'calibration.json').read_text())
            result=decode_structured(images,cal,sigma,int(block))
            np.savez_compressed(path,**{k:result[k] for k in ('left_q','right_q','weights','left_pixels','right_pixels')})
    with np.load(path) as f:obs={k:f[k] for k in f.files}
    obs['regions']=len(obs['weights']);return obs

def setup(c,variant='legacy'):
    cs=c['captures'];obs=[observation(cap,variant,c['id']) for cap in cs]
    cal=json.loads((Path(cs[0]['folder'])/'calibration.json').read_text())
    known=[cap['known_liquid_height_m'] for cap in cs[:5]]
    return obs,cal,known

def row(c,j,pred,stage,truth):
    v=pred.get('height_m')
    if v is not None and not np.isfinite(v):v=None
    return dict(config_id=c['id'],sample_id=j,family=c['family'],condition=c['condition'],
        bottom_m=c['camera_to_bottom_m'],stage=stage,true_height_m=truth,predicted_height_m=v,
        absolute_error_m=abs(v-truth) if v is not None else None,tolerance_m=max(.005,.02*truth),
        diagnostics=pred,gate_mode='all_finite',qualified=False)

def truths(c):
    hs=[]
    for cap in c['captures'][5:]:
        with np.load(cap['evaluator_truth']) as f:hs.append(float(f['height_m']))
    return hs

def baseline(c,obs,cal,known):
    mode=BASE_POLICY[c['family']+'/'+c['condition']].split('_')[1]
    p=v25.fit_prior(obs[:5],known,cal,c['condition'],mode)
    return [v25.infer(o,p,cal) for o in obs[5:]]

def development():
    OUT.mkdir(parents=True,exist_ok=True)
    configs=[c for c in json.loads(DEV.read_text())['configs'] if c['condition'] in v25.TARGET_CONDITIONS]
    contexts={};all_candidates=[];selection={}
    # Stage 1 uses cached legacy observations and physically solved raw estimates.
    for c in configs:
        obs,cal,hs=setup(c);p=v26.fit_prior(obs[:5],hs,cal,c['condition'])
        contexts[c['id']]=(obs,cal,hs,p,[v26.solve_height(o,p,cal) for o in obs[5:]],truths(c))
    for condition in sorted(v25.TARGET_CONDITIONS):
        cs=[c for c in configs if c['family']=='active_stereo' and c['condition']==condition]
        candidates=[]
        for corr in corrections():
            rs=[]
            for c in cs:
                obs,cal,hs,p,preds,ts=contexts[c['id']]
                for j,(pred,t) in enumerate(zip(preds,ts),5):
                    v=pred['height_m']
                    value=v if corr['kind']=='none' else v26.bounded_height(v,p,cal,corr['shrink'],corr['slope_cap'],corr['tail'])
                    rs.append(row(c,j,dict(height_m=value),'stage1',t))
            candidates.append(dict(correction=corr,**summarize(rs)))
        best=min(candidates,key=lambda x:x['score'])
        selection['active_stereo/'+condition]=dict(correction=best['correction'],harmonics=0,ridge=.02,matching='legacy')
        all_candidates.append(dict(stage=1,family='active_stereo',condition=condition,candidates=candidates,selected=best))
        print('STAGE1',condition,best,flush=True)
    dump(OUT/'stage1-policy.json',selection)
    # Stage 2: learn periodic subpixel bias jointly with effective index and offset.
    for condition in sorted(v25.TARGET_CONDITIONS):
        cs=[c for c in configs if c['family']=='active_stereo' and c['condition']==condition]
        key='active_stereo/'+condition;corr=selection[key]['correction'];candidates=[]
        for harmonics,ridge in [(0,.02),(1,.005),(1,.02),(1,.1),(2,.02),(2,.1)]:
            rs=[];failed=[]
            for c in cs:
                obs,cal,hs,_,_,ts=contexts[c['id']]
                try:p=v26.fit_prior(obs[:5],hs,cal,c['condition'],harmonics,ridge)
                except ValueError as exc:p=None;failed.append(dict(config=c['id'],reason=str(exc)))
                for j,t in enumerate(ts,5):
                    pred=v26.infer(obs[j],p,cal,corr) if p else dict(height_m=None,reasons=['prior_failed'])
                    rs.append(row(c,j,pred,'stage2',t))
            candidates.append(dict(harmonics=harmonics,ridge=ridge,fit_failures=failed,**summarize(rs)))
        best=min(candidates,key=lambda x:x['score']);selection[key].update(harmonics=best['harmonics'],ridge=best['ridge'])
        all_candidates.append(dict(stage=2,family='active_stereo',condition=condition,candidates=candidates,selected=best))
        print('STAGE2',condition,best,flush=True)
    dump(OUT/'stage2-policy.json',selection)
    # Stage 3 changes phase decoding only; V25's per-condition correction stays fixed.
    for condition in sorted(v25.TARGET_CONDITIONS):
        cs=[c for c in configs if c['family']=='structured_light' and c['condition']==condition]
        candidates=[]
        for variant in ['legacy','phase_0_16','phase_0.8_16','phase_0_24','phase_0.8_24']:
            rs=[];failed=[]
            for c in cs:
                obs,cal,hs=setup(c,variant)
                try:preds=baseline(c,obs,cal,hs)
                except ValueError as exc:preds=[dict(height_m=None,reasons=[str(exc)])]*len(obs[5:]);failed.append(dict(config=c['id'],reason=str(exc)))
                for j,(pred,t) in enumerate(zip(preds,contexts[c['id']][-1]),5):rs.append(row(c,j,pred,'stage3',t))
            candidates.append(dict(matching=variant,fit_failures=failed,**summarize(rs)))
            print('PHASE',condition,variant,summarize(rs),flush=True)
        best=min(candidates,key=lambda x:x['score'])
        selection['structured_light/'+condition]=dict(matching=best['matching'],baseline_mode=BASE_POLICY['structured_light/'+condition].split('_')[1])
        all_candidates.append(dict(stage=3,family='structured_light',condition=condition,candidates=candidates,selected=best))
        print('STAGE3',condition,best,flush=True)
    dump(OUT/'development-candidates.json',all_candidates)
    dump(OUT/'frozen-policy.json',dict(selection=selection,selection_data=str(DEV),
        former_v23_test_repurposed_as_development=True,v25_test_used_for_selection=False,
        strong_scatter_included=False,coverage_filter=False,score='mean capped tolerance-normalized error + 0.2*p90'))

def evaluate(manifest_path,label):
    policy=json.loads((OUT/'frozen-policy.json').read_text())['selection']
    configs=[c for c in json.loads(Path(manifest_path).read_text())['configs'] if c['condition'] in v25.TARGET_CONDITIONS]
    records=[];fit_records=[]
    stage1=json.loads((OUT/'stage1-policy.json').read_text())
    for ci,c in enumerate(configs):
        obs,cal,hs=setup(c);key=c['family']+'/'+c['condition'];setting=policy[key]
        ts=truths(c);predictions={}
        try:predictions['v25']=baseline(c,obs,cal,hs)
        except ValueError as exc:predictions['v25']=[dict(height_m=None,reasons=[str(exc)])]*len(ts)
        if c['family']=='active_stereo':
            for stage,harmonics,ridge in [('bounded',0,.02),('pixel_bias',setting['harmonics'],setting['ridge'])]:
                try:
                    p=v26.fit_prior(obs[:5],hs,cal,c['condition'],harmonics,ridge)
                    predictions[stage]=[v26.infer(o,p,cal,stage1[key]['correction']) for o in obs[5:]]
                    fit_records.append(dict(config_id=c['id'],stage=stage,prior=p))
                except ValueError as exc:predictions[stage]=[dict(height_m=None,reasons=[str(exc)])]*len(ts)
            predictions['phase']=predictions['pixel_bias']
        else:
            predictions['bounded']=predictions['v25'];predictions['pixel_bias']=predictions['v25']
            new_obs,_,_=setup(c,setting['matching'])
            try:predictions['phase']=baseline(c,new_obs,cal,hs)
            except ValueError as exc:predictions['phase']=[dict(height_m=None,reasons=[str(exc)])]*len(ts)
        for stage,preds in predictions.items():
            for j,(pred,t) in enumerate(zip(preds,ts),5):records.append(row(c,j,pred,stage,t))
        print('EVALUATED',label,ci+1,len(configs),flush=True)
    groups=defaultdict(list)
    for r in records:groups[r['family'],r['condition'],r['stage']].append(r)
    summary=[dict(family=f,condition=c,stage=s,**summarize(rs)) for (f,c,s),rs in groups.items()]
    dump(OUT/(label+'-raw.json'),dict(records=records,fits=fit_records))
    dump(OUT/(label+'-summary.json'),summary)
    for r in summary:print(r,flush=True)

if __name__=='__main__':
    cv2.setNumThreads(4)
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['develop','evaluate']);ap.add_argument('--manifest',default=str(V25TEST));ap.add_argument('--label',default='v25-paired');args=ap.parse_args()
    if args.stage=='develop':development()
    else:evaluate(args.manifest,args.label)
