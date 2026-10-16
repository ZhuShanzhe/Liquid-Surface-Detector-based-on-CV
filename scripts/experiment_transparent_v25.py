"""Compare all measurable target frames without coverage or confidence filtering."""
import argparse, json, hashlib, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from liquid_depth import prior_reliability_v24 as v24
from liquid_depth import transparent_diagnostic_v25 as v25
from liquid_depth import refraction_inverse_v23 as v23

ROOT=Path('/root/autodl-tmp')
OUT=ROOT/'liquid-depth-artifacts/evaluation/transparent-diagnostic-v25-20261011'
DEV=ROOT/'liquid-depth-data/synthetic/image-refraction-v23-final-20261011/manifest.json'
OLDTEST=ROOT/'liquid-depth-data/synthetic/prior-reliability-v24b-20261011/manifest.json'
METHODS=['v23_3','v23_5','v24_affine_5','v24_piecewise_5','v24_pchip_5','pixel_none_5','pixel_affine_5','pixel_pchip_5']

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def observation(c):
    with np.load(Path(c['folder'])/'matches_optimized.npz') as f:o={k:f[k] for k in f.files}
    o['regions']=len(o['weights']);return o

def metrics(rows):
    finite=[r for r in rows if r['predicted_height_m'] is not None]
    e=np.array([r['absolute_error_m'] for r in finite]);h=np.array([r['true_height_m'] for r in finite])
    t=np.maximum(.005,.02*h);n=len(rows)
    return dict(frames=n,finite_outputs=len(finite),finite_rate_pct=100*len(finite)/max(n,1),
        mae_mm=float(e.mean()*1000) if len(e) else None,absrel_pct=float(np.mean(e/h)*100) if len(e) else None,
        pass_pct=float(np.mean(e<=t)*100) if len(e) else None,
        all_frame_pass_pct=float(np.sum(e<=t)*100/max(n,1)),
        p95_mm=float(np.percentile(e,95)*1000) if len(e) else None,
        normalized_error=float(np.mean(e/t)) if len(e) else None,
        signed_bias_mm=float(np.mean([r['predicted_height_m']-r['true_height_m'] for r in finite])*1000) if finite else None)

def evaluate(configs,label):
    OUT.mkdir(parents=True,exist_ok=True);records=[];fits=[]
    for ci,c in enumerate(configs):
        v25.require_target(c['condition'])
        cs=c['captures'];obs=[observation(x) for x in cs]
        cal=json.loads((Path(cs[0]['folder'])/'calibration.json').read_text())
        for method in METHODS:
            ids=[0,2,4] if method=='v23_3' else list(range(5))
            hs=[cs[i]['known_liquid_height_m'] for i in ids]
            start=time.perf_counter();failure=None
            try:
                if method.startswith('v23'):p=v23.fit_prior([obs[i] for i in ids],hs,cal)
                elif method.startswith('v24'):p=v24.fit_prior([obs[i] for i in ids],hs,cal,method.split('_')[1])
                else:p=v25.fit_prior([obs[i] for i in ids],hs,cal,c['condition'],method.split('_')[1])
            except ValueError as exc:p=None;failure=str(exc)
            fits.append(dict(config=c['id'],method=method,prior=p,failure=failure,fit_ms=(time.perf_counter()-start)*1000))
            for j,capture in enumerate(cs[5:],5):
                start=time.perf_counter()
                if p is None:pred=dict(height_m=None,reasons=['prior_failed:'+failure])
                elif method.startswith('v23'):pred=v23.infer(obs[j],p,cal)
                elif method.startswith('v24'):pred=v24.infer(obs[j],p,cal)
                else:pred=v25.infer(obs[j],p,cal)
                elapsed=(time.perf_counter()-start)*1000
                hpred=pred['height_m']
                if hpred is not None and not np.isfinite(hpred):hpred=None
                # Truth is accessed exclusively after prediction; no acceptance filtering.
                with np.load(capture['evaluator_truth']) as f:h=float(f['height_m'])
                records.append(dict(config_id=c['id'],sample_id=j,family=c['family'],condition=c['condition'],
                    bottom_m=cal['bottom_m'],method=method,true_height_m=h,predicted_height_m=hpred,
                    absolute_error_m=abs(hpred-h) if hpred is not None else None,
                    tolerance_m=max(.005,.02*h),regions=obs[j]['regions'],diagnostics=pred,inference_ms=elapsed))
        print(label,ci+1,len(configs),flush=True)
    dump(OUT/(label+'-raw.json'),dict(records=records,fits=fits))
    groups=defaultdict(list)
    for r in records:groups[r['family'],r['condition'],r['method']].append(r)
    summary=[dict(family=f,condition=c,method=m,**metrics(rs)) for (f,c,m),rs in groups.items()]
    dump(OUT/(label+'-summary.json'),summary)
    for row in summary:print(row,flush=True)
    return records,summary

def select(summary):
    chosen={}
    for family in ('active_stereo','structured_light'):
        for condition in v25.TARGET_CONDITIONS:
            # No coverage constraint or threshold tuning. Numerical failures get a
            # fixed 10-tolerance diagnostic cost; risk is not used to hide outputs.
            options=[]
            for r in summary:
                if r['family']!=family or r['condition']!=condition or r['method'].startswith('v23'):continue
                loss=((r['normalized_error'] or 0)*r['finite_outputs']+10*(r['frames']-r['finite_outputs']))/r['frames']
                options.append((loss,r['method']))
            chosen[family+'/'+condition]=min(options)[1]
    dump(OUT/'frozen-selection.json',dict(methods=chosen,selection='target-only development mean tolerance-normalized error; missing=10',coverage_gate=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['develop','diagnose','test']);ap.add_argument('--manifest');args=ap.parse_args()
    path=Path(args.manifest) if args.manifest else (DEV if args.stage=='develop' else OLDTEST)
    configs=[c for c in json.loads(path.read_text())['configs'] if c['condition'] in v25.TARGET_CONDITIONS]
    if args.stage=='develop':configs=[c for c in configs if c['split'] in ('development','test')]
    records,summary=evaluate(configs,args.stage)
    if args.stage=='develop':select(summary)
