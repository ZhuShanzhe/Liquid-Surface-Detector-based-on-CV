"""Offline calibration diagnostics; audit labels never enter production fitting."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import least_squares
from liquid_depth.refraction_inverse_v23 import residuals,infer

ROOT=Path('/root/autodl-tmp/liquid-depth-data/synthetic/image-refraction-v23-final-20261011')
OUT=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/prior-reliability-v24-20261011')

def observation(c):
    with np.load(Path(c['folder'])/'matches_optimized.npz') as f:obs={k:f[k] for k in f.files}
    obs['regions']=len(obs['weights']);obs['visibility']=0.
    return obs

def main():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'manifest.json').read_text());records=[]
    for c in manifest['configs']:
        observations=[observation(x) for x in c['captures']]
        cal=json.loads((Path(c['captures'][0]['folder'])/'calibration.json').read_text())
        for count,ids in [(3,[0,2,4]),(5,[0,1,2,3,4])]:
            hs=[c['captures'][i]['known_liquid_height_m'] for i in ids];obs=[observations[i] for i in ids]
            def fit(keep):
                def fun(p):return np.concatenate([(residuals(h,p[0],p[1],o,cal)*np.sqrt(np.maximum(o['weights'],.02))[:,None]/np.sqrt(len(o['weights']))).ravel() for i,(o,h) in enumerate(zip(obs,hs)) if i in keep])
                return least_squares(fun,[1.333,0],bounds=([1.2,-2],[1.6,2]),loss='soft_l1',f_scale=.025,max_nfev=100)
            row=dict(config=c['id'],family=c['family'],condition=c['condition'],split=c['split'],bottom_m=cal['bottom_m'],count=count,regions=[o['regions'] for o in obs])
            if min(row['regions'])<20:row['failure']='insufficient';records.append(row);continue
            fitted=fit(set(range(count)));p=fitted.x
            loo=np.array([fit(set(range(count))-{i}).x for i in range(count)])
            centered=loo-loo.mean(0);cov=(count-1)/count*centered.T@centered
            singular=np.linalg.svd(fitted.jac,compute_uv=False)
            errors=[]
            for i,(o,h) in enumerate(zip(obs,hs)):
                pred=infer(o,dict(index=loo[i,0],bias_px=loo[i,1]),cal)['height_m']
                errors.append((pred-h)*1000 if pred is not None else None)
            row.update(index=float(p[0]),bias_px=float(p[1]),index_sd=float(np.sqrt(cov[0,0])),bias_sd=float(np.sqrt(cov[1,1])),
                correlation=float(cov[0,1]/max(np.sqrt(cov[0,0]*cov[1,1]),1e-15)),jac_condition=float(singular[0]/max(singular[-1],1e-15)),loo_error_mm=errors)
            row['failure']='unstable' if p[0]<=1.205 or p[0]>=1.595 or row['index_sd']>.04 else ('loo' if any(e is None or abs(e)>max(5,50*h) for e,h in zip(errors,hs)) else 'none')
            values=[]
            for capture,o in zip(c['captures'][5:],observations[5:]):
                pred=infer(o,dict(index=p[0],bias_px=p[1]),cal)
                with np.load(capture['evaluator_truth']) as truth:h=float(truth['height_m'])
                if pred['observable']:values.append(dict(error_m=abs(pred['height_m']-h),rel=abs(pred['height_m']-h)/h,passed=abs(pred['height_m']-h)<=max(.005,.02*h)))
            row['diagnostic_bypass_outputs']=values
            records.append(row)
    (OUT/'calibration-audit.json').write_text(json.dumps(records,indent=2,allow_nan=False))
    for family in ('active_stereo','structured_light'):
        rows=[r for r in records if r['split']=='test' and r['count']==3 and r['family']==family]
        print(family,dict(Counter(r['failure'] for r in rows)),flush=True)
        for r in rows:
            if r['failure']!='none':print(r['bottom_m'],r['condition'],r['failure'],round(r.get('index',0),4),round(r.get('correlation',0),3),r.get('loo_error_mm'),flush=True)

if __name__=='__main__':main()
