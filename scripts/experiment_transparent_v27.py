"""Five-prior budget ablation and frozen, fully unfiltered liquid-height tests."""
import argparse,hashlib,json,time
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from liquid_depth.refraction_images_v23 import render_capture
from liquid_depth.refraction_matching_v23 import stereo_matches
from liquid_depth.refraction_matching_v26 import decode_structured
from liquid_depth import transparent_diagnostic_v25 as v25
from liquid_depth import transparent_refinement_v26 as v26

ROOT=Path('/root/autodl-tmp')
OUT=ROOT/'liquid-depth-artifacts/evaluation/transparent-shallow-v27-20261011'
DATA=ROOT/'liquid-depth-data/synthetic/transparent-shallow-v27-20261011'

class _LazyV26Policy:
    """Keep historical utilities importable when V26 artifacts are absent."""

    def __init__(self):
        self._selection = None

    def __getitem__(self, key):
        if self._selection is None:
            path = ROOT/'liquid-depth-artifacts/evaluation/transparent-refinement-v26-20261011/frozen-policy.json'
            if not path.is_file():
                raise FileNotFoundError(f'Historical V26 policy unavailable: {path}; reproduce V26 first')
            self._selection = json.loads(path.read_text())['selection']
        return self._selection[key]

POLICY26=_LazyV26Policy()
LAYOUTS={'original':[.10,.22,.35,.50,.65],
         'near_balanced':[.03,.08,.22,.48,.76],
         'shallow_focus':[.025,.055,.12,.32,.70],
         'uniform_span':[.025,.21,.395,.58,.765]}
REPO=Path(__file__).resolve().parents[1]

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def progress(stage,**extra):dump(OUT/'progress.json',dict(stage=stage,**extra))

def generate(split,selected=None):
    folder=DATA/split
    if (folder/'manifest.json').exists():raise FileExistsError('manifest already exists')
    if split=='development':
        distances=[.4,2.5,7.5,9.5];indices=[1.33,1.40];seed0=371209
        unknown=[.018,.045,.075,.105,.16,.27,.44,.61,.73,.80]
        layouts=LAYOUTS
    else:
        distances=[.3,.9,2.2,6.5,9.8];indices=[1.305,1.455];seed0=478921
        unknown=[.015,.04,.065,.095,.145,.24,.41,.57,.69,.81]
        # Evaluate all predeclared layouts after freezing; never reselect on test.
        layouts=LAYOUTS
    configs=[];total=len(distances)*len(indices)*4
    for fi,(family,baseline) in enumerate([('active_stereo',.055),('structured_light',.035)]):
        for ci,(bottom,n) in enumerate([(b,n) for b in distances for n in indices]):
            seed=seed0+ci*241+fi*10000
            for condition in ('transparent','translucent_mild'):
                config_id=f'{family}_{ci:02d}_{condition}';base=folder/config_id
                needed=layouts
                ratios=sorted(set(v for values in needed.values() for v in values))
                priors={};tests=[]
                for role,values in [('prior',ratios),('test',unknown)]:
                    for j,ratio in enumerate(values):
                        cap=base/f'{role}_{j:02d}';cap.mkdir(parents=True,exist_ok=True)
                        images,cal,truth=render_capture(seed,bottom*ratio,n,bottom,baseline,condition,family)
                        for name,img in images.items():
                            if not cv2.imwrite(str(cap/(name+'.png')),img):raise IOError('image write failed')
                        dump(cap/'calibration.json',cal)
                        np.savez_compressed(cap/'evaluator_truth.npz',**truth)
                        obs=stereo_matches(images['left'],images['right'],cal,True) if family=='active_stereo' else decode_structured(images,cal,0,24)
                        np.savez_compressed(cap/'observation.npz',**{k:obs[k] for k in ('left_q','right_q','left_pixels','right_pixels','weights')})
                        record=dict(folder=str(cap),role=role,evaluator_truth=str(cap/'evaluator_truth.npz'))
                        if role=='prior':
                            # Same level always shares the exact observation and label across layouts.
                            rng=np.random.default_rng(seed+int(round(ratio*1e6))*65537)
                            record['known_liquid_height_m']=float(bottom*ratio+rng.normal(0,.0005+.001*bottom*ratio))
                            priors[ratio]=record
                        else:tests.append(record)
                configs.append(dict(id=config_id,family=family,condition=condition,camera_to_bottom_m=bottom,
                    priors={k:[priors[v] for v in values] for k,values in needed.items()},tests=tests))
                progress('generate_'+split,configs=len(configs),total_configs=total)
                print('GENERATE',split,len(configs),total,flush=True)
    manifest=dict(configs=configs,split=split,layouts=layouts,unknown_ratios=unknown,indices=indices,
        seed0=seed0,shared_forward_inverse=True,strong_scatter=False,unknown_truth_for_scoring_only=True)
    dump(folder/'manifest.json',manifest)

def load(c,layout):
    caps=c['priors'][layout]+c['tests'];obs=[]
    for cap in caps:
        with np.load(Path(cap['folder'])/'observation.npz') as f:o={k:f[k] for k in f.files}
        o['regions']=len(o['weights']);obs.append(o)
    cal=json.loads((Path(caps[0]['folder'])/'calibration.json').read_text())
    labels=[cap['known_liquid_height_m'] for cap in caps[:5]]
    return obs,cal,labels

def predict26(c,layout):
    obs,cal,hs=load(c,layout);setting=POLICY26[c['family']+'/'+c['condition']]
    try:
        if c['family']=='active_stereo':
            prior=v26.fit_prior(obs[:5],hs,cal,c['condition'],setting['harmonics'],setting['ridge'])
            preds=[v26.infer(o,prior,cal,setting['correction']) for o in obs[5:]]
        else:
            prior=v25.fit_prior(obs[:5],hs,cal,c['condition'],setting['baseline_mode'])
            preds=[v25.infer(o,prior,cal) for o in obs[5:]]
        return preds,prior
    except ValueError as exc:
        return [dict(height_m=None,reasons=['prior_failed',str(exc)],qualified=False)]*len(c['tests']),dict(error=str(exc))

def rows(c,preds,stage,layout):
    result=[]
    for j,(cap,pred) in enumerate(zip(c['tests'],preds)):
        with np.load(cap['evaluator_truth']) as f:h=float(f['height_m'])
        value=pred.get('height_m');value=float(value) if value is not None and np.isfinite(value) else None
        result.append(dict(config_id=c['id'],sample_id=j,family=c['family'],condition=c['condition'],
            bottom_m=c['camera_to_bottom_m'],stage=stage,layout=layout,true_height_m=h,
            predicted_height_m=value,absolute_error_m=abs(value-h) if value is not None else None,
            tolerance_m=max(.005,.02*h),diagnostics=pred,qualified=False,gate_mode='all_finite'))
    if len(result)!=len(c['tests']):raise RuntimeError('prediction count mismatch')
    return result

def summary(rs):
    finite=[r for r in rs if r['predicted_height_m'] is not None]
    e=np.array([r['absolute_error_m'] for r in finite]);h=np.array([r['true_height_m'] for r in finite])
    z=e/np.maximum(.005,.02*h)
    return dict(frames=len(rs),outputs=len(finite),numeric_output_pct=100*len(finite)/max(len(rs),1),
        passed=int(np.sum(z<=1)),pass_pct=100*int(np.sum(z<=1))/max(len(rs),1),
        mae_mm=float(e.mean()*1000) if len(e) else None,absrel_pct=float(np.mean(e/h)*100) if len(e) else None,
        p95_mm=float(np.percentile(e,95)*1000) if len(e) else None,max_mm=float(e.max()*1000) if len(e) else None,
        normalized_mean=float(np.mean(np.minimum(z,20))) if len(e) else 20.,
        normalized_p90=float(np.percentile(z,90)) if len(e) else 20.)

def score(rs):
    all_=summary(rs);far=summary([r for r in rs if r['bottom_m']>=6 and r['true_height_m']/r['bottom_m']<.18])
    return .5*all_['normalized_mean']+.5*far['normalized_mean']+.2*all_['normalized_p90']+10*(1-all_['outputs']/max(len(rs),1))

def select_layout():
    configs=json.loads((DATA/'development/manifest.json').read_text())['configs']
    selected={};candidates=[];records=[]
    for family in ('active_stereo','structured_light'):
        for condition in ('transparent','translucent_mild'):
            subset=[c for c in configs if c['family']==family and c['condition']==condition];choices=[]
            for layout in LAYOUTS:
                rs=[]
                for c in subset:
                    preds,_=predict26(c,layout);rs+=rows(c,preds,'layout',layout)
                metric=dict(layout=layout,score=score(rs),overall=summary(rs),
                    far_shallow=summary([r for r in rs if r['bottom_m']>=6 and r['true_height_m']/r['bottom_m']<.18]))
                choices.append(metric);records+=rs;print('LAYOUT',family,condition,metric,flush=True)
            best=min(choices,key=lambda r:r['score']);selected[family+'/'+condition]=best['layout']
            candidates.append(dict(family=family,condition=condition,candidates=choices,selected=best['layout']))
    dump(OUT/'layout-candidates.json',candidates);dump(OUT/'layout-development-raw.json',records)
    dump(OUT/'layout-policy.json',dict(selection=selected,layouts=LAYOUTS,prior_budget=5,test_used=False,
        objective='0.5*all_mean_capped_normalized + 0.5*far_shallow_mean_capped_normalized + 0.2*all_p90 + 10*missing_fraction'))

if __name__=='__main__':
    cv2.setNumThreads(4)
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['generate_dev','select_layout']);args=ap.parse_args()
    if args.stage=='generate_dev':generate('development')
    else:select_layout()
