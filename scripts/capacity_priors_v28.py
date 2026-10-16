"""Capacity-limited five-level calibration; unfiltered research evaluation."""
import argparse,json,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import cv2,numpy as np
from liquid_depth.refraction_images_v23 import render_capture
from liquid_depth.refraction_matching_v23 import stereo_matches
from liquid_depth.refraction_matching_v26 import decode_structured
from experiment_transparent_v27 import load,predict26,rows,summary,dump,POLICY26

ROOT=Path('/root/autodl-tmp')
OUT=ROOT/'liquid-depth-artifacts/evaluation/transparent-capacity-v28-20261011'
DATA=ROOT/'liquid-depth-data/synthetic/transparent-capacity-v28-20261011'
REPO=Path(__file__).resolve().parents[1]
LAYOUTS={'capacity_uniform':[.10,.30,.50,.70,.90],
         'low_dense':[.03,.08,.22,.48,.90],
         'endpoints':[.02,.12,.35,.65,.95],
         'wide_uniform':[.05,.275,.50,.725,.95]}

def progress(stage,**kw):dump(OUT/'progress.json',dict(stage=stage,**kw))

def validate_layout(values,capacity,bottom):
    if not 0<capacity<bottom:raise ValueError('invalid_container_capacity')
    hs=np.asarray(values,float)*capacity
    if len(hs)!=5 or not np.isfinite(hs).all() or not np.all(np.diff(hs)>0):raise ValueError('five_distinct_levels_required')
    if not np.all((hs>0)&(hs<capacity)):raise ValueError('levels_outside_container')
    return hs

def specifications(split):
    if split=='development':
        distances=[3.,8.5];capacities=[.20,.50];indices=[1.325,1.405];seed0=589317
        unknown=[.04,.09,.17,.26,.38,.52,.64,.76,.86,.97]
    elif split=='test':
        distances=[2.2,6.5,9.8];capacities=[.15,.35,.75];indices=[1.315,1.445];seed0=691753
        unknown=[.035,.075,.15,.23,.37,.49,.61,.73,.84,.975]
    else:raise ValueError('unsupported_split')
    jobs=[]
    for fi,(family,baseline) in enumerate([('active_stereo',.055),('structured_light',.035)]):
        for ci,(b,cap,n) in enumerate([(b,c,n) for b in distances for c in capacities for n in indices]):
            for condition in ('transparent','translucent_mild'):
                jobs.append(dict(split=split,id=f'{family}_{ci:02d}_{condition}',family=family,condition=condition,
                    bottom=b,capacity=cap,index=n,baseline=baseline,seed=seed0+ci*263+fi*10000,unknown=unknown))
    return jobs

def render_config(job):
    cv2.setNumThreads(1)
    b=job['bottom'];cap=job['capacity'];folder=DATA/job['split']/job['id'];priors={};tests=[]
    for values in LAYOUTS.values():validate_layout(values,cap,b)
    for role,ratios in [('prior',sorted(set(v for values in LAYOUTS.values() for v in values))),('test',job['unknown'])]:
        for j,ratio in enumerate(ratios):
            sample=folder/f'{role}_{j:02d}';sample.mkdir(parents=True,exist_ok=True)
            images,cal,truth=render_capture(job['seed'],cap*ratio,job['index'],b,job['baseline'],job['condition'],job['family'])
            cal['container_max_height_m']=cap
            for name,img in images.items():
                if not cv2.imwrite(str(sample/(name+'.png')),img):raise IOError('image_write_failed')
            dump(sample/'calibration.json',cal);np.savez_compressed(sample/'evaluator_truth.npz',**truth)
            obs=stereo_matches(images['left'],images['right'],cal,True) if job['family']=='active_stereo' else decode_structured(images,cal,0,24)
            np.savez_compressed(sample/'observation.npz',**{k:obs[k] for k in ('left_q','right_q','left_pixels','right_pixels','weights')})
            record=dict(folder=str(sample),role=role,evaluator_truth=str(sample/'evaluator_truth.npz'))
            if role=='prior':
                rng=np.random.default_rng(job['seed']+int(round(ratio*1e6))*65537)
                record['known_liquid_height_m']=float(cap*ratio+rng.normal(0,.0005+.001*cap*ratio))
                priors[ratio]=record
            else:tests.append(record)
    return dict(id=job['id'],family=job['family'],condition=job['condition'],camera_to_bottom_m=b,
        container_max_height_m=cap,priors={k:[priors[v] for v in values] for k,values in LAYOUTS.items()},tests=tests)

def generate(split):
    folder=DATA/split
    if folder.exists():raise FileExistsError('split exists; do not overwrite generated evidence')
    folder.mkdir(parents=True)
    jobs=specifications(split);configs=[]
    with ProcessPoolExecutor(max_workers=min(4,os.cpu_count() or 1)) as pool:
        futures=[pool.submit(render_config,j) for j in jobs]
        for f in as_completed(futures):
            configs.append(f.result());progress('generate_'+split,configs=len(configs),total_configs=len(jobs))
            print('GENERATE',split,len(configs),len(jobs),flush=True)
    dump(folder/'manifest.json',dict(configs=sorted(configs,key=lambda c:c['id']),layouts=LAYOUTS,split=split,
        prior_budget=5,ratios_referenced_to='container_max_height_not_camera_distance',
        inherited_image_proxy=True,finite_vessel_walls_rendered=False,shared_snell_model=True))

def result_rows(c,preds,stage,layout):
    rs=rows(c,preds,stage,layout)
    for r in rs:
        r['container_max_height_m']=c['container_max_height_m']
        if r['predicted_height_m'] is not None and not 0<r['predicted_height_m']<c['container_max_height_m']:
            # Preserve finite out-of-capacity estimates; do not silently clip or reject.
            r['diagnostics']=dict(r['diagnostics'],reasons=r['diagnostics'].get('reasons',[])+['outside_container_capacity'])
    return rs

def score(rs):
    a=summary(rs);b=summary([r for r in rs if r['bottom_m']>=6])
    return .5*a['normalized_mean']+.5*b['normalized_mean']+.2*a['normalized_p90']+10*(1-a['outputs']/max(a['frames'],1))

def select():
    configs=json.loads((DATA/'development/manifest.json').read_text())['configs'];records=[];selected={};groups=[]
    for family in ('active_stereo','structured_light'):
        for condition in ('transparent','translucent_mild'):
            cs=[c for c in configs if c['family']==family and c['condition']==condition];choices=[]
            for layout in LAYOUTS:
                rs=[]
                for c in cs:
                    preds,_=predict26(c,layout);rs+=result_rows(c,preds,layout,layout)
                choices.append(dict(layout=layout,score=score(rs),overall=summary(rs),far=summary([r for r in rs if r['bottom_m']>=6])))
                records+=rs;print('LAYOUT',family,condition,choices[-1],flush=True)
            best=min(choices,key=lambda m:m['score']);selected[family+'/'+condition]=best['layout']
            groups.append(dict(family=family,condition=condition,candidates=choices,selected=best))
    dump(OUT/'development-raw.json',records);dump(OUT/'layout-candidates.json',groups)
    dump(OUT/'frozen-policy.json',dict(selection=selected,layouts=LAYOUTS,prior_budget=5,
        confidence_gate=False,coverage_gate=False,test_used=False,measurement_core='unchanged_V26',
        residual_V27_field_not_enabled=True,capacity_clipping=False))

if __name__=='__main__':
    cv2.setNumThreads(4)
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['generate_dev','select']);args=ap.parse_args()
    if args.stage=='generate_dev':generate('development')
    else:select()
