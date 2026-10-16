"""Finite-bottom footprint ablation, frozen new all-frame stereo test."""
import argparse,os,json,time,sys,hashlib,subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import cv2,numpy as np
from liquid_depth.finite_refraction_images_v30 import render
from liquid_depth import finite_relative_v30 as new
from liquid_depth import transparent_relative_v29 as old
from experiment_transparent_v27 import rows,summary,dump

ROOT=Path('/root/autodl-tmp');REPO=Path(__file__).resolve().parents[1]
OUT=ROOT/'liquid-depth-artifacts/evaluation/transparent-finite-v30-20261011'
DATA=ROOT/'liquid-depth-data/synthetic/transparent-finite-v30-20261011'
LEVELS=[0.,.15,.40,.65,.90]
METHODS=['v29_finite','small_lk','small_lk_corrected','small_lk_response','direct','direct_corrected','direct_response','direct_response_floor']

def progress(stage,**kw):dump(OUT/'progress.json',dict(stage=stage,**kw))

def specifications(split):
    if split=='development':
        bs=[3.4,8.6];ds=[.55,1.1];cs=[.18,.40];ns=[1.33,1.405];grains=[.020,.045];seed=1031723
        unknown=[.04,.09,.17,.26,.38,.52,.64,.76,.86,.97]
    elif split=='test':
        bs=[2.6,6.6,9.6];ds=[.4,.8,1.6];cs=[.12,.32];ns=[1.315,1.455];grains=[.015,.035];seed=1173911
        unknown=[.035,.075,.12,.18,.26,.36,.47,.58,.69,.80,.915,.975]
    else:raise ValueError('unknown_split')
    jobs=[]
    for i,(b,d,c,n,g) in enumerate((b,d,c,n,g) for b in bs for d in ds for c in cs for n in ns for g in grains):
        for condition in ('transparent','translucent_mild'):
            jobs.append(dict(split=split,id=f'finite_{i:03d}_{condition}',bottom=b,width=d,capacity=c,index=n,
                grain=g,seed=seed+353*i,condition=condition,unknown=unknown))
    return jobs

def render_config(job):
    cv2.setNumThreads(1);priors=[];tests=[];folder=DATA/job['split']/job['id']
    for role,ratios in [('prior',LEVELS),('test',job['unknown'])]:
        for j,ratio in enumerate(ratios):
            dst=folder/f'{role}_{j:02d}';dst.mkdir(parents=True)
            images,cal,masks,truth=render(job['seed'],job['capacity']*ratio,job['index'],job['bottom'],job['capacity'],job['width'],
                'transparent' if ratio==0 else job['condition'],job['grain'])
            for ch in ('left','right'):
                if not cv2.imwrite(str(dst/(ch+'.png')),images[ch]):raise IOError('write_image_failed')
                if role=='prior' and j==0:cv2.imwrite(str(folder/('roi_'+ch+'.png')),masks[ch])
            dump(dst/'calibration.json',cal);np.savez_compressed(dst/'evaluator_truth.npz',**truth)
            item=dict(folder=str(dst),evaluator_truth=str(dst/'evaluator_truth.npz'),role=role)
            if role=='prior':
                rng=np.random.default_rng(job['seed']+j*65537)
                item['known_liquid_height_m']=0. if j==0 else float(job['capacity']*ratio+rng.normal(0,.0005+.001*job['capacity']*ratio))
                priors.append(item)
            else:tests.append(item)
    return dict(id=job['id'],family='active_stereo',condition=job['condition'],camera_to_bottom_m=job['bottom'],
        container_max_height_m=job['capacity'],container_width_m=job['width'],empty_width_px=cal['empty_width_px'],
        grain_m=job['grain'],priors=priors,tests=tests,folder=str(folder))

def generate(split):
    folder=DATA/split
    if folder.exists():raise FileExistsError('preserve_existing_evidence')
    folder.mkdir(parents=True);jobs=specifications(split);configs=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(render_config,j) for j in jobs]
        for f in as_completed(futures):
            configs.append(f.result());progress('generate_'+split,configs=len(configs),total=len(jobs));print('GENERATE',split,len(configs),len(jobs),flush=True)
    dump(folder/'manifest.json',dict(configs=sorted(configs,key=lambda c:c['id']),prior_budget=5,levels=LEVELS,
        operator_roi='fixed_empty_bottom',supersample=3,metric_texture=True,opaque_sides=True,glass_walls=False,
        confidence_gate=False,coverage_gate=False,capacity_clipping=False))

def images(c):
    caps=c['priors']+c['tests'];ims={}
    for i,cap in enumerate(caps):
        for ch in ('left','right'):
            im=cv2.imread(str(Path(cap['folder'])/(ch+'.png')),0)
            if im is None:raise ValueError('missing_image')
            ims[i,ch]=im
    masks={ch:cv2.imread(str(Path(c['folder'])/('roi_'+ch+'.png')),0) for ch in ('left','right')}
    cal=json.loads((Path(caps[0]['folder'])/'calibration.json').read_text())
    return ims,masks,cal,[p['known_liquid_height_m'] for p in c['priors']]

def ensure_cache(c,kind,ims,masks,cal):
    path=Path(c['folder'])/('tracks_'+kind+'.npz')
    if path.exists():
        cache={}
        with np.load(path) as f:
            for k in f.files:
                a,b,ch,field=k.split('_',3);p=cache.setdefault((int(a),int(b),ch),dict(channel=ch));p[field]=f[k]
        for p in cache.values():p['regions']=len(p['weights'])
        return cache
    fn={'v29_finite':new.baseline_pair,'small_lk':new.small_lk,'direct':new.direct_pair}[kind];cache={}
    for a in range(5):
        for b in range(a+1,5+len(c['tests'])):
            for ch in ('left','right'):
                cache[a,b,ch]=fn(ims[a,ch],ims[b,ch],cal,masks[ch],ch)
    data={f'{a}_{b}_{ch}_{key}':p[key] for (a,b,ch),p in cache.items() for key in ('anchor_q','current_q','weights')}
    np.savez_compressed(path,**data);return cache

def evaluate_config(c):
    cv2.setNumThreads(1);ims,masks,cal,hs=images(c);records=[];fits=[];obsstats=[]
    for kind in ('v29_finite','small_lk','direct'):
        start=time.perf_counter();cache=ensure_cache(c,kind,ims,masks,cal)
        ps=[(a,b,p) for (a,b,ch),p in cache.items() if b<5]
        obsstats.append(dict(kind=kind,matching_and_cache_s=time.perf_counter()-start,
            groups=len(cache),regions_min=min(p['regions'] for p in cache.values()),
            regions_median=float(np.median([p['regions'] for p in cache.values()])),
            empty_prior_groups=sum(p['regions']==0 for a,b,p in ps)))
        variants=['corrected'] if kind=='v29_finite' else ['raw','corrected','response']+(['response_floor'] if kind=='direct' else [])
        for variant in variants:
            stage=kind+('_'+variant if variant!='raw' and kind!='v29_finite' else '')
            try:
                if kind=='v29_finite':
                    prior=old.fit_relative(ps,hs,cal,c['condition'],'multi');corr=old.empirical_correction(ps,prior,cal)
                    preds=[old.infer_relative([(a,p) for (a,b,ch),p in cache.items() if b==j+5],prior,cal,corr) for j in range(len(c['tests']))]
                    prior['empirical_correction']=corr
                elif variant.startswith('response'):
                    prior=new.fit_response(ps,hs,cal,c['condition'],.05 if variant=='response_floor' else .005)
                    preds=[new.infer_response([(a,p) for (a,b,ch),p in cache.items() if b==j+5],prior,cal) for j in range(len(c['tests']))]
                else:
                    prior=new.fit(ps,hs,cal,c['condition'],variant=='corrected')
                    preds=[new.infer([(a,p) for (a,b,ch),p in cache.items() if b==j+5],prior,cal) for j in range(len(c['tests']))]
            except ValueError as e:
                prior=dict(error=str(e));preds=[dict(height_m=None,reasons=['prior_failed',str(e)],qualified=False) for _ in c['tests']]
            rs=rows(c,preds,stage,'empty_four')
            for r in rs:r.update(container_max_height_m=c['container_max_height_m'],container_width_m=c['container_width_m'],empty_width_px=c['empty_width_px'])
            records+=rs;fits.append(dict(stage=stage,prior=prior))
    return dict(config_id=c['id'],records=records,fits=fits,observations=obsstats)

def evaluate(split):
    cs=json.loads((DATA/split/'manifest.json').read_text())['configs'];records=[];fits=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(evaluate_config,c) for c in cs]
        for i,f in enumerate(as_completed(fs)):
            x=f.result();records+=x['records'];fits.append(x);progress('evaluate_'+split,configs=i+1,total=len(cs));print('EVALUATE',split,i+1,len(cs),flush=True)
    dump(OUT/(split+'-raw.json'),dict(records=records,fits=fits))
    if split=='development':select(records)

def score(rs):
    a=summary(rs);small=summary([r for r in rs if r['empty_width_px']<80])
    return .5*a['normalized_mean']+.5*small['normalized_mean']+.2*a['normalized_p90']+10*(1-a['outputs']/max(len(rs),1))

def select(records):
    choices={};selected={}
    for condition in ('transparent','translucent_mild'):
        candidates=[]
        for stage in METHODS:
            rs=[r for r in records if r['stage']==stage and r['condition']==condition]
            candidates.append(dict(stage=stage,score=score(rs),overall=summary(rs),small=summary([r for r in rs if r['empty_width_px']<80])))
        choices[condition]=candidates;selected[condition]=min(candidates,key=lambda x:x['score'])['stage']
    dump(OUT/'development-candidates.json',choices);dump(OUT/'frozen-policy.json',dict(selection=selected,prior_budget=5,levels=LEVELS,
        test_used=False,confidence_gate=False,coverage_gate=False,capacity_clipping=False,production_changed=False))
    print('POLICY',json.dumps(selected),flush=True)

def run_test():
    if (OUT/'frozen-source.json').exists():raise FileExistsError('frozen_test_already_started')
    start=time.time();paths=[REPO/p for p in ('src/liquid_depth/finite_refraction_images_v30.py','src/liquid_depth/finite_relative_v30.py',
        'src/liquid_depth/transparent_relative_v29.py','src/liquid_depth/refraction_images_v23.py','src/liquid_depth/refraction_matching_v23.py',
        'src/liquid_depth/transparent_diagnostic_v25.py','src/liquid_depth/transparent_refinement_v26.py',
        'scripts/experiment_finite_v30.py','scripts/experiment_transparent_v27.py','tests/test_finite_relative_v30.py')]+[OUT/'frozen-policy.json']
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};dump(OUT/'frozen-source.json',frozen)
    with (OUT/'regression.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','pytest','-q'],env=dict(os.environ,PYTHONPATH='src:scripts'),stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('regression_failed')
    generate('test');evaluate('test')
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items())
    progress('complete',status='complete',source_and_policy_unchanged=unchanged,elapsed_s=time.time()-start)
    if not unchanged:raise RuntimeError('frozen_source_changed')

if __name__=='__main__':
    cv2.setNumThreads(1);ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['generate_dev','evaluate_dev','run_test']);args=ap.parse_args()
    if args.stage=='generate_dev':generate('development')
    elif args.stage=='evaluate_dev':evaluate('development')
    else:run_test()
