"""Empty + four priors versus five wet priors, with independent all-frame tests."""
import argparse,json,os,time,hashlib,subprocess,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import cv2,numpy as np
from liquid_depth.refraction_images_v23 import render_capture
from liquid_depth.refraction_matching_v23 import stereo_matches
from liquid_depth import transparent_relative_v29 as rel
from liquid_depth import transparent_refinement_v26 as v26
from experiment_transparent_v27 import load,predict26,rows,summary,dump,POLICY26

ROOT=Path('/root/autodl-tmp');REPO=Path(__file__).resolve().parents[1]
OUT=ROOT/'liquid-depth-artifacts/evaluation/transparent-relative-v29-20261011'
DATA=ROOT/'liquid-depth-data/synthetic/transparent-relative-v29-20261011'
LAYOUTS={'wet_uniform':[.10,.30,.50,.70,.90],
    'empty_four':[0.,.15,.40,.65,.90], 'empty_wide':[0.,.10,.35,.65,.95]}
METHODS=['absolute_empty','relative_empty','relative_multi','relative_corrected']

def progress(stage,**kw):dump(OUT/'progress.json',dict(stage=stage,**kw))

def specifications(split):
    if split=='development':
        bs=[3.2,8.4];cs=[.18,.45];ns=[1.33,1.405];seed=812371
        unknown=[.04,.09,.17,.26,.38,.52,.64,.76,.86,.97]
    elif split=='test':
        bs=[2.4,6.7,9.7];cs=[.12,.30,.60];ns=[1.315,1.455];seed=923147
        unknown=[.035,.075,.12,.18,.26,.36,.47,.58,.69,.80,.915,.975]
    else:raise ValueError('unknown_split')
    jobs=[]
    for i,(b,c,n,repeat) in enumerate((b,c,n,r) for b in bs for c in cs for n in ns for r in range(2)):
        for condition in ('transparent','translucent_mild'):
            jobs.append(dict(split=split,id=f'stereo_{i:02d}_{condition}',bottom=b,capacity=c,index=n,
                seed=seed+317*i,condition=condition,unknown=unknown,repeat=repeat))
    return jobs

def read_image(folder,channel):
    im=cv2.imread(str(Path(folder)/(channel+'.png')),cv2.IMREAD_GRAYSCALE)
    if im is None:raise ValueError('missing_observation_image')
    return im

def render_config(job):
    cv2.setNumThreads(1);priors={};tests=[]
    for role,ratios in [('prior',sorted({x for v in LAYOUTS.values() for x in v})),('test',job['unknown'])]:
        for j,ratio in enumerate(ratios):
            folder=DATA/job['split']/job['id']/f'{role}_{j:02d}';folder.mkdir(parents=True)
            # Empty vessel has air rather than the liquid's scattering medium.
            condition='transparent' if ratio==0 else job['condition']
            images,cal,truth=render_capture(job['seed'],ratio*job['capacity'],job['index'],job['bottom'],.055,condition,'active_stereo')
            cal['container_max_height_m']=job['capacity']
            for name,im in images.items():
                if not cv2.imwrite(str(folder/(name+'.png')),im):raise IOError('image_write_failed')
            dump(folder/'calibration.json',cal);np.savez_compressed(folder/'evaluator_truth.npz',**truth)
            o=stereo_matches(images['left'],images['right'],cal,True)
            np.savez_compressed(folder/'observation.npz',**{k:o[k] for k in ('left_q','right_q','left_pixels','right_pixels','weights')})
            item=dict(folder=str(folder),role=role,evaluator_truth=str(folder/'evaluator_truth.npz'))
            if role=='prior':
                rng=np.random.default_rng(job['seed']+int(round(ratio*1e6))*65537)
                item['known_liquid_height_m']=0. if ratio==0 else float(ratio*job['capacity']+rng.normal(0,.0005+.001*ratio*job['capacity']))
                priors[ratio]=item
            else:tests.append(item)
    return dict(id=job['id'],family='active_stereo',condition=job['condition'],camera_to_bottom_m=job['bottom'],
        container_max_height_m=job['capacity'],priors={k:[priors[x] for x in v] for k,v in LAYOUTS.items()},tests=tests)

def generate(split):
    folder=DATA/split
    if folder.exists():raise FileExistsError('preserve_existing_evidence')
    folder.mkdir(parents=True);jobs=specifications(split);configs=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(render_config,j) for j in jobs]
        for f in as_completed(fs):
            configs.append(f.result());progress('generate_'+split,configs=len(configs),total=len(jobs));print('GENERATE',split,len(configs),len(jobs),flush=True)
    manifest=dict(configs=sorted(configs,key=lambda c:c['id']),split=split,layouts=LAYOUTS,prior_budget=5,
        fixed_camera=True,finite_vessel_walls_rendered=False,empty_label_exact_zero=True,source='image_proxy_v23')
    dump(folder/'manifest.json',manifest)

def flow_cache(c,layout,include_tests):
    caps=c['priors'][layout]+(c['tests'] if include_tests else [])
    cal=json.loads((Path(caps[0]['folder'])/'calibration.json').read_text());cache={}
    images={(i,ch):read_image(p['folder'],ch) for i,p in enumerate(caps) for ch in ('left','right')}
    for a in range(5):
        for b in range(a+1,len(caps)):
            if b<5 or include_tests:
                for ch in ('left','right'):
                    cache[a,b,ch]=rel.track_pair(images[a,ch],images[b,ch],cal,ch)
    return cache

def cache_path(c,layout):return DATA/('development' if '/development/' in c['tests'][0]['folder'] else 'test')/c['id']/('tracks_'+layout+'.npz')

def save_cache(path,cache):
    data={f'{a}_{b}_{ch}_{key}':p[key] for (a,b,ch),p in cache.items() for key in ('anchor_q','current_q','weights')}
    np.savez_compressed(path,**data)

def read_cache(path):
    cache={}
    with np.load(path) as f:
        for key in f.files:
            a,b,ch,field=key.split('_',3);p=cache.setdefault((int(a),int(b),ch),dict(channel=ch));p[field]=f[key]
    for p in cache.values():p['regions']=len(p['weights'])
    return cache

def ensure_cache(c,layout):
    path=cache_path(c,layout)
    if not path.exists():save_cache(path,flow_cache(c,layout,True))
    return read_cache(path)

def evaluate_config(c):
    cv2.setNumThreads(1);records=[];fits=[]
    baseline,prior=predict26(c,'wet_uniform');records+=rows(c,baseline,'wet_uniform','wet_uniform');fits.append(dict(method='wet_uniform',layout='wet_uniform',prior=prior))
    for layout in ('empty_four','empty_wide'):
        obs,cal,hs=load(c,layout);setting=POLICY26['active_stereo/'+c['condition']]
        try:
            prior=rel.fit_absolute_empty(obs[:5],hs,cal,c['condition'],setting['harmonics'],setting['ridge'])
            preds=[v26.infer(o,prior,cal,setting['correction']) for o in obs[5:]]
        except ValueError as e:
            prior=dict(error=str(e));preds=[dict(height_m=None,reasons=['prior_failed',str(e)],qualified=False) for _ in c['tests']]
        records+=rows(c,preds,layout+'/absolute_empty',layout);fits.append(dict(method='absolute_empty',layout=layout,prior=prior))
        cache=ensure_cache(c,layout)
        pairs=[(a,b,p) for (a,b,ch),p in cache.items() if b<5]
        for method in METHODS[1:]:
            try:
                mode='empty' if method=='relative_empty' else 'multi'
                prior=rel.fit_relative(pairs,hs,cal,c['condition'],mode)
                corr=rel.empirical_correction(pairs,prior,cal) if method=='relative_corrected' else None
                preds=[rel.infer_relative([(a,p) for (a,b,ch),p in cache.items() if b==j+5],prior,cal,corr) for j in range(len(c['tests']))]
                if corr is not None:prior=dict(prior,empirical_correction=corr)
            except ValueError as e:
                prior=dict(error=str(e));preds=[dict(height_m=None,reasons=['prior_failed',str(e)],qualified=False) for _ in c['tests']]
            records+=rows(c,preds,layout+'/'+method,layout);fits.append(dict(method=method,layout=layout,prior=prior))
    for r in records:
        r['container_max_height_m']=c['container_max_height_m']
        p=r['predicted_height_m']
        if p is not None and not 0<=p<=c['container_max_height_m']:
            reasons=r['diagnostics'].setdefault('reasons',[])
            if 'outside_container_capacity' not in reasons:reasons.append('outside_container_capacity')
    return dict(config_id=c['id'],records=records,fits=fits)

def evaluate(split):
    configs=json.loads((DATA/split/'manifest.json').read_text())['configs'];records=[];fits=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(evaluate_config,c) for c in configs]
        for i,f in enumerate(as_completed(fs)):
            x=f.result();records+=x['records'];fits.append(x);progress('evaluate_'+split,configs=i+1,total=len(configs));print('EVALUATE',split,i+1,len(configs),flush=True)
    dump(OUT/(split+'-raw.json'),dict(records=records,fits=fits))
    if split=='development':select(records)
    return records

def score(rs):
    a=summary(rs);b=summary([r for r in rs if r['bottom_m']>=6])
    return .5*a['normalized_mean']+.5*b['normalized_mean']+.2*a['normalized_p90']+10*(1-a['outputs']/max(len(rs),1))

def select(records):
    selected={};choices={}
    for condition in ('transparent','translucent_mild'):
        candidates=[]
        for stage in sorted({r['stage'] for r in records}):
            rs=[r for r in records if r['stage']==stage and r['condition']==condition]
            candidates.append(dict(stage=stage,score=score(rs),metrics=summary(rs)))
        best=min(candidates,key=lambda x:x['score']);selected[condition]=best['stage'];choices[condition]=candidates
    dump(OUT/'development-candidates.json',choices)
    dump(OUT/'frozen-policy.json',dict(selection=selected,test_used=False,prior_budget=5,
        confidence_gate=False,coverage_gate=False,capacity_clipping=False,production_changed=False))
    print('POLICY',json.dumps(selected),flush=True)

def run_test():
    if (OUT/'frozen-source.json').exists():raise FileExistsError('frozen_test_already_started')
    start=time.time();files=[REPO/p for p in ('src/liquid_depth/transparent_relative_v29.py','scripts/experiment_relative_v29.py',
        'scripts/audit_relative_v29.py',
        'tests/test_transparent_relative_v29.py','src/liquid_depth/refraction_images_v23.py','src/liquid_depth/refraction_matching_v23.py',
        'src/liquid_depth/transparent_refinement_v26.py','src/liquid_depth/transparent_diagnostic_v25.py','scripts/experiment_transparent_v27.py')]
    files+=[OUT/'frozen-policy.json',ROOT/'liquid-depth-artifacts/evaluation/transparent-refinement-v26-20261011/frozen-policy.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};dump(OUT/'frozen-source.json',hashes)
    progress('regression',status='running')
    with (OUT/'regression.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','pytest','-q'],env=dict(os.environ,PYTHONPATH='src:scripts'),stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('regression_failed')
    generate('test');evaluate('test')
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    progress('complete',status='complete',source_and_policy_unchanged=unchanged,elapsed_s=time.time()-start)
    if not unchanged:raise RuntimeError('frozen_source_changed')

if __name__=='__main__':
    cv2.setNumThreads(1);ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['generate_dev','evaluate_dev','run_test']);args=ap.parse_args()
    if args.stage=='generate_dev':generate('development')
    elif args.stage=='evaluate_dev':evaluate('development')
    else:run_test()
