"""Development-only selection followed by a new frozen all-frame simulation."""
import argparse,json,os,time,hashlib,subprocess,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import cv2
import experiment_range_v31 as v31
from liquid_depth.range_images_v31 import RangeScene
from liquid_depth import range_relative_v31 as baseline
from liquid_depth import range_relative_v32 as model

REPO=Path(__file__).resolve().parents[1]
OUT=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/transparent-focus-v32-20261012')
DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/transparent-focus-v32-20261012')
WORKERS=20;LEVELS=[0.,.15,.40,.65,.90]
BASE='baseline_v31'


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def progress(stage,**kw):dump(OUT/'progress.json',dict(stage=stage,**kw))


def choices():
    out={BASE:None}
    for cal in ('global','local'):
        for shape in ('linear','pchip'):
            for fusion in ('median','mean','robust'):
                for band in (None,.025,.06):
                    name=f'{cal}_{shape}_{fusion}_{band}'
                    out[name]=dict(calibration=cal,shape=shape,fusion=fusion,band=band)
    return out


def measurement_levels(split):
    uniform=np.linspace(0.,9.,37 if split=='development' else 73)
    step=.05 if split=='development' else .025
    levels={round(float(h),8):'uniform' for h in uniform}
    for h in np.arange(step,1.,step):levels.setdefault(round(float(h),8),'focus_low')
    for h in np.arange(2.,3.+step/2,step):levels.setdefault(round(float(h),8),'focus_mid')
    for h in (.002,.005,.01,.02,.04,.06,.08,.10):levels.setdefault(h,'near_zero')
    out=[dict(height=h,population=p) for h,p in sorted(levels.items())]
    out += [dict(height=0.,population='zero_repeat') for _ in range((12 if split=='development' else 24)-1)]
    return out


def specs(split):
    if split=='development':
        geometries=[(9.55,9.12,136),(9.85,9.22,148)];indices=[1.32,1.405];grains=[.048,.072]
        ext=[0.,.022,.085];seed0=1410121
    elif split=='test':
        geometries=[(9.65,9.28,128),(10.,9.45,160)];indices=[1.315,1.455];grains=[.042,.058,.082,.095]
        ext=[0.,.032,.105];seed0=1513117
    else:raise ValueError('unknown_split')
    jobs=[]
    for i,(g,n,t) in enumerate((g,n,t) for g in geometries for n in indices for t in grains):
        for k,e in enumerate(ext):
            jobs.append(dict(id=f'focus_{i:03d}_{k}',split=split,bottom=g[0],capacity=g[1],width_px=g[2],
                index=n,grain=t,extinction=e,seed=seed0+i*1009,condition='transparent' if k==0 else 'translucent_mild',
                material_regime=('transparent','semi_low_extinction','semi_medium_extinction')[k],levels=measurement_levels(split)))
    return jobs


def render(job):
    cv2.setNumThreads(1);folder=DATA/job['split']/job['id'];folder.mkdir(parents=True)
    scene=RangeScene(job['seed'],job['bottom'],job['capacity'],job['width_px'],job['index'],job['grain'],job['extinction'],job['condition'])
    priors=[];tests=[]
    for role,items in [('prior',[dict(height=x*job['capacity'],population='prior') for x in LEVELS]),('test',job['levels'])]:
        for j,item in enumerate(items):
            h=item['height'];dst=folder/f'{role}_{j:03d}';dst.mkdir()
            images,truth=scene.capture(float(h),j if role=='prior' else 2000+j)
            for ch,im in images.items():
                if not cv2.imwrite(str(dst/(ch+'.png')),im):raise IOError('image_write_failed')
            dump(dst/'calibration.json',scene.cal);np.savez_compressed(dst/'evaluator_truth.npz',**truth)
            record=dict(folder=str(dst),evaluator_truth=str(dst/'evaluator_truth.npz'),role=role,population=item['population'])
            if role=='prior':
                rng=np.random.default_rng(job['seed']+j*65537)
                record['known_liquid_height_m']=0. if j==0 else float(h+rng.normal(0.,.0005+.001*h))
                priors.append(record)
            else:tests.append(record)
    for ch,mask in scene.masks.items():cv2.imwrite(str(folder/('roi_'+ch+'.png')),mask)
    return dict(id=job['id'],split=job['split'],folder=str(folder),condition=job['condition'],material_regime=job['material_regime'],
        bottom_m=job['bottom'],capacity_m=job['capacity'],empty_width_px=job['width_px'],priors=priors,tests=tests,
        grain_m=job['grain'],texture_cluster=job['seed'])


def generate(split):
    if (DATA/split).exists():raise FileExistsError('preserve_existing_dataset')
    jobs=specs(split);configs=[];(DATA/split).mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for f in as_completed([pool.submit(render,j) for j in jobs]):
            configs.append(f.result());progress('generate_'+split,configs=len(configs),total=len(jobs));print('GENERATE',split,len(configs),len(jobs),flush=True)
    dump(DATA/split/'manifest.json',dict(configs=sorted(configs,key=lambda c:c['id']),prior_groups=5,
        liquid_height_range_m=[0,9],camera_to_bottom_max_m=10,confidence_gate=False,coverage_gate=False,
        zero_snapping=False,capacity_clipping=False,test_zero_independent_capture=True,
        strong_scattering_excluded_by_predeclared_scope=True))


def rows_for(c,preds,stage):
    if len(preds)!=len(c['tests']):raise RuntimeError('missing_test_frame')
    rows=[]
    for j,(cap,pred) in enumerate(zip(c['tests'],preds)):
        with np.load(cap['evaluator_truth']) as f:h=float(f['height_m'])
        value=pred.get('height_m');value=float(value) if value is not None and np.isfinite(value) else None
        rows.append(dict(config_id=c['id'],sample_id=j,stage=stage,condition=c['condition'],material_regime=c['material_regime'],
            texture_cluster=c['texture_cluster'],bottom_m=c['bottom_m'],capacity_m=c['capacity_m'],empty_width_px=c['empty_width_px'],
            population=cap.get('population','uniform'),true_height_m=h,predicted_height_m=value,
            absolute_error_m=abs(value-h) if value is not None else None,tolerance_m=max(.005,.02*h),diagnostics=pred))
    return rows


def evaluate_job(c,names,checkpoint):
    cv2.setNumThreads(1);ims,masks,cal,hs=v31.load(c);obs=v31.cache(c,'wide',ims,masks,cal)
    ps=[(a,b,p) for (a,b,ch),p in obs.items() if b<5]
    groups=[[(a,p) for (a,b,ch),p in obs.items() if b==j+5] for j in range(len(c['tests']))]
    options=choices();records=[];fits=[]
    for name in names:
        try:
            if name==BASE:
                prior=baseline.fit_curve(ps,hs,cal,c['condition']);preds=[baseline.infer_curve(g,prior,cal) for g in groups]
            else:
                opt=options[name];prior=model.fit(ps,hs,cal,c['condition'],opt)
                preds=[model.infer(g,prior,cal,opt['fusion'],opt['band']) for g in groups]
        except ValueError as exc:
            prior=dict(error=str(exc));preds=[dict(height_m=None,reasons=['prior_failed',str(exc)],qualified=False) for _ in groups]
        records+=rows_for(c,preds,name);fits.append(dict(stage=name,prior=prior))
    result=dict(config_id=c['id'],records=records,fits=fits)
    dump(OUT/(checkpoint+'-configs')/(c['id']+'.json'),result)
    return result


def evaluate(configs,names,checkpoint):
    rows=[];fits=[]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for i,f in enumerate(as_completed([pool.submit(evaluate_job,c,names,checkpoint) for c in configs])):
            result=f.result();rows+=result['records'];fits.append(dict(config_id=result['config_id'],fits=result['fits']))
            progress('evaluate_'+checkpoint,configs=i+1,total=len(configs));print('EVALUATE',checkpoint,i+1,len(configs),flush=True)
    dump(OUT/(checkpoint+'-raw.json'),dict(records=rows,fits=fits))
    return rows


def buckets(rows):
    return dict(zero=v31.summary([r for r in rows if r['true_height_m']==0]),
        low=v31.summary([r for r in rows if 0<r['true_height_m']<1]),
        mid=v31.summary([r for r in rows if 2<=r['true_height_m']<3]),
        rest=v31.summary([r for r in rows if r['true_height_m']>=1 and not 2<=r['true_height_m']<3]))


def rank(rows,names,condition):
    base=[r for r in rows if r['stage']==BASE and r['condition']==condition];bb=buckets(base);ranked=[]
    for name in names:
        rr=[r for r in rows if r['stage']==name and r['condition']==condition];bs=buckets(rr);ss=v31.summary(rr)
        cost=sum(bs[k]['normalized_mean'] if bs[k]['normalized_mean'] is not None else 100 for k in ('zero','low','mid'))
        cost+=.2*(bs['rest']['normalized_mean'] or 100)+.05*(ss['normalized_p95'] or 100)+50*(1-ss['outputs']/max(1,ss['frames']))
        eligible=all(bs[k]['mae_mm'] is not None and bs[k]['mae_mm']<=bb[k]['mae_mm']*1.05 for k in ('zero','low','mid'))
        eligible &= bs['rest']['mae_mm'] is not None and bs['rest']['mae_mm']<=bb['rest']['mae_mm']*1.2
        ranked.append(dict(stage=name,cost=cost,eligible=bool(eligible),overall=ss,buckets=bs))
    return sorted(ranked,key=lambda r:r['cost'])


def screen():
    old_manifest=json.loads((v31.DATA/'development/manifest.json').read_text())['configs']
    cs=[c for c in old_manifest if c['material_regime']!='legacy_depth_attenuation'];names=list(choices())
    rows=evaluate(cs,names,'screen');rankings={c:rank(rows,names,c) for c in ('transparent','translucent_mild')}
    shortlist={BASE}
    for rr in rankings.values():shortlist.update(r['stage'] for r in rr[:6])
    dump(OUT/'screen-ranking.json',rankings);dump(OUT/'shortlist.json',sorted(shortlist))
    print('SHORTLIST',sorted(shortlist),flush=True)


def development():
    names=json.loads((OUT/'shortlist.json').read_text());cs=json.loads((DATA/'development/manifest.json').read_text())['configs']
    rows=evaluate(cs,names,'development');rankings={c:rank(rows,names,c) for c in ('transparent','translucent_mild')}
    selected={c:next((r['stage'] for r in rr if r['eligible']),BASE) for c,rr in rankings.items()}
    dump(OUT/'development-ranking.json',rankings)
    dump(OUT/'frozen-policy.json',dict(selection=selected,options={c:choices()[n] for c,n in selected.items()},
        prior_groups=5,levels=LEVELS,confidence_gate=False,coverage_gate=False,zero_snapping=False,capacity_clipping=False,
        range_m=[0,9],camera_to_bottom_max_m=10,production_changed=False,test_used_for_selection=False))
    print('POLICY',selected,flush=True)


def run_test():
    if (OUT/'frozen-source.json').exists():raise FileExistsError('preserve_frozen_evidence')
    files=list((REPO/'src/liquid_depth').glob('*.py'))+[REPO/p for p in ('scripts/experiment_range_v31.py',
        'scripts/experiment_focus_v32.py','tests/test_range_relative_v32.py')]+[OUT/'frozen-policy.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};dump(OUT/'frozen-source.json',hashes)
    with (OUT/'regression.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','pytest','-q'],env=dict(os.environ,PYTHONPATH='src:scripts'),stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('regression_failed')
    start=time.time();generate('test');policy=json.loads((OUT/'frozen-policy.json').read_text())
    names=sorted({BASE,*policy['selection'].values()});cs=json.loads((DATA/'test/manifest.json').read_text())['configs']
    evaluate(cs,names,'test')
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    progress('complete',source_and_policy_unchanged=unchanged,elapsed_s=time.time()-start)
    if not unchanged:raise RuntimeError('frozen_source_changed')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','generate_dev','development','run_test']);args=ap.parse_args()
    if args.stage=='screen':screen()
    elif args.stage=='generate_dev':generate('development')
    elif args.stage=='development':development()
    else:run_test()
