"""Five-capture SR/registration development, freeze, and independent tests."""
import argparse,hashlib,json,os,sys,time,subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import cv2,numpy as np
import experiment_focus_v32 as previous
from experiment_range_v31 import load,summary
from liquid_depth import range_registration_v33 as reg
from liquid_depth import range_relative_v32 as model

REPO=Path(__file__).resolve().parents[1]
OUT=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/transparent-sr-v33-20261012')
DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/transparent-sr-v33-20261012')
WORKERS=24;BASE='baseline_v32'

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def progress(stage,**kw):dump(OUT/'progress.json',dict(stage=stage,**kw))

def levels(split):
    if split=='test':return previous.measurement_levels('test')
    out={round(float(x),8):'uniform' for x in np.linspace(0,9,19)}
    for x in np.arange(.05,1.,.05):out.setdefault(round(float(x),8),'focus_low')
    for x in (.002,.005,.01,.02,.04,.06,.08,.1):out.setdefault(x,'near_zero')
    return [dict(height=h,population=p) for h,p in sorted(out.items())]+[dict(height=0.,population='zero_repeat') for _ in range(11)]

def specs(split):
    if split=='development':
        geos=[(9.6,9.17,136),(9.9,9.33,152)];ns=[1.325,1.42];grains=[.047,.079];ext=[0.,.027,.095];seed0=1615331
    else:
        geos=[(9.72,9.29,128),(10.,9.46,160)];ns=[1.31,1.448];grains=[.041,.061,.084,.098];ext=[0.,.033,.108];seed0=1719559
    jobs=[]
    for i,(g,n,t) in enumerate((g,n,t) for g in geos for n in ns for t in grains):
        for k,e in enumerate(ext):
            jobs.append(dict(id=f'sr_{i:03d}_{k}',split=split,bottom=g[0],capacity=g[1],width_px=g[2],index=n,
                 grain=t,extinction=e,seed=seed0+i*1009,condition='transparent' if k==0 else 'translucent_mild',
                 material_regime=('transparent','semi_low_extinction','semi_medium_extinction')[k],levels=levels(split)))
    return jobs

def render(job):
    previous.DATA=DATA
    return previous.render(job)

def generate(split):
    if (DATA/split).exists():raise FileExistsError('preserve_existing_dataset')
    (DATA/split).mkdir(parents=True)
    jobs=specs(split);configs=[]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for f in as_completed([pool.submit(render,j) for j in jobs]):
            configs.append(f.result());progress('generate_'+split,configs=len(configs),total=len(jobs))
    dump(DATA/split/'manifest.json',dict(configs=sorted(configs,key=lambda c:c['id']),prior_groups=5,
         source='unchanged RangeScene v31',capture_noise_independent=True,no_test_label_in_inference=True))

def configs(split):return json.loads((DATA/split/'manifest.json').read_text())['configs']

def prepare_sr(split,cs=None):
    cv2.setNumThreads(1)
    from liquid_depth.range_registration_v33 import SwinIRX2
    sr=SwinIRX2()
    for i,c in enumerate(cs or configs(split)):
        path=OUT/'sr-images'/split/(c['id']+'.npz')
        if path.exists():continue
        ims,_,_,_=load(c);keys=list(ims);arrays={};times=[]
        for start in range(0,len(keys),4):
            ks=keys[start:start+4];t=time.perf_counter();ys=sr([ims[k] for k in ks]);times.append((time.perf_counter()-t)/len(ks))
            arrays.update({f'{k[0]}_{k[1]}':y for k,y in zip(ks,ys)})
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
        dump(path.with_suffix('.json'),dict(images=len(keys),mean_image_s=float(np.mean(times)),
              batch=4,grayscale_repeated_three_channels=True,weights='official lightweight x2 PSNR model',
              domain='synthetic IR-like grayscale, not DIV2K bicubic degradation',fine_tuned=False))
        progress('prepare_sr_'+split,configs=i+1,total=len(cs or configs(split)))
        print('SR',split,i+1,flush=True)

def observation_cache(c,name,ims,masks,cal):
    path=OUT/'tracks'/c['split']/name/(c['id']+'.npz')
    if path.exists():
        obs={}
        with np.load(path) as data:
            for k in data.files:
                a,b,ch,key=k.split('_',3);obs.setdefault((int(a),int(b),ch),dict(channel=ch))[key]=data[k]
        for p in obs.values():p['regions']=len(p['weights'])
        return obs
    if 'swinir' in name:
        with np.load(OUT/'sr-images'/c['split']/(c['id']+'.npz')) as data:
            use={tuple([int(k.split('_')[0]),k.split('_')[1]]):data[k] for k in data.files}
    elif 'bicubic' in name:
        use={k:cv2.resize(im.astype(np.float32),None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC) for k,im in ims.items()}
    else:use=ims
    factor=1 if name==BASE else reg.OPTIONS[name]['factor']
    mm={ch:cv2.resize(m,None,fx=factor,fy=factor,interpolation=cv2.INTER_NEAREST) if factor!=1 else m for ch,m in masks.items()}
    obs={}
    for a in range(5):
        for b in range(a+1,len(c['tests'])+5):
            for ch in ('left','right'):obs[a,b,ch]=reg.pair(use[a,ch],use[b,ch],cal,mm[ch],ch,name)
    arrays={f'{a}_{b}_{ch}_{key}':p[key] for (a,b,ch),p in obs.items() for key in ('anchor_q','current_q','weights')}
    for (a,b,ch),p in obs.items():
        for key in ('correlation','photometric_rms'):
            if key in p:arrays[f'{a}_{b}_{ch}_{key}']=np.asarray(p[key])
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
    return obs

def evaluate_job(c,names,checkpoint):
    cv2.setNumThreads(1);ims,masks,cal,hs=load(c);rows=[];fits=[]
    for name in names:
        t=time.perf_counter();obs=observation_cache(c,name,ims,masks,cal)
        ps=[(a,b,p) for (a,b,ch),p in obs.items() if b<5]
        groups=[[(a,p) for (a,b,ch),p in obs.items() if b==j+5] for j in range(len(c['tests']))]
        try:
            prior=model.fit_local(ps,hs,cal,c['condition'],'pchip')
            preds=[model.infer(g,prior,cal,'robust',None) for g in groups]
        except ValueError as exc:
            prior=dict(error=str(exc));preds=[dict(height_m=None,reasons=['prior_failed',str(exc)],qualified=False) for _ in groups]
        rows+=previous.rows_for(c,preds,name);fits.append(dict(stage=name,prior=prior,elapsed_s=time.perf_counter()-t))
    result=dict(config_id=c['id'],records=rows,fits=fits);dump(OUT/(checkpoint+'-configs')/(c['id']+'.json'),result)
    return result

def evaluate(cs,names,checkpoint):
    rows=[];fits=[]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for i,f in enumerate(as_completed([pool.submit(evaluate_job,c,names,checkpoint) for c in cs])):
            x=f.result();rows+=x['records'];fits.append(dict(config_id=x['config_id'],fits=x['fits']))
            progress('evaluate_'+checkpoint,configs=i+1,total=len(cs));print(checkpoint,i+1,len(cs),flush=True)
    dump(OUT/(checkpoint+'-raw.json'),dict(records=rows,fits=fits))
    return rows

def buckets(rows):
    return dict(zero=[r for r in rows if r['true_height_m']==0],near=[r for r in rows if 0<r['true_height_m']<=.1],
         shallow=[r for r in rows if .1<r['true_height_m']<1],rest=[r for r in rows if r['true_height_m']>=1],
         uniform=[r for r in rows if r['population']=='uniform'],all=rows)

def select(rows):
    # Scope is semitransparent; transparent chain stays unchanged regardless of truth height.
    semi=[r for r in rows if r['condition']=='translucent_mild'];ranked=[]
    baseline={k:summary(v) for k,v in buckets([r for r in semi if r['stage']==BASE]).items()}
    timing=json.loads((OUT/'development-runtime.json').read_text()) if (OUT/'development-runtime.json').exists() else []
    for name in reg.OPTIONS:
        ss={k:summary(v) for k,v in buckets([r for r in semi if r['stage']==name]).items()}
        cost=sum((ss[k]['mae_mm']/baseline[k]['mae_mm']+.25*ss[k]['p95_mm']/baseline[k]['p95_mm']) for k in ('zero','near','shallow'))
        cost+=.2*ss['rest']['mae_mm']/baseline['rest']['mae_mm']+100*(1-ss['all']['outputs']/ss['all']['frames'])
        eligible=all(ss[k]['outputs']==ss[k]['frames'] and ss[k]['mae_mm']<=1.05*baseline[k]['mae_mm'] for k in ('zero','near','shallow'))
        eligible &= ss['rest']['mae_mm']<=1.2*baseline['rest']['mae_mm'] and ss['rest']['p95_mm']<=1.3*baseline['rest']['p95_mm']
        tt=[x for x in timing if x['method']==name]
        latency_eligible=len(tt)==2 and max(x['p95_ms'] for x in tt)<=450.
        ranked.append(dict(name=name,cost=cost,eligible=bool(eligible),latency_eligible=latency_eligible,buckets=ss))
    ranked.sort(key=lambda x:x['cost']);chosen=next(x['name'] for x in ranked if x['eligible'] and x['latency_eligible'])
    dump(OUT/'development-ranking.json',ranked)
    dump(OUT/'frozen-policy.json',dict(selection=dict(transparent=BASE,translucent_mild=chosen),
         prior_groups=5,prior_levels=[0,.15,.4,.65,.9],confidence_gate=False,coverage_gate=False,
         no_zero_snapping=True,no_height_clipping=True,no_true_height_routing=True,
         development_objective='zero/near/shallow MAE and P95, rest regression guard; not pass-rate-only',
         test_used_for_selection=False,production_changed=False,
         compute_latency_budget_ms=450.,end_to_end_target_ms=500.,development_runtime='development-runtime.json',
         independent_sr_audit_methods=['bicubic_x2','swinir_x2','midpoint_bicubic2','midpoint_swinir2'],
         independent_sr_audit_population='all H<1 frames at texture grain indices 0 and 2 for each geometry and refractive index, semitransparent only'))
    print('SELECTED',chosen,[(x['name'],round(x['cost'],3),x['eligible']) for x in ranked],flush=True)

def develop():
    generate('development');prepare_sr('development');rows=evaluate(configs('development'),list(reg.OPTIONS),'development')
    subprocess.run([sys.executable,str(REPO/'scripts/probe_sr_v33.py')],check=True)
    select(rows)
    progress('development_complete')

def expand_development():
    rows=evaluate(configs('development'),list(reg.OPTIONS),'development-expanded')
    subprocess.run([sys.executable,str(REPO/'scripts/probe_sr_v33.py')],check=True)
    select(rows);progress('development_complete')

def run_test():
    if (OUT/'frozen-source.json').exists():raise FileExistsError('preserve_frozen_test')
    paths=list((REPO/'src/liquid_depth').glob('*.py'))+[REPO/p for p in ('scripts/experiment_sr_v33.py',
         'scripts/experiment_focus_v32.py','scripts/experiment_range_v31.py','scripts/probe_sr_v33.py',
         'scripts/setup_sr_v33.py','tests/test_range_registration_v33.py')]+[OUT/p for p in
         ('frozen-policy.json','development-runtime.json','development-ranking.json')]
    weights=Path('/root/autodl-tmp/liquid-depth-artifacts/pretrained/swinir_v33')
    paths += [weights/p for p in ('network_swinir.py','swinir_x2.pth','provenance.json')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};dump(OUT/'frozen-source.json',hashes)
    with (OUT/'regression.log').open('w') as log:
        ret=subprocess.run([sys.executable,'-m','pytest','-q'],env=dict(os.environ,PYTHONPATH='src:scripts'),stdout=log,stderr=subprocess.STDOUT)
    if ret.returncode:raise RuntimeError('regression_failed')
    generate('test');policy=json.loads((OUT/'frozen-policy.json').read_text());cs=configs('test')
    names=sorted({BASE,*policy['selection'].values()})
    if any('swinir' in n for n in names):prepare_sr('test')
    evaluate(cs,names,'test')
    progress('test_complete',source_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()))

def sr_audit():
    policy=json.loads((OUT/'frozen-policy.json').read_text());cs=[]
    for c in configs('test'):
        if c['condition']!='translucent_mild' or int(c['id'].split('_')[1])%4 not in (0,2):continue
        c=dict(c);c['tests']=[cap for cap in c['tests'] if cap['population'] in ('zero_repeat','near_zero','focus_low') or
             (cap['population']=='uniform' and float(np.load(cap['evaluator_truth'])['height_m'])<1)]
        # Predeclared truth-based population only for evaluation, never in inference.
        # Remapped capture indices require their own cache namespace.
        c['id']+='_sr_audit';cs.append(c)
    prepare_sr('test',cs);evaluate(cs,sorted({BASE,*policy['independent_sr_audit_methods'],*policy['selection'].values()}),'sr-audit')
    progress('sr_audit_complete')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['develop','expand_dev','test','sr_audit']);args=ap.parse_args()
    if args.stage=='develop':develop()
    elif args.stage=='expand_dev':expand_development()
    elif args.stage=='test':run_test()
    else:sr_audit()
