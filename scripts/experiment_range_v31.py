"""Uniform actual-liquid-height assessment and frozen range optimization."""
import argparse,os,json,time,hashlib,sys,subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import cv2,numpy as np
from liquid_depth.range_images_v31 import RangeScene
from liquid_depth import finite_relative_v30 as base

REPO=Path(__file__).resolve().parents[1];ROOT=Path('/root/autodl-tmp')
OUT=ROOT/'liquid-depth-artifacts/evaluation/transparent-range-v31-20261011'
DATA=ROOT/'liquid-depth-data/synthetic/transparent-range-v31-20261011'
WORKERS=8;LEVELS=[0.,.15,.40,.65,.90]
METHODS=['baseline_v30','wide_response','wide_curve','wide_curve_best']


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def progress(stage,**kw):dump(OUT/'progress.json',dict(stage=stage,**kw))


def specs(split):
    if split=='development':
        geometry=[(9.5,9.15,136),(9.9,9.25,152)];ns=[1.33,1.41];grains=[.045,.075]
        ext=[0.,.025,.09,.65];seed0=1203109;levels=np.linspace(0,9,37)
    elif split=='test':
        geometry=[(9.7,9.3,128),(10.,9.4,160)];ns=[1.305,1.46];grains=[.04,.055,.085,.10]
        ext=[0.,.035,.11,.65];seed0=1307933;levels=np.linspace(0,9,73)
    else:raise ValueError('unknown_split')
    jobs=[]
    for i,(geo,n,g) in enumerate((geo,n,g) for geo in geometry for n in ns for g in grains):
        for k,e in enumerate(ext):
            condition='transparent' if k==0 else 'translucent_mild'
            jobs.append(dict(id=f'range_{i:03d}_{k}',split=split,bottom=geo[0],capacity=geo[1],width_px=geo[2],
                index=n,grain=g,seed=seed0+i*1009,extinction=e,condition=condition,
                material_regime=('transparent','semi_low_extinction','semi_medium_extinction','legacy_depth_attenuation')[k],
                heights=levels.tolist()))
    return jobs


def render_job(job):
    cv2.setNumThreads(1);folder=DATA/job['split']/job['id'];folder.mkdir(parents=True)
    scene=RangeScene(job['seed'],job['bottom'],job['capacity'],job['width_px'],job['index'],job['grain'],job['extinction'],job['condition'])
    priors=[];tests=[]
    for role,hs in [('prior',np.array(LEVELS)*job['capacity']),('test',job['heights'])]:
        for j,h in enumerate(hs):
            dst=folder/f'{role}_{j:03d}';dst.mkdir()
            images,truth=scene.capture(float(h),j if role=='prior' else 1000+j)
            for ch,im in images.items():
                if not cv2.imwrite(str(dst/(ch+'.png')),im):raise IOError('image_write_failed')
            dump(dst/'calibration.json',scene.cal);np.savez_compressed(dst/'evaluator_truth.npz',**truth)
            record=dict(folder=str(dst),evaluator_truth=str(dst/'evaluator_truth.npz'),role=role)
            if role=='prior':
                rng=np.random.default_rng(job['seed']+j*65537)
                record['known_liquid_height_m']=0. if j==0 else float(h+rng.normal(0,.0005+.001*h))
                priors.append(record)
            else:tests.append(record)
    for ch,mask in scene.masks.items():cv2.imwrite(str(folder/('roi_'+ch+'.png')),mask)
    return dict(id=job['id'],split=job['split'],folder=str(folder),condition=job['condition'],material_regime=job['material_regime'],
        bottom_m=job['bottom'],capacity_m=job['capacity'],empty_width_px=job['width_px'],priors=priors,tests=tests,
        grain_m=job['grain'],texture_cluster=job['seed'])


def generate(split):
    if (DATA/split).exists():raise FileExistsError('preserve_existing_evidence')
    (DATA/split).mkdir(parents=True);jobs=specs(split);cs=[]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for f in as_completed([pool.submit(render_job,j) for j in jobs]):
            cs.append(f.result());progress('generate_'+split,configs=len(cs),total=len(jobs));print('GENERATE',split,len(cs),len(jobs),flush=True)
    dump(DATA/split/'manifest.json',dict(configs=sorted(cs,key=lambda c:c['id']),prior_capture_groups=5,
        true_height_range_m=[0,9],camera_to_bottom_max_m=10,confidence_gate=False,coverage_gate=False,capacity_clipping=False,
        zero_test_independent_of_empty_prior=True,legacy_attenuation_is_boundary_audit=True))


def load(c):
    ims={};caps=c['priors']+c['tests']
    for j,cap in enumerate(caps):
        for ch in ('left','right'):
            im=cv2.imread(str(Path(cap['folder'])/(ch+'.png')),0)
            if im is None:raise ValueError('missing_observed_image')
            ims[j,ch]=im
    cal=json.loads((Path(caps[0]['folder'])/'calibration.json').read_text())
    masks={ch:cv2.imread(str(Path(c['folder'])/('roi_'+ch+'.png')),0) for ch in ('left','right')}
    return ims,masks,cal,[cap['known_liquid_height_m'] for cap in c['priors']]


def cache(c,kind,ims,masks,cal):
    path=Path(c['folder'])/('tracks_'+('wide_s3' if kind=='wide' else kind)+'.npz')
    if path.exists():
        out={}
        with np.load(path) as f:
            for k in f.files:
                a,b,ch,key=k.split('_',3);p=out.setdefault((int(a),int(b),ch),dict(channel=ch));p[key]=f[k]
        for p in out.values():p['regions']=len(p['weights'])
        return out
    if kind=='baseline':fn=base.direct_pair
    else:
        from liquid_depth.range_relative_v31 import wide_pair
        fn=wide_pair
    out={}
    for a in range(5):
        for b in range(a+1,len(c['tests'])+5):
            for ch in ('left','right'):out[a,b,ch]=fn(ims[a,ch],ims[b,ch],cal,masks[ch],ch)
    arrays={f'{a}_{b}_{ch}_{key}':p[key] for (a,b,ch),p in out.items() for key in ('anchor_q','current_q','weights')}
    # Keep scalar registration diagnostics as arrays, for graph calibration.
    for (a,b,ch),p in out.items():
        for key in ('correlation','photometric_rms'):
            if key in p:arrays[f'{a}_{b}_{ch}_{key}']=np.asarray(p[key])
    np.savez_compressed(path,**arrays);return out


def score_rows(c,preds,stage):
    result=[]
    if len(preds)!=len(c['tests']):raise RuntimeError('all_frame_count_mismatch')
    for j,(cap,pred) in enumerate(zip(c['tests'],preds)):
        with np.load(cap['evaluator_truth']) as f:h=float(f['height_m']);t=float(f['centre_ballistic_transmission'])
        value=pred.get('height_m');value=float(value) if value is not None and np.isfinite(value) else None
        result.append(dict(config_id=c['id'],sample_id=j,stage=stage,condition=c['condition'],material_regime=c['material_regime'],
            texture_cluster=c['texture_cluster'],bottom_m=c['bottom_m'],capacity_m=c['capacity_m'],empty_width_px=c['empty_width_px'],
            true_height_m=h,predicted_height_m=value,absolute_error_m=abs(value-h) if value is not None else None,
            tolerance_m=max(.005,.02*h),ballistic_transmission_evaluator_only=t,diagnostics=pred))
    return result


def summary(rs):
    finite=[r for r in rs if r['predicted_height_m'] is not None];e=np.array([r['absolute_error_m'] for r in finite])
    h=np.array([r['true_height_m'] for r in finite]);tol=np.maximum(.005,.02*h);positive=h>0
    return dict(frames=len(rs),outputs=len(finite),passed=int(np.sum(e<=tol)),pass_pct=100*int(np.sum(e<=tol))/max(len(rs),1),
        numeric_output_pct=100*len(finite)/max(len(rs),1),mae_mm=float(e.mean()*1000) if len(e) else None,
        p95_mm=float(np.percentile(e,95)*1000) if len(e) else None,max_mm=float(e.max()*1000) if len(e) else None,
        absrel_positive_pct=float(np.mean(e[positive]/h[positive])*100) if positive.any() else None,
        absrel_positive_frames=int(positive.sum()),zero_height_frames=int(np.sum(h==0)),
        normalized_mean=float(np.mean(e/tol)) if len(e) else None,normalized_p95=float(np.percentile(e/tol,95)) if len(e) else None)


def evaluate_job(c,methods):
    cv2.setNumThreads(1);ims,masks,cal,hs=load(c);records=[];fits=[]
    for kind in sorted(set('baseline' if m=='baseline_v30' else 'wide' for m in methods)):
        obs=cache(c,kind,ims,masks,cal);ps=[(a,b,p) for (a,b,ch),p in obs.items() if b<5]
        for stage in [m for m in methods if ('baseline' if m=='baseline_v30' else 'wide')==kind]:
            try:
                if stage in ('baseline_v30','wide_response'):
                    prior=base.fit_response(ps,hs,cal,c['condition'],.005 if c['condition']=='transparent' else .05)
                    infer=base.infer_response
                else:
                    from liquid_depth.range_relative_v31 import fit_curve,infer_curve
                    prior=fit_curve(ps,hs,cal,c['condition']);infer=lambda gs,pr,ca:infer_curve(gs,pr,ca,best=stage.endswith('_best'))
                preds=[infer([(a,p) for (a,b,ch),p in obs.items() if b==j+5],prior,cal) for j in range(len(c['tests']))]
            except ValueError as exc:
                prior=dict(error=str(exc));preds=[dict(height_m=None,reasons=['prior_failed',str(exc)],qualified=False) for _ in c['tests']]
            records+=score_rows(c,preds,stage);fits.append(dict(stage=stage,prior=prior))
    result=dict(config_id=c['id'],records=records,fits=fits);dump(OUT/(c['split']+'-configs')/(c['id']+'.json'),result)
    return result


def evaluate(split,methods):
    cs=json.loads((DATA/split/'manifest.json').read_text())['configs'];records=[];fits=[]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for i,f in enumerate(as_completed([pool.submit(evaluate_job,c,methods) for c in cs])):
            x=f.result();records+=x['records'];fits.append(x);progress('evaluate_'+split,configs=i+1,total=len(cs));print('EVALUATE',split,i+1,len(cs),flush=True)
    dump(OUT/(split+('-initial' if methods==['baseline_v30'] else '')+'-raw.json'),dict(records=records,fits=fits))
    return records


def intervals(rs):
    out={'zero':summary([r for r in rs if r['true_height_m']==0])}
    for k in range(9):out[str(k)+'_'+str(k+1)]=summary([r for r in rs if (k<r['true_height_m']<k+1 if k==0 else k<=r['true_height_m']<k+1) or (k==8 and r['true_height_m']==9)])
    return out


def select(rs):
    selected={};candidates={}
    # The old high-extinction liquid becomes bottom-invisible at large H.
    # It is reported, but not a target for new strong-scatter optimization.
    for condition in ('transparent','translucent_mild'):
        opts=[]
        for stage in METHODS:
            rows=[r for r in rs if r['stage']==stage and r['condition']==condition and r['material_regime']!='legacy_depth_attenuation']
            s=summary(rows);bins=intervals(rows);finite_bins=[v for v in bins.values() if v['frames']]
            # Unclipped error and tail costs, plus equal emphasis on range bins.
            cost=float(np.mean([v['normalized_mean'] if v['normalized_mean'] is not None else 100 for v in finite_bins]))
            cost+=.25*(s['normalized_p95'] if s['normalized_p95'] is not None else 100)+20*(1-s['outputs']/max(s['frames'],1))
            opts.append(dict(stage=stage,score=cost,overall=s,intervals=bins))
        selected[condition]=min(opts,key=lambda x:x['score'])['stage'];candidates[condition]=opts
    dump(OUT/'development-candidates.json',candidates)
    dump(OUT/'frozen-policy.json',dict(selection=selected,prior_capture_groups=5,levels=LEVELS,actual_height_range_m=[0,9],
        camera_to_bottom_max_m=10,confidence_gate=False,coverage_gate=False,capacity_clipping=False,test_used=False,production_changed=False))
    print('POLICY',selected,flush=True)


def run_test():
    if (OUT/'frozen-source.json').exists():raise FileExistsError('preserve_frozen_evidence')
    files=[REPO/p for p in ('src/liquid_depth/range_images_v31.py','src/liquid_depth/range_relative_v31.py',
        'src/liquid_depth/finite_relative_v30.py','src/liquid_depth/transparent_relative_v29.py','src/liquid_depth/refraction_images_v23.py',
        'scripts/experiment_range_v31.py','tests/test_range_relative_v31.py')]+[OUT/'frozen-policy.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};dump(OUT/'frozen-source.json',hashes)
    with (OUT/'regression.log').open('w') as log:
        ret=subprocess.run([sys.executable,'-m','pytest','-q'],env=dict(os.environ,PYTHONPATH='src:scripts'),stdout=log,stderr=subprocess.STDOUT)
    if ret.returncode:raise RuntimeError('regression_failed')
    start=time.time();generate('test');evaluate('test',METHODS)
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    progress('complete',source_and_policy_unchanged=unchanged,elapsed_s=time.time()-start)
    if not unchanged:raise RuntimeError('frozen_source_changed')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['generate_dev','baseline_dev','evaluate_dev','run_test']);args=ap.parse_args()
    if args.stage=='generate_dev':generate('development')
    elif args.stage=='baseline_dev':
        rs=evaluate('development',['baseline_v30']);dump(OUT/'initial-range-analysis.json',
            {m:intervals([r for r in rs if r['material_regime']==m]) for m in sorted({r['material_regime'] for r in rs})})
    elif args.stage=='evaluate_dev':select(evaluate('development',METHODS))
    else:run_test()
