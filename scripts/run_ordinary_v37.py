"""Frozen sequence generation, covariance audit, causal endpoint evaluation."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
BLENDER='/root/autodl-tmp/tools/blender/blender-4.2.0-linux-x64/blender'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--replicates',type=int,default=9);p.add_argument('--frames',type=int,default=12);p.add_argument('--stage',default='test');a=p.parse_args()
    out=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation')/a.name;data=Path('/root/autodl-tmp/liquid-depth-data/synthetic')/a.name
    out.mkdir(exist_ok=False);data.mkdir(exist_ok=False)
    files=['scripts/'+s for s in ('generate_ordinary_v37.py','generate_ordinary_v35.py','generate_other_scenes_v34.py',
        'generate_synthetic_liquid.py','generate_fullframe_v16.py','evaluate_ordinary_v37.py','run_ordinary_v37.py','summarize_other_scenes_v34.py')]
    files+=['src/liquid_depth/'+s for s in ('ordinary_temporal_v37.py','correlated_noise_v37.py','ordinary_surface_v36.py',
        'ordinary_surface_v35.py','surface_refinement.py','simulation.py','liquid_height_metrics.py','surface_memory.py')]
    frozen={f:sha(f) for f in files};(out/'frozen.json').write_text(json.dumps(frozen,indent=2))
    (out/'protocol.json').write_text(json.dumps(dict(stage=a.stage,seed=a.seed,replicates=a.replicates,frames=a.frames,
        methods=['single','joint2','joint4','mean4','guarded4'],frame_endpoint_indices=list(range(3,a.frames,4)),
        guarded4='current-only if measured clearance <3m or observed block temporal correlation >.5 or unavailable; else joint4',
        calibration='ideal fixed camera and vertical convex container; noise calibration unchanged; no liquid priors',
        noise_modes=dict(iid='independent',spatial='50 percent variance in independent 8px block fields each frame',
            persistent='50 percent variance in 8px block fields, AR1 coefficient .9 across frames'),
        margin_variance='unchanged ensemble; no framewise centering',gate='no confidence rejection, zeroing or clipping',
        condition='static flat surface; no observed motion; callers must flag motion; ToF remains current frame'),indent=2))
    start=time.time();jobs=[];logs=[]
    for worker in range(4):
        f=(out/f'generate-{worker}.log').open('w');logs.append(f)
        jobs.append(subprocess.Popen([BLENDER,'-b','-t','4','--python-exit-code','1','--python','scripts/generate_ordinary_v37.py','--',
            '--root',str(data),'--seed',str(a.seed),'--replicates',str(a.replicates),'--frames',str(a.frames),
            '--worker',str(worker),'--workers','4'],stdout=f,stderr=subprocess.STDOUT))
    while any(j.poll() is None for j in jobs):
        (out/'progress.json').write_text(json.dumps(dict(stage='generate',done=len(list(data.glob('*/noise-audit.json'))),total=33*a.replicates)))
        time.sleep(5)
    for f in logs:f.close()
    if any(j.returncode for j in jobs):raise RuntimeError('Generation failed')
    manifests=list(data.glob('*/frames.json'))
    if len(manifests)!=33*a.replicates:raise RuntimeError('Incomplete geometry generation')
    if any(len(json.loads(f.read_text()))!=a.frames*3 for f in manifests):raise RuntimeError('Incomplete noise sequences')
    (out/'progress.json').write_text(json.dumps(dict(stage='evaluate')))
    with (out/'evaluate.log').open('w') as f:
        subprocess.run([sys.executable,'scripts/evaluate_ordinary_v37.py','--data',str(data),'--output',str(out)],check=True,stdout=f,stderr=subprocess.STDOUT)
    checks={f:sha(f)==v for f,v in frozen.items()};assert all(checks.values())
    metas=[json.loads(p.read_text()) for p in data.glob('*/metadata.json')];noise=[r for f in data.glob('*/noise-audit.json') for r in json.loads(f.read_text())]
    audit={mode:{key:float(np.mean([r[key] for r in noise if r['mode']==mode and r[key] is not None]))
        for key in ('mean','variance','horizontal_adjacent_correlation','temporal_correlation')} for mode in ('iid','spatial','persistent')}
    (out/'verification.json').write_text(json.dumps(dict(source_unchanged=checks,noise_audit=audit,geometries=len(metas),
        camera_bottom_max_m=max(m['camera_to_bottom_m'] for m in metas),height_max_m=max(m['nominal_liquid_height_m'] for m in metas)),indent=2))
    (out/'progress.json').write_text(json.dumps(dict(stage='completed',elapsed_s=time.time()-start)))
if __name__=='__main__':main()
