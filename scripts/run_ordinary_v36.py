"""Freeze code and select new seeds before any independent V36 evaluation."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
BLENDER='/root/autodl-tmp/tools/blender/blender-4.2.0-linux-x64/blender'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--replicates',type=int,default=9);a=p.parse_args()
    out=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation')/a.name;data=Path('/root/autodl-tmp/liquid-depth-data/synthetic')/a.name
    out.mkdir(exist_ok=False);data.mkdir(exist_ok=False)
    files=['scripts/'+s for s in ('generate_ordinary_v35.py','generate_other_scenes_v34.py','generate_synthetic_liquid.py',
        'generate_fullframe_v16.py','evaluate_ordinary_v36.py','run_ordinary_v36.py','summarize_other_scenes_v34.py')]
    files+=['src/liquid_depth/'+s for s in ('ordinary_surface_v35.py','ordinary_surface_v36.py','surface_refinement.py',
        'simulation.py','liquid_height_metrics.py','surface_memory.py')]
    frozen={f:sha(f) for f in files};(out/'frozen.json').write_text(json.dumps(frozen,indent=2))
    (out/'protocol.json').write_text(json.dumps(dict(seed=a.seed,replicates=a.replicates,repeats=6,
        development='ordinary-v35-test-20261012 now reused as development, NOT fresh evidence',
        selected='V36 quantized likelihood + bottom echo + flat-surface uniform weights + near range support widening',
        new_model_weights=False,simulator_noise_unchanged=True,gate='all finite measurements, no zeroing or clipping',
        calibration='ideal bottom, pose, intrinsics, vertical convex vessel and sensor quantization; no liquid priors',
        scope='ordinary static flat opaque liquid; ToF and transparent routes unchanged'),indent=2))
    start=time.time();jobs=[];logs=[]
    for worker in range(4):
        f=(out/f'generate-{worker}.log').open('w');logs.append(f)
        jobs.append(subprocess.Popen([BLENDER,'-b','-t','4','--python','scripts/generate_ordinary_v35.py','--',
            '--root',str(data),'--seed',str(a.seed),'--replicates',str(a.replicates),'--worker',str(worker),'--workers','4'],stdout=f,stderr=subprocess.STDOUT))
    while any(j.poll() is None for j in jobs):
        (out/'progress.json').write_text(json.dumps(dict(stage='generate',done=len(list(data.glob('*/frames.json'))))))
        time.sleep(5)
    for f in logs:f.close()
    if any(j.returncode for j in jobs):raise RuntimeError('Generation failed')
    (out/'progress.json').write_text(json.dumps(dict(stage='evaluate')))
    with (out/'evaluate.log').open('w') as f:
        subprocess.run([sys.executable,'scripts/evaluate_ordinary_v36.py','--data',str(data),'--output',str(out)],check=True,stdout=f,stderr=subprocess.STDOUT)
    verified={f:sha(f)==v for f,v in frozen.items()};assert all(verified.values())
    metas=[json.loads(p.read_text()) for p in data.glob('*/metadata.json')]
    floors=[m['dry_floor_max_ray_error_m'] for m in metas if m['empty']]
    (out/'verification.json').write_text(json.dumps(dict(source_unchanged=verified,dry_floor_cases=len(floors),
        floor_max_error_m=max(floors),camera_bottom_max_m=max(m['camera_to_bottom_m'] for m in metas),
        height_max_m=max(m['nominal_liquid_height_m'] for m in metas)),indent=2))
    (out/'progress.json').write_text(json.dumps(dict(stage='completed',elapsed_s=time.time()-start)))
if __name__=='__main__':main()
