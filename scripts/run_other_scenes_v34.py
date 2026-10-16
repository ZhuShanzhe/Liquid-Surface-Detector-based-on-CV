"""Freeze source before generation and orchestrate a fresh scene-depth evaluation."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/other-scenes-v34-20261012')
DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/other-scenes-v34-20261012')
BLENDER='/root/autodl-tmp/tools/blender/blender-4.2.0-linux-x64/blender'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ROOT.mkdir(exist_ok=False);DATA.mkdir(exist_ok=False)
    files=['scripts/generate_other_scenes_v34.py','scripts/evaluate_other_scenes_v34.py',
        'scripts/run_other_scenes_v34.py','scripts/summarize_other_scenes_v34.py',
        'src/liquid_depth/simulation.py','src/liquid_depth/surface_refinement.py',
        'src/liquid_depth/liquid_height_metrics.py','scripts/evaluate_liquid_height_v21.py',
        'scripts/generate_synthetic_liquid.py','scripts/generate_fullframe_v16.py','src/liquid_depth/illumination.py',
        '/root/autodl-tmp/liquid-depth-artifacts/training/universal-liquid-v8-ordinary-sensor-anchor/best.pth',
        '/root/autodl-tmp/liquid-depth-artifacts/training/universal-liquid-v6-v3-balanced-full/best.pth']
    (ROOT/'frozen-source.json').write_text(json.dumps({f:sha(f) for f in files},indent=2))
    (ROOT/'protocol.json').write_text(json.dumps(dict(seed=202610120341,scenes=6,geometry_clusters=99,
        rgb_scenes=594,repeats_per_rgb=4,height_bins=['empty','0-.1','.1-1']+[f'{i}-{i+1}' for i in range(1,9)],
        paired_scene_geometry=True,calibration='ideal interior bottom, pose, intrinsics, footprint; no liquid priors',
        checkpoint_frozen=True,gate='diagnostic numeric output; no positive-height/confidence screening; retain structural no-solution',
        transparent_frozen=True,extreme_failure_evaluate_only=True,no_training=True),indent=2))
    start=time.time();procs=[];handles=[]
    for w in range(4):
        f=(ROOT/f'generate-{w}.log').open('w');handles.append(f)
        procs.append(subprocess.Popen([BLENDER,'-b','-t','4','--python','scripts/generate_other_scenes_v34.py',
            '--','--root',str(DATA),'--worker',str(w),'--workers','4'],stdout=f,stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in procs):
        (ROOT/'progress.json').write_text(json.dumps(dict(stage='generating',done=len(list(DATA.glob('*/frames.json'))),
            total=594,elapsed_s=time.time()-start)))
        time.sleep(5)
    for f in handles:f.close()
    if any(p.returncode for p in procs):raise RuntimeError('Generation worker failed; see logs')
    for stage,script in [('evaluate','evaluate_other_scenes_v34.py'),('summarize','summarize_other_scenes_v34.py')]:
        (ROOT/'progress.json').write_text(json.dumps(dict(stage=stage,elapsed_s=time.time()-start)))
        with (ROOT/f'{stage}.log').open('w') as f:
            subprocess.run([sys.executable,'scripts/'+script,'--data',str(DATA),'--output',str(ROOT)],stdout=f,stderr=subprocess.STDOUT,check=True)
    frozen=json.loads((ROOT/'frozen-source.json').read_text())
    verification={f:sha(f)==v for f,v in frozen.items()}
    (ROOT/'verification.json').write_text(json.dumps(verification,indent=2))
    if not all(verification.values()):raise RuntimeError('Frozen source changed')
    (ROOT/'progress.json').write_text(json.dumps(dict(stage='completed',elapsed_s=time.time()-start)))
if __name__=='__main__':main()
