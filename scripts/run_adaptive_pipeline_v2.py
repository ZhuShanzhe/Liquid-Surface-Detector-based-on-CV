"""Frozen geometry backup, grouped multi-head training, learned optional-use policy."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

BASE=Path('/root/autodl-tmp');REPO=BASE/'Liquid-Surface-Detector-based-on-CV'
ROOT=BASE/'liquid-depth-data/synthetic/adaptive-evidence-v2-20261015'
OUT=BASE/'liquid-depth-artifacts/adaptive-route-v2-20261015'
PY=BASE/'envs/liquid-depth/bin/python';SEED=202610154119


def status(stage,**kw):
    (OUT/'status.json').write_text(json.dumps({'stage':stage,'time_unix':time.time(),**kw},indent=2))


def run(stage,args):
    status(stage)
    env={**os.environ,'PYTHONPATH':str(REPO/'src')+':'+str(REPO/'scripts'),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'}
    with (OUT/f'{stage}.log').open('w') as log:
        subprocess.run([str(PY),*args],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    expected=BASE/'liquid-depth-archives/determined-symmetric-geometry-v1-20261015/manifest.json'
    manifest=json.loads(expected.read_text())
    for p,digest in manifest['source_sha256'].items():
        if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=digest:raise ValueError('frozen_geometric_source_changed')
    if len(list(ROOT.glob('*/supervision_offline_01.npz')))!=792:raise ValueError('expanded_capture_not_complete')
    plan={'seed':SEED,'installations':792,'train':396,'validation':198,'test':198,
          'input':'aligned RGB, depth, empty depth and installed container geometry only',
          'supervision':'independent semantic, simulator-origin and metric-consistency labels',
          'float_diversity':'block, bottle, ellipsoid, foam; 3-48 projected pixels; 30-300mm protrusion',
          'baseline':str(expected),'modes':['off','on','auto'],
          'candidate_policy':'choose auto threshold on validation only; geometric fallback when not justified',
          'liquid_tolerance':'max(20mm,2% actual bottom-relative liquid depth)',
          'highest_tolerance':'max(50mm,5% actual highest bottom-relative height)',
          'no_zero_snap_no_output_clip':True,'record_all_frames_including_unobservable':True,
          'transparency':'original five-prior path retained; not retrained here'}
    (OUT/'experiment-plan.json').write_text(json.dumps(plan,indent=2))
    run('pixel-training',['scripts/train_rgbd_evidence_v2.py','--root',str(ROOT),'--output',str(OUT/'pixel-model'),
                          '--seed',str(SEED),'--epochs','50'])
    run('candidate-training',['scripts/train_candidate_trust_v2.py','--root',str(ROOT),'--model',str(OUT/'pixel-model/best.pt'),
                              '--output',str(OUT/'candidate-model'),'--seed',str(SEED)])
    run('evaluation',['scripts/evaluate_adaptive_route_v2.py','--root',str(ROOT),'--model',str(OUT/'pixel-model/best.pt'),
                      '--gate',str(OUT/'candidate-model/gate.pt'),'--output',str(OUT/'evaluation'),'--seed',str(SEED)])
    status('complete')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():status('failed',error=repr(exc))
        raise
