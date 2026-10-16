"""Fresh capacity-jittered scenes with consistent floating-surface sensor noise."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch

BASE=Path('/root/autodl-tmp');REPO=BASE/'Liquid-Surface-Detector-based-on-CV'
OUT=BASE/'liquid-depth-artifacts/adaptive-route-v2-20261015'
ROOT=BASE/'liquid-depth-data/synthetic/adaptive-evidence-v2-confirmation-20261015'
PY=BASE/'envs/liquid-depth/bin/python';SEED=202610154229


def status(stage,**kw):
    (OUT/'confirmation-status.json').write_text(json.dumps({'stage':stage,'time_unix':time.time(),**kw},indent=2))


def main():
    env={**os.environ,'PYTHONPATH':str(REPO/'src')+':'+str(REPO/'scripts'),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'}
    processes=[];logs=[];status('generating_fresh_sensor_consistent_scenes')
    for worker in range(4):
        log=(OUT/f'confirmation-generate-{worker}.log').open('w');logs.append(log)
        processes.append(subprocess.Popen([str(BASE/'tools/blender/blender-4.2.0-linux-x64/blender'),'-b','-t','4',
            '--python','scripts/generate_adaptive_evidence_v2.py','--','--root',str(ROOT),'--seed',str(SEED),
            '--start-index','594','--count','792','--worker',str(worker),'--workers','4',
            '--consistent-float-sensor','--random-headroom'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in processes):
        status('generating_fresh_sensor_consistent_scenes',complete=len(list(ROOT.glob('*/supervision_offline_01.npz'))) if ROOT.exists() else 0);time.sleep(15)
    if any(p.returncode for p in processes) or len(list(ROOT.glob('*/supervision_offline_01.npz')))!=198:raise RuntimeError('fresh_capture_failed')
    for log in logs:log.close()
    status('waiting_for_prior_training')
    while True:
        p=OUT/'prior-status.json';s=json.loads(p.read_text()) if p.exists() else {}
        if s.get('stage')=='failed':raise RuntimeError('prior_training_failed:'+s.get('error',''))
        if s.get('stage')=='complete':break
        time.sleep(15)
    # Keep both variants. Main default is selected by validation semantic score only.
    options=['pixel-model','pixel-model-with-real']
    val={n:torch.load(OUT/n/'best.pt',map_location='cpu',weights_only=False)['validation']['score'] for n in options}
    selected=max(val,key=val.get)
    files=[REPO/'src/liquid_depth/adaptive_rgbd_meter.py',REPO/'src/liquid_depth/rgbd_evidence_model.py',
           REPO/'scripts/evaluate_adaptive_route_v2.py',REPO/'scripts/generate_adaptive_evidence_v2.py']
    (OUT/'confirmation-frozen-protocol.json').write_text(json.dumps({'seed':SEED,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
       'validation_scores':val,'validation_selected_default':selected,'fresh_installations':198,
       'changes':['new seed','randomized rim freeboard, no fixed 120mm fill relation','float stereo/SL disparity noise and quantization included'],
       'not_used_for_training_or_threshold_selection':True},indent=2))
    for model,name,gate in [('pixel-model','confirmation-synthetic','candidate-model'),
                            ('pixel-model-with-real','confirmation-with-real','candidate-model-with-real')]:
        status('evaluating_fresh',model=model)
        with (OUT/f'{name}.log').open('w') as log:
            subprocess.run([str(PY),'scripts/evaluate_adaptive_route_v2.py','--root',str(ROOT),'--model',str(OUT/model/'best.pt'),
                '--gate',str(OUT/gate/'gate.pt'),'--output',str(OUT/name),'--seed',str(SEED)],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    status('complete',validation_selected_default=selected)


if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
