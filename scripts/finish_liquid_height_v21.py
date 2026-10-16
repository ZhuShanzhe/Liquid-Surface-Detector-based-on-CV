#!/usr/bin/env python3
"""Frozen-model testing and report generation; no test-driven model selection."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--training-pid',type=int,required=True)
a=p.parse_args();root=Path(a.root);network=root/'network-finetune'
status=root/'finalization-progress.json'
def save_status(stage,**kw):status.write_text(json.dumps(dict(stage=stage,**kw),indent=2))
started=time.time();save_status('waiting_for_frozen_training')
while not (network/'frozen.json').exists():
    if time.time()-started>3600:raise TimeoutError('Training timeout; inspect logs')
    try:os.kill(a.training_pid,0)
    except ProcessLookupError:
        save_status('training_failed');raise RuntimeError('Training ended without frozen model')
    time.sleep(5)
frozen=json.loads((network/'frozen.json').read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert digest(network/'best.pth')==frozen['checkpoint_sha256']
data=Path('/root/autodl-tmp/liquid-depth-data/synthetic')
test=root/'height-network-tests';test.mkdir(exist_ok=False)
manifests=[('new-c','liquid-height-v21-fresh-c-20261011'),('historical-a','precision-v17-fresh'),('historical-b','precision-v17b-fresh')]
protocol=dict(frozen_checkpoint_sha256=frozen['checkpoint_sha256'],selected_epoch=frozen['selected_epoch'],
    test_started_after_frozen_checkpoint=True,new_c_seed=20261011071,
    manifests={name:digest(data/folder/'manifest.csv') for name,folder in manifests},
    no_threshold_or_epoch_selection_on_tests=True)
(test/'frozen-test-protocol.json').write_text(json.dumps(protocol,indent=2))
for name,folder in manifests:
    for variant in ('baseline','height-aware'):
        save_status('evaluating',dataset=name,variant=variant)
        output=test/f'{name}-{variant}.json'
        command=[sys.executable,'scripts/evaluate_liquid_height_v21.py','--backend','v17','--manifest',str(data/folder/'manifest.csv'),
            '--output',str(output),'--label',variant]
        if variant=='height-aware':command+=['--checkpoint',str(network/'best.pth')]
        with (test/f'{name}-{variant}.stdout').open('w') as stdout,(test/f'{name}-{variant}.stderr').open('w') as stderr:
            subprocess.run(command,stdout=stdout,stderr=stderr,check=True)
for backend,name in (('raw','new-c-raw'),('v17_raw','new-c-v17-raw')):
    output=root/f'{name}.json'
    if output.exists():continue
    save_status('raw_point_diagnostic',backend=backend)
    with (root/f'{name}.log').open('w') as stdout:
        subprocess.run([sys.executable,'scripts/evaluate_liquid_height_v21.py','--backend',backend,
            '--manifest',str(data/'liquid-height-v21-fresh-c-20261011/manifest.csv'),
            '--output',str(output)],stdout=stdout,stderr=stdout,check=True)
confidence=root/'confidence-height-v8-engineering-final.json'
if not confidence.exists():
    save_status('calibrating_actual_height_confidence')
    with (root/'confidence-engineering.log').open('w') as stdout:
        subprocess.run([sys.executable,'scripts/calibrate_scenario_confidence.py',
            '--checkpoint','/root/autodl-tmp/liquid-depth-artifacts/training/universal-liquid-v8-ordinary-sensor-anchor/best.pth',
            '--manifest',str(data/'liquid-sim-v3-calibration-top-12000/manifest.csv'),
            '--thresholds','0.30,0.50,0.70,0.90','--output',str(confidence)],stdout=stdout,stderr=stdout,check=True)
save_status('writing_report')
subprocess.run([sys.executable,'scripts/report_liquid_height_v21.py','--root',str(root)],check=True)
save_status('completed',elapsed_s=time.time()-started)
