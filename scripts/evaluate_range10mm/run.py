"""Formal evaluation entry. Outputs never live in the code checkout."""
import argparse,os,subprocess,sys
from pathlib import Path
from range_protocol import ROOT
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=ROOT)
p.add_argument('--prepare',action='store_true')
p.add_argument('--capture',action='store_true')
p.add_argument('--evaluate',action='store_true')
a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,LIQUID_EVAL_ROOT=str(a.root),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
def run(name,*args):
    subprocess.run([sys.executable,str(Path(__file__).with_name(name)),*args],env=env,check=True)
if a.prepare:run('prepare_range.py')
if a.capture:run('diverse_launch.py')
if a.evaluate:
    for attempt in range(4):
        try:run('diverse_preflight.py');break
        except subprocess.CalledProcessError:
            import json
            geometry=a.root/'preflight-geometry.json'
            if not geometry.exists() or json.loads(geometry.read_text())['errors']:raise
            phase=a.root/'preflight-phase.json'
            if not phase.exists() or not json.loads(phase.read_text())['rejected'] or attempt==3:raise
            run('diverse_launch.py')
    run('audit_capture_inputs.py')
    run('diverse_evaluate.py')
    run('finalize_diverse.py')
