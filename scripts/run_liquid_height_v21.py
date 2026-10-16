#!/usr/bin/env python3
"""Non-overwriting evaluation and height-head experiment orchestration."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
data=Path('/root/autodl-tmp/liquid-depth-data/synthetic')
start=time.time();completed=[]

def run(name,script,*args):
    status=dict(stage=name,completed=completed,elapsed_s=time.time()-start,state='running')
    (out/'progress.json').write_text(json.dumps(status,indent=2))
    with (out/f'{name}.stdout').open('w') as stdout,(out/f'{name}.stderr').open('w') as stderr:
        code=subprocess.call([sys.executable,'scripts/'+script,*map(str,args)],stdout=stdout,stderr=stderr)
    if code:
        status.update(state='failed',exit_code=code)
        (out/'progress.json').write_text(json.dumps(status,indent=2));raise SystemExit(code)
    completed.append(name)

def evaluate(name,backend,manifest,split='test'):
    run(name,'evaluate_liquid_height_v21.py','--backend',backend,'--manifest',data/manifest/'manifest.csv',
        '--split',split,'--output',out/f'{name}.json')

for backend in ('v6','v8','v17','raw'):
    evaluate(f'all-scenes-{backend}',backend,'liquid-sim-v3-calibration-top-12000')
evaluate('development-train-v17','v17','observable-v16-development','train')
evaluate('development-val-v17','v17','observable-v16-development','val')
run('train-height-head','optimize_liquid_height_v21.py','train',
    '--train',out/'development-train-v17.json','--val',out/'development-val-v17.json','--output',out/'height-head')
for name,dataset in [('fresh-a','precision-v17-fresh'),('fresh-b','precision-v17b-fresh')]:
    evaluate(name+'-v17','v17',dataset)
    run(name+'-v21','optimize_liquid_height_v21.py','evaluate','--input',out/(name+'-v17.json'),
        '--head',out/'height-head/height-head.pth','--output',out/(name+'-v21.json'))
run('all-scenes-v21-diagnostic','optimize_liquid_height_v21.py','evaluate','--input',out/'all-scenes-v17.json',
    '--head',out/'height-head/height-head.pth','--output',out/'all-scenes-v21-diagnostic.json')
(out/'progress.json').write_text(json.dumps(dict(state='completed',completed=completed,elapsed_s=time.time()-start),indent=2))
print('ALL_COMPLETED',time.time()-start,flush=True)
