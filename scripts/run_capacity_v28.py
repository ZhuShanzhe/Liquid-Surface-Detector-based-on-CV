"""Freeze layout choices and source before generating a new capacity-limited test."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import cv2
from capacity_priors_v28 import OUT,DATA,REPO,LAYOUTS,generate,predict26,result_rows,summary,dump,progress

def main():
    start=time.time()
    if (OUT/'frozen-source.json').exists():raise FileExistsError('frozen evaluation already started')
    policy=json.loads((OUT/'frozen-policy.json').read_text())
    files=[REPO/p for p in ['scripts/capacity_priors_v28.py','scripts/run_capacity_v28.py','scripts/audit_capacity_v28.py',
        'tests/test_capacity_priors_v28.py','scripts/experiment_transparent_v27.py',
        'src/liquid_depth/transparent_refinement_v26.py','src/liquid_depth/transparent_diagnostic_v25.py',
        'src/liquid_depth/refraction_images_v23.py','src/liquid_depth/refraction_matching_v23.py',
        'src/liquid_depth/refraction_matching_v26.py']]+[OUT/'frozen-policy.json',
        Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/transparent-refinement-v26-20261011/frozen-policy.json')]
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};dump(OUT/'frozen-source.json',frozen)
    env=dict(os.environ,PYTHONPATH='src:scripts')
    progress('regression',status='running')
    with (OUT/'regression.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:progress('regression',status='failed');raise SystemExit(result.returncode)
    generate('test')
    configs=json.loads((DATA/'test/manifest.json').read_text())['configs'];records=[];fits=[]
    for i,c in enumerate(configs):
        key=c['family']+'/'+c['condition'];selected=policy['selection'][key]
        for layout in LAYOUTS:
            preds,prior=predict26(c,layout);rs=result_rows(c,preds,layout,layout);records+=rs
            fits.append(dict(config_id=c['id'],layout=layout,prior=prior))
            if layout==selected:records+=[dict(r,stage='selected') for r in rs]
        progress('evaluate_test',configs=i+1,total_configs=len(configs),status='running')
        print('TEST',i+1,len(configs),flush=True)
    dump(OUT/'test-raw.json',dict(records=records,fits=fits))
    metrics=[]
    for stage in list(LAYOUTS)+['selected']:
        rs=[r for r in records if r['stage']==stage]
        metrics.append(dict(stage=stage,overall=summary(rs),far=summary([r for r in rs if r['bottom_m']>=6])))
    dump(OUT/'test-summary.json',metrics)
    print(json.dumps(metrics,indent=2),flush=True)
    progress('audit',status='running')
    with (OUT/'audit.log').open('w') as log:
        result=subprocess.run([sys.executable,'scripts/audit_capacity_v28.py'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:progress('audit',status='failed');raise SystemExit(result.returncode)
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items())
    progress('complete',status='complete',source_and_policy_unchanged=unchanged,elapsed_s=time.time()-start)
    if not unchanged:raise RuntimeError('frozen source changed during evaluation')

if __name__=='__main__':
    cv2.setNumThreads(4);main()
