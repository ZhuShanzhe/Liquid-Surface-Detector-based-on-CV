"""Freeze development choices before creating independent images and labels."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import cv2
from experiment_transparent_v27 import OUT,DATA,LAYOUTS,REPO,generate,predict26,rows,summary,dump,progress
from select_transparent_v27 import predict

def main():
    started=time.time()
    if (OUT/'frozen-source.json').exists():raise FileExistsError('frozen test already started')
    policy=json.loads((OUT/'frozen-policy.json').read_text())
    files=[REPO/p for p in ['src/liquid_depth/transparent_refinement_v27.py','src/liquid_depth/transparent_refinement_v26.py',
        'src/liquid_depth/transparent_diagnostic_v25.py','src/liquid_depth/refraction_matching_v26.py',
        'src/liquid_depth/refraction_matching_v23.py','src/liquid_depth/refraction_images_v23.py',
        'scripts/experiment_transparent_v27.py','scripts/select_transparent_v27.py','scripts/run_transparent_v27.py',
        'tests/test_transparent_refinement_v27.py']]+[OUT/'layout-policy.json',OUT/'frozen-policy.json',
        Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/transparent-refinement-v26-20261011/frozen-policy.json')]
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    dump(OUT/'frozen-source.json',frozen)
    env=dict(os.environ,PYTHONPATH='src:scripts')
    progress('regression',status='running')
    with (OUT/'regression.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:progress('regression',status='failed');raise SystemExit(result.returncode)
    generate('test',policy['layouts'])
    configs=json.loads((DATA/'test/manifest.json').read_text())['configs']
    records=[];fits=[]
    for i,c in enumerate(configs):
        key=c['family']+'/'+c['condition'];layout=policy['layouts'][key]
        for candidate in LAYOUTS:
            preds,p=predict26(c,candidate)
            records+=rows(c,preds,'v26_'+candidate,candidate)
            fits.append(dict(config_id=c['id'],stage='v26_'+candidate,prior=p))
        preds,p=predict(c,layout,policy['bias'][key]);records+=rows(c,preds,'v27_final',layout)
        fits.append(dict(config_id=c['id'],stage='v27_final',prior=p))
        progress('evaluate_test',status='running',configs=i+1,total_configs=len(configs))
        print('TEST',i+1,len(configs),flush=True)
    dump(OUT/'test-raw.json',dict(records=records,fits=fits))
    metrics=[]
    for stage in ['v26_'+k for k in LAYOUTS]+['v27_final']:
        rs=[r for r in records if r['stage']==stage]
        metrics.append(dict(stage=stage,overall=summary(rs),far_shallow=summary([
            r for r in rs if r['bottom_m']>=6 and r['true_height_m']/r['bottom_m']<.18])))
    dump(OUT/'test-summary.json',metrics)
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in frozen.items())
    progress('complete',status='complete',source_and_policy_unchanged=unchanged,elapsed_s=time.time()-started)
    if not unchanged:raise RuntimeError('frozen source changed')
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':cv2.setNumThreads(4);main()
