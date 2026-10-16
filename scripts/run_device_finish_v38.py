"""Predeclared aspect-ratio control, independent prior completion, and reports."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/device-v38-test-20261012')
DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/device-v38-test-20261012')
def main():
    os.chdir(REPO);env=dict(os.environ,PYTHONPATH='src:scripts',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    source=[REPO/'scripts'/s for s in ('evaluate_device_letterbox_v38.py','finalize_prior_device_v38.py','run_device_finish_v38.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    (ROOT/'adapter-plan.json').write_text(json.dumps(dict(primary_preprocessing='aspect-preserving letterbox to 320x256',
         comparison_preprocessing='anisotropic resize to 320x256',reason='preserve native optical aspect ratio',
         selected_by_test_errors=False,source=hashes),indent=2))
    while not (ROOT/'summary.json').exists():time.sleep(5)
    with (ROOT/'letterbox.log').open('w') as log:
        subprocess.run([sys.executable,'scripts/evaluate_device_letterbox_v38.py','--data',str(DATA),'--output',str(ROOT/'letterbox')],
                       stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    while json.loads((ROOT/'progress.json').read_text()).get('stage')!='completed':time.sleep(5)
    for script in ('finalize_prior_device_v38.py','report_device_v38.py','analyze_device_v38.py'):
        with (ROOT/(script+'.log')).open('w') as log:
            subprocess.run([sys.executable,'scripts/'+script],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    check={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()}
    if not all(check.values()):raise RuntimeError('adapter_source_changed')
    (ROOT/'final-verification.json').write_text(json.dumps(dict(adapter=check,all_original_sources_unchanged=all(json.loads((ROOT/'verification.json').read_text()).values()),
          native_observations=6864,independent_prior_observations=960,completed=True),indent=2))
    print('FINAL_COMPLETE',flush=True)
if __name__=='__main__':main()
