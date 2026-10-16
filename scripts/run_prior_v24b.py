"""Frozen, bounded independent test; does not run development selection."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path

repo=Path(__file__).resolve().parents[1]
out=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/prior-reliability-v24b-20261011')
policy=out/'frozen-calibration-policy.json'
assert policy.is_file()
if (out/'frozen-source.json').exists():raise FileExistsError('Independent test already started')
files=[repo/p for p in ['src/liquid_depth/prior_reliability_v24.py','src/liquid_depth/refraction_images_v23.py',
    'src/liquid_depth/refraction_matching_v23.py','src/liquid_depth/refraction_inverse_v23.py',
    'scripts/experiment_image_refraction_v23.py','scripts/experiment_prior_v24b.py','scripts/run_prior_v24b.py',
    'tests/test_prior_reliability_v24.py']]+[policy]
frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(out/'frozen-source.json').write_text(json.dumps(frozen,indent=2))
env=dict(os.environ,PYTHONPATH='src:scripts');start=time.time()
stages=[('regression',[sys.executable,'-m','pytest','-q']),
    ('generate',[sys.executable,'scripts/experiment_prior_v24b.py','generate']),
    ('test',[sys.executable,'scripts/experiment_prior_v24b.py','test'])]
for stage,cmd in stages:
    (out/'progress.json').write_text(json.dumps(dict(stage=stage,status='running',pid=os.getpid(),elapsed_s=time.time()-start)))
    with (out/(stage+'.log')).open('w') as log:
        result=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        (out/'progress.json').write_text(json.dumps(dict(stage=stage,status='failed',returncode=result.returncode)))
        raise SystemExit(result.returncode)
unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items())
(out/'progress.json').write_text(json.dumps(dict(stage='complete',status='complete',elapsed_s=time.time()-start,source_and_policy_unchanged=unchanged)))
if not unchanged:raise RuntimeError('Frozen source or policy modified during test')
