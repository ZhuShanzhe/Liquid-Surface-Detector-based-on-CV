#!/usr/bin/env python3
"""One bounded generation/evaluation job with immutable source fingerprint."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path

repo=Path(__file__).resolve().parents[1]
out=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/image-refraction-v23-20261011/final')
data=Path('/root/autodl-tmp/liquid-depth-data/synthetic/image-refraction-v23-final-20261011')
out.mkdir(exist_ok=False)
sources=[repo/'scripts/experiment_image_refraction_v23.py',repo/'scripts/run_image_refraction_v23.py']
sources+=list((repo/'src/liquid_depth').glob('*v23.py'))
fingerprint={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
(out/'frozen-source.json').write_text(json.dumps(fingerprint,indent=2))
env=dict(os.environ,PYTHONPATH='src:scripts')
stages=[('tests',[sys.executable,'-m','pytest','-q','tests/test_refraction_images_v23.py']),
        ('generate',[sys.executable,'scripts/experiment_image_refraction_v23.py','generate','--data',str(data)]),
        ('evaluate',[sys.executable,'scripts/experiment_image_refraction_v23.py','evaluate','--data',str(data),'--output',str(out/'evaluation')])]
start=time.time()
for name,command in stages:
    (out/'progress.json').write_text(json.dumps(dict(stage=name,status='running',pid=os.getpid(),elapsed_s=time.time()-start)))
    with (out/(name+'.log')).open('w') as log:
        result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        (out/'progress.json').write_text(json.dumps(dict(stage=name,status='failed',returncode=result.returncode,elapsed_s=time.time()-start)))
        raise SystemExit(result.returncode)
unchanged=all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in fingerprint.items())
(out/'progress.json').write_text(json.dumps(dict(stage='complete',status='complete',elapsed_s=time.time()-start,source_unchanged=unchanged)))
if not unchanged:raise RuntimeError('Source changed during frozen evaluation')
