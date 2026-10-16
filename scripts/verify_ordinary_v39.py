import json,hashlib,subprocess,time,sys
from pathlib import Path
import numpy as np
from experiment_ordinary_v39 import configuration,clean_maps,PROFILE,MODEL
from liquid_depth.device_profiles_v38 import noisy_depth
from liquid_depth.ordinary_surface_v39 import calibrate_dry_reference
from liquid_depth.ordinary_surface_v39b import estimate_v39
repo=Path(__file__).parents[1]
root=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/ordinary-v39b-test-20261012')
protocol=json.loads((root/'protocol.json').read_text());complete=json.loads((root/'complete.json').read_text())
for file,digest in protocol['source_hashes'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
rows=[json.loads(x) for x in (root/'records.jsonl').read_text().splitlines()]
assert len(rows)==544*2*4*2==complete['records']
assert len({r['cluster'] for r in rows})==544
assert len({(r['cluster'],r['regime'],r['repeat']) for r in rows})==4352
assert len({(r['cluster'],r['regime'],r['repeat'],r['method']) for r in rows})==len(rows)
assert max(r['camera_to_bottom_m'] for r in rows)<=10
tests=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_ordinary_v39b.py','tests/test_ordinary_v39.py','tests/test_ordinary_v35.py','tests/test_ordinary_v36.py','tests/test_ordinary_temporal_v37.py','tests/test_device_profiles_v38.py'],cwd=repo,text=True,capture_output=True)
assert tests.returncode==0,tests.stdout+tests.stderr
timings=[]
for index,group in [(0,'far_shallow'),(1,'uniform_7')]:
    _,cal,rim,h,_,_,_=configuration(202610124101,index,group)
    dry,wet,_,_=clean_maps(cal,rim,h)
    ref=calibrate_dry_reference(noisy_depth(dry,PROFILE,391001,99),cal,rim,MODEL)
    raw=noisy_depth(wet,PROFILE,391001,0)
    estimate_v39(raw,cal,rim,MODEL,ref)
    for i in range(10):
        start=time.perf_counter();estimate_v39(raw,cal,rim,MODEL,ref);timings.append(time.perf_counter()-start)
result=dict(frozen_sources_match=True,independent_configurations=544,observations=4352,method_records=len(rows),
            test_output=tests.stdout,tests_returncode=tests.returncode,
            single_process_estimator_timing_s=dict(p50=float(np.median(timings)),p95=float(np.percentile(timings,95)),samples=len(timings),scope='two geometric inputs; estimator only; no RGB, acquisition or I/O'),
            files={str(x.relative_to(repo)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [repo/'src/liquid_depth/ordinary_surface_v39.py',repo/'src/liquid_depth/ordinary_surface_v39b.py',*sorted(repo.glob('scripts/*ordinary_v39*.py')),repo/'scripts/inspect_bias_v39.py',repo/'tests/test_ordinary_v39.py',repo/'tests/test_ordinary_v39b.py']})
(root/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
