import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import replay_ordinary_v39 as base
from liquid_depth.ordinary_surface_v39b import estimate_v39
base.estimate_v39=estimate_v39

if __name__=='__main__':
    root=Path('/root/autodl-tmp/liquid-depth-data/synthetic/device-v38-test-20261012')
    out=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/ordinary-v39b-replay-20261012');out.mkdir(exist_ok=False)
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for records in pool.map(base.run,sorted(x.parent for x in root.glob('*/metadata.json'))):rows+=records
    (out/'records.json').write_text(json.dumps(rows,allow_nan=False))
    (out/'summary.json').write_text(json.dumps(base.summarize(rows),indent=2,allow_nan=False))
    (out/'complete.json').write_text(json.dumps(dict(records=len(rows),purpose='Secondary rendered-frame replay, observed-return mixture')))
    print('COMPLETE',len(rows),flush=True)
