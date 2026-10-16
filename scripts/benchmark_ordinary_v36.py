"""Isolated sequential estimator latency; excludes disk, capture, network, GUI."""
import json,time
from pathlib import Path
from dataclasses import replace
import cv2,numpy as np
from liquid_depth.ordinary_surface_v35 import estimate_ordinary
from liquid_depth.ordinary_surface_v36 import estimate_v36
from liquid_depth.liquid_height_metrics import HeightCalibration
root=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/ordinary-v36-test-20261012')
rows=json.loads((root/'raw.json').read_text())['records'][::18];cv2.setNumThreads(1)
timing={'v35':[],'v36':[]}
for i,r in enumerate(rows):
    m=json.loads((Path(r['frame']['folder'])/'metadata.json').read_text());raw=np.load(r['frame']['raw_path'])
    cal=replace(HeightCalibration.from_simulation(m),bottom_z_m=m['interior_bottom_z_m'])
    for method in timing:
        start=time.perf_counter()
        if method=='v35':p=estimate_ordinary(None,raw,cal,m['container_rim_z_m'],m['sensor_family'],0)
        else:p=estimate_v36(None,raw,cal,m['container_rim_z_m'],m['sensor_family'],uniform=True)
        elapsed=(time.perf_counter()-start)*1000
        if i>=3:timing[method].append(elapsed)
        if p['statistics'] is not None:
            assert abs(p['statistics']['mean_m']-r['methods'][method]['statistics']['mean_m'])<1e-7
out={m:dict(frames=len(v),p50_ms=float(np.percentile(v,50)),p95_ms=float(np.percentile(v,95)),p99_ms=float(np.percentile(v,99)),max_ms=max(v)) for m,v in timing.items()}
(root/'latency-isolated.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
