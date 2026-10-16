"""Post-hoc attribution only; no tuning and no deployable oracle claims."""
import json
from pathlib import Path
from dataclasses import replace
from collections import defaultdict
import cv2,numpy as np
from liquid_depth.ordinary_surface_v35 import interior_ray_roi
from liquid_depth.liquid_height_metrics import HeightCalibration

root=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/ordinary-v35-test-20261012')
rows=json.loads((root/'raw.json').read_text())['records'];groups=defaultdict(list)
for r in rows:
    m=json.loads((Path(r['frame']['folder'])/'metadata.json').read_text())
    cal=replace(HeightCalibration.from_simulation(m),bottom_z_m=m['interior_bottom_z_m'])
    roi=interior_ray_roi(cal,m['container_rim_z_m'],(m['height'],m['width']),.65)
    if roi.sum()<256:roi=interior_ray_roi(replace(cal,bottom_z_m=m['container_rim_z_m']-1e-5),m['container_rim_z_m'],roi.shape,.65)
    raw=np.load(r['frame']['raw_path']);geom=np.load(Path(r['frame']['folder'])/'geometry.npz')
    valid=cv2.erode(roi.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&(raw>0)
    truth=geom['mask'].astype(bool);target=geom['target'];selected=valid&truth
    p=r['methods']['core_0']
    row=dict(cluster=r['cluster'],selected=int(valid.sum()),outside_true_fraction=float((valid&~truth).sum()/max(valid.sum(),1)),
        error_mm=p.get('signed_error_m',0)*1000,bad=not p.get('within_tolerance',False),
        raw_depth_mae_mm=float(abs(raw[selected]-target[selected]).mean()*1000) if selected.any() else None)
    keys=['all']
    if r['camera_to_surface_m']>=3 and r['true_height_m']<1:keys+=['far_shallow','far_shallow/'+r['sensor']]
    for key in keys:groups[key].append(row)
def stat(q):
    cs=defaultdict(list)
    for r in q:cs[r['cluster']].append(r['error_mm'])
    return dict(frames=len(q),failed=sum(r['bad'] for r in q),
        mean_selected_points=float(np.mean([r['selected'] for r in q])),
        mean_non_surface_fraction=float(np.mean([r['outside_true_fraction'] for r in q])),
        maximum_non_surface_fraction=max(r['outside_true_fraction'] for r in q),
        mean_single_pixel_depth_mae_mm=float(np.mean([r['raw_depth_mae_mm'] for r in q if r['raw_depth_mae_mm'] is not None])),
        mean_within_case_repeat_std_mm=float(np.mean([np.std(x,ddof=1) for x in cs.values()])),
        mean_abs_case_bias_mm=float(np.mean([abs(np.mean(x)) for x in cs.values()])))
out={k:stat(q) for k,q in groups.items()};(root/'attribution.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
