"""Offline raw-versus-target return comparison for selected large errors."""
import json
from pathlib import Path

import numpy as np

from liquid_depth.liquid_height_metrics import HeightCalibration

ROOT=Path('/root/autodl-tmp/liquid-depth-data/synthetic/uniform-native-v50-confirmation-20261013')
for case,repeat in ((50,1),(29,3),(38,2),(41,0),(20,0)):
    folder=ROOT/f'{case*6+5:05d}'
    meta=json.loads((folder/'metadata.json').read_text())
    with np.load(folder/'geometry.npz') as z:
        target=z['target']
        mask=z['mask'].astype(bool)
    frame=json.loads((folder/'frames.json').read_text())[repeat]
    raw=np.load(frame['raw_path'])
    good=mask&np.isfinite(raw)&(raw>0)&np.isfinite(target)&(target>0)
    t=target[good]
    r=raw[good]
    k=HeightCalibration.from_simulation(meta).intrinsics
    sensor=meta['sensor_family']
    output={'case':case,'repeat':repeat,'sensor':sensor,'band':meta['band'],
            'valid_surface_pixels':int(good.sum()),
            'target_depth_median_m':float(np.median(t)),
            'raw_depth_median_m':float(np.median(r)),
            'raw_minus_target_depth_median_mm':float(1000*np.median(r-t)),
            'raw_minus_target_depth_p05_p95_mm':(1000*np.percentile(r-t,[5,95])).tolist(),
            'raw_gt_100mm_fraction':float(np.mean(abs(r-t)>.1))}
    if sensor!='tof':
        baseline={'active_stereo':.055,'structured_light':.035}[sensor]
        fb=float(k[0,0]*baseline)
        dr=fb/r
        dt=fb/t
        output.update(fb_px_m=fb,
                      true_disparity_median_px=float(np.median(dt)),
                      measured_disparity_median_px=float(np.median(dr)),
                      disparity_error_median_px=float(np.median(dr-dt)),
                      disparity_quantization_step_px=.0625)
    print(json.dumps(output),flush=True)
