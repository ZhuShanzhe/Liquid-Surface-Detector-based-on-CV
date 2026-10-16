"""Oracle-geometry-only counterfactual for the eight failed installations."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluate_unrestricted_surface_v51 import make_masks, median_height
from liquid_depth.liquid_height_metrics import HeightCalibration, reference_statistics

ROOT=Path('/root/autodl-tmp/liquid-depth-data/synthetic/uniform-native-v50-confirmation-20261013')
failures=json.load(open('/root/autodl-tmp/liquid-depth-artifacts/evaluation/unrestricted-v51b-v50-attribution-20261013/geometry_failures.json'))
rows=[]
for failure in failures:
    case,rep=failure['case'],failure['repeat']
    folder=ROOT/f'{case*6+5:05d}'
    meta=json.loads((folder/'metadata.json').read_text())
    cal=replace(HeightCalibration.from_simulation(meta),
                bottom_z_m=float(meta['interior_bottom_z_m']))
    rim=float(meta['container_rim_z_m'])
    frame=json.loads((folder/'frames.json').read_text())[rep]
    raw=np.load(frame['raw_path'])
    _,_,container=make_masks(raw,cal,rim)
    prediction,n=median_height(raw,cal,container)
    truth=float(reference_statistics(meta,cal)['mean_m'])
    rows.append({'case':case,'repeat':rep,'sensor':meta['sensor_family'],
                 'band':meta['band'],'true_height_m':truth,
                 'oracle_geometry_prediction_m':prediction,
                 'oracle_geometry_error_mm':None if prediction is None else
                    1000*(prediction-truth),
                 'oracle_geometry_selected_pixels':n,
                 'use':'offline diagnosis only; no installed geometry available'})
out=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/unrestricted-v51b-v50-attribution-20261013/geometry_failure_oracle_counterfactual.json')
out.write_text(json.dumps(rows,indent=2))
for row in rows:
    print(json.dumps(row),flush=True)
