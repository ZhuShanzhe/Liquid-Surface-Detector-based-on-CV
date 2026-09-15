#!/usr/bin/env python3
"""Verify portable inference matches research evaluation and time warmed inference."""
import json,time
from pathlib import Path
import cv2,numpy as np,torch
from liquid_depth.observation_v16 import prepare_observation
from liquid_depth.models.precision_v17 import fit_surface
from predict_precision_v17 import load_pipeline
import train_precision_v17 as train
import evaluate_precision_v17b as phase_b
OUT=train.OUT
SOURCE=Path('/root/autodl-tmp/liquid-depth-data/research/sources/TransCG')
def main():
    torch.set_num_threads(4)
    fixture=OUT/'input-contract-smoke'
    rgb,raw=prepare_observation(cv2.imread(str(fixture/'rgb.png')),np.load(fixture/'raw.npy'),json.loads((fixture/'camera.json').read_text()))
    rgb=cv2.resize(rgb,(320,180));raw=cv2.resize(raw,(320,180),interpolation=cv2.INTER_NEAREST)
    rgb=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].cuda();raw=torch.from_numpy(raw[None,None]).cuda()
    records=[]
    with torch.inference_mode():
        for name in ('v16','selected','distance_blend_diagnostic'):
            pipe,config,resolved=load_pipeline(OUT,SOURCE,name)
            def run():
                p=pipe(rgb,raw)
                if config['surface_fit']:p['depth_m']=fit_surface(p['depth_m'],p['mask_probability'])
                return p
            actual=run()
            if name=='v16':reference=train.v16.make_pipeline('gated',train.v16.OUT/'gated/best.pth').eval()
            elif name=='selected':reference=phase_b.make('v17')
            else:
                freeze=json.loads((OUT/'phase-b/frozen.json').read_text())['diagnostic']
                reference=phase_b.make(freeze['name'],freeze['window_m'])
            expected=reference(rgb,raw)
            error=float((actual['depth_m']-expected['depth_m']).abs().max())
            assert error<1e-6,(name,error)
            assert torch.equal(actual['mask_probability'],expected['mask_probability'])
            del expected,reference
            for _ in range(10):run()
            torch.cuda.synchronize();durations=[]
            for _ in range(50):
                start=time.perf_counter();run();torch.cuda.synchronize();durations.append((time.perf_counter()-start)*1000)
            record=dict(variant=name,resolved=resolved,max_difference_m=error,warm_calls=10,timed_calls=50,median_ms=float(np.median(durations)),p95_ms=float(np.percentile(durations,95)),device=torch.cuda.get_device_name(0),input='RGB + raw depth + SDK metadata only, fixture has no labels',batch=1,resolution=[320,180],excludes='file I/O and SDK registration')
            records.append(record);print('VERIFY',record,flush=True)
            del pipe,actual;torch.cuda.empty_cache()
    (OUT/'portable-verification.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
