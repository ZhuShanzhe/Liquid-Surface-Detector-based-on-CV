import json,time
import cv2,numpy as np
import experiment_sr_v33 as run
from experiment_range_v31 import load
from liquid_depth.range_runtime_v33 import FivePriorImageMeter

cv2.setNumThreads(1);result=[]
sr=None
for width in (136,152):
    c=next(c for c in run.configs('development') if c['condition']=='translucent_mild' and c['empty_width_px']==width)
    ims,masks,cal,hs=load(c)
    for name in run.reg.OPTIONS:
        if 'swinir' in name and sr is None:sr=run.reg.SwinIRX2()
        t=time.perf_counter();meter=FivePriorImageMeter(cal,masks,[{ch:ims[j,ch] for ch in ('left','right')} for j in range(5)],hs,c['condition'],name,sr_backend=sr)
        prior=time.perf_counter()-t;times=[]
        for j in [0,1,15,25,35,45]:
            t=time.perf_counter();meter.measure({ch:ims[j+5,ch] for ch in ('left','right')});times.append((time.perf_counter()-t)*1000)
        result.append(dict(method=name,width_px=width,prior_s=prior,p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),frames=len(times)))
run.dump(run.OUT/'development-runtime.json',result);print(json.dumps(result))
