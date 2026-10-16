"""Development-only shared-noise diagnostic; never changes the frozen policy."""
import json
import numpy as np
import cv2
import experiment_sr_v33 as run
from experiment_range_v31 import load
from liquid_depth import finite_relative_v30 as base
from liquid_depth import range_relative_v32 as model

def diagnose():
    cv2.setNumThreads(1);result=[]
    for c in run.configs('development'):
        if c['condition']!='translucent_mild':continue
        ims,masks,cal,hs=load(c);obs=run.observation_cache(c,run.BASE,ims,masks,cal)
        prior=model.fit_local([(a,b,p) for (a,b,ch),p in obs.items() if b<5],hs,cal,c['condition'])
        values=[]
        for j,cap in enumerate(c['tests']):
            with np.load(cap['evaluator_truth']) as f:h=float(f['height_m'])
            if h!=0:continue
            row=[]
            for ch in ('left','right'):
                for a in range(5):
                    pair=obs[a,j+5,ch];s=base.observed_log_scale(pair);curve=prior['curves'][f'{a}:{ch}']
                    row.append(model.inverse_curve(s,curve['scale'],curve['heights'])*1000)
            values.append(row)
        x=np.array(values);corr=np.corrcoef(x,rowvar=False)
        same_channel=[corr[a,b] for ch in (0,5) for a in range(ch,ch+5) for b in range(a+1,ch+5)]
        cross_channel=corr[:5,5:].ravel()
        result.append(dict(config_id=c['id'],zero_frames=len(x),single_reference_mean_absolute_bias_mm=float(abs(x.mean(0)).mean()),
             single_reference_mean_repeat_std_mm=float(x.std(0,ddof=1).mean()),
             same_channel_reference_error_correlation_mean=float(np.mean(same_channel)),
             cross_channel_error_correlation_mean=float(np.mean(cross_channel))))
    out=dict(configurations=result,average={k:float(np.mean([x[k] for x in result])) for k in result[0] if k not in ('config_id','zero_frames')},
             interpretation='repeated zero images only; fixed reference errors centered out by correlation; diagnostic association, not a hardware causal identification')
    run.dump(run.OUT/'shared-noise-development.json',out);print(json.dumps(out,ensure_ascii=False))

if __name__=='__main__':diagnose()
