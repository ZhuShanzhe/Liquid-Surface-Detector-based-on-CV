"""Post-test analysis and serial inference timings; never tunes inference."""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
from collections import Counter
import cv2,numpy as np
from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.ordinary_temporal_v37 import estimate_window,estimate_guarded4

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cv2.setNumThreads(1)
    rows=json.loads((a.output/'raw.json').read_text())['records'];s=json.loads((a.output/'summary.json').read_text())
    result={'routing':{},'latency_ms':{},'prediction_consistency':True}
    for mode in ('iid','spatial','persistent'):
        q=[r for r in rows if r['noise']==mode]
        result['routing'][mode]={'frames_used':dict(Counter(r['methods']['guarded4']['frames_used'] for r in q)),
            'reasons':dict(Counter(x for r in q for x in r['methods']['guarded4']['reasons']))}
    # 3 sensors x 3 modes x 4 pre-defined positions in sorted sequences.
    selected=[]
    for sensor in ('active_stereo','structured_light','tof'):
        for mode in ('iid','spatial','persistent'):
            q=[r for r in rows if r['sensor']==sensor and r['noise']==mode]
            selected.extend(q[i] for i in np.linspace(0,len(q)-1,4,dtype=int))
    times={m:[] for m in ('single','joint2','joint4','guarded4')};details=[]
    for r in selected:
        folder=Path(r['frame']['raw_path']).parent;m=json.loads((folder/'metadata.json').read_text())
        cal=replace(HeightCalibration.from_simulation(m),bottom_z_m=m['interior_bottom_z_m'])
        end=r['frame']['repeat'];fs=sorted([f for f in json.loads((folder/'frames.json').read_text())
            if f['variant']==r['noise'] and end-3<=f['repeat']<=end],key=lambda f:f['repeat'])
        raw=[np.load(f['raw_path']) for f in fs];stamp=[f['timestamp'] for f in fs]
        assert len(raw)==4
        # Disk IO is excluded; warm one current-frame call before timing.
        estimate_window(raw[-1:],cal,m['container_rim_z_m'],r['sensor'])
        d={'cluster':r['cluster'],'noise':r['noise'],'sensor':r['sensor'],'endpoint':end,'compute_ms':{}}
        for method,n in [('single',1),('joint2',2),('joint4',4),('guarded4',4)]:
            fn=estimate_guarded4 if method=='guarded4' else estimate_window
            start=time.perf_counter();out=fn(raw[-n:],cal,m['container_rim_z_m'],r['sensor'],timestamps=stamp[-n:])
            t=(time.perf_counter()-start)*1000;times[method].append(t);d['compute_ms'][method]=t
            expected=r['methods'][method]['statistics'];got=out['statistics']
            assert (expected is None)==(got is None)
            if expected is not None:assert abs(expected['mean_m']-got['mean_m'])<1e-10
        details.append(d)
    result['latency_ms']={m:dict(zip(('p50','p95','p99','max'),map(float,np.percentile(v,[50,95,99,100])))) for m,v in times.items()}
    result['latency_windows']=len(selected);result['latency_details']=details
    (a.output/'post-test-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='latency_details'},indent=2))
if __name__=='__main__':main()
