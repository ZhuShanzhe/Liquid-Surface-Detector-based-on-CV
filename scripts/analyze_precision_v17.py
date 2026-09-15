#!/usr/bin/env python3
import csv,json
from pathlib import Path
import numpy as np
OUT=Path('/root/autodl-tmp/liquid-depth-artifacts/training/precision-v17-20260916')
def main():
    r=json.loads((OUT/'routing-diagnostic.json').read_text())['r4']['frames']
    from train_observable_v16 import Observations,DEV
    ds=Observations(DEV,'val',0)
    assert len(ds.rows)==len(r)
    frames=[];bins={}
    for f,row in zip(r,ds.rows):
        truth=np.load(row['target_depth_path']);mask=(np.load(row['mask_path'])>0)&(truth>=.1)&(truth<=10)
        z=float(np.median(truth[mask])) if mask.any() else 0
        band=('0.1-0.3m','0.3-1m','1-3m','3-10m')[int(np.searchsorted([.3,1,3],z))]
        a=f['all'];item=dict(sequence_id=f['sequence_id'],material=f['scenario'],median_depth_m=z,band=band,mae_mm=1000*a['abs_sum']/max(a['n'],1),success=a['good']/max(a['possible'],1),n=a['n'],abs_sum=a['abs_sum'],good=a['good'],possible=a['possible'])
        frames.append(item);bins.setdefault(band,[]).append(item)
    summary={k:dict(frames=len(v),mae_mm=1000*sum(f['abs_sum'] for f in v)/max(sum(f['n'] for f in v),1),success=sum(f['good'] for f in v)/max(sum(f['possible'] for f in v),1)) for k,v in bins.items()}
    result={'baseline':'V16 gated on development fold0','range_summary':summary,'worst_frames':sorted(frames,key=lambda f:f['mae_mm'],reverse=True)[:12],'frame_median_mae_mm':float(np.median([f['mae_mm'] for f in frames])),'frame_p90_mae_mm':float(np.percentile([f['mae_mm'] for f in frames],90)),'labels_used_only_for_error_analysis':True}
    (OUT/'error-diagnosis.json').write_text(json.dumps(result,indent=2));print(summary)
if __name__=='__main__':main()
