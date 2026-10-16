"""Independently verify every published stratum from the shipped slim predictions.

No raw images, model inference, fonts, ReportLab, or server paths are required.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import numpy as np


def metric(rows,highest=False):
    key='highest_height_m' if highest else 'liquid_depth_m'
    truth='highest_height_m' if highest else 'liquid_height_m'
    good=[r for r in rows if r.get(key) is not None and math.isfinite(r[key]) and truth in r['truth']]
    error=np.array([abs(r[key]-r['truth'][truth]) for r in good])
    relative=np.array([abs(r[key]-r['truth'][truth])/r['truth'][truth] for r in good if r['truth'][truth]>0])
    passed=sum(abs(r[key]-r['truth'][truth])<=max(.05,.05*r['truth'][truth])
               if highest or r['scene']=='uneven_surface'
               else abs(r[key]-r['truth'][truth])<=max(.02,.02*r['truth'][truth]) for r in good)
    n=len(rows)
    return dict(n=n,numeric=len(good),coverage=100*len(good)/n if n else None,passed=int(passed),
                pass_pct=100*passed/n if n else None,no_output=n-len(good),relative_n=len(relative),
                mae_mm=float(error.mean()*1000) if len(error) else None,
                p95_mm=float(np.percentile(error,95)*1000) if len(error) else None,
                max_mm=float(error.max()*1000) if len(error) else None,
                absrel_pct=float(relative.mean()*100) if len(relative) else None,
                absolute_bins_pct=(np.histogram(error,[0,.005,.02,.05,.1,np.inf])[0]*100/n).tolist() if n else [],
                relative_bins_pct=(np.histogram(relative,[0,.01,.02,.05,.1,np.inf])[0]*100/len(relative)).tolist() if len(relative) else [])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parents[1]/'reports/evidence')
    args=parser.parse_args()
    with gzip.open(args.evidence/'current_predictions_20261016.jsonl.gz','rt',encoding='utf-8') as f:
        rows=[json.loads(line) for line in f]
    expected=json.loads((args.evidence/'scene_metrics_20261016.json').read_text(encoding='utf-8'))
    counts=0
    for key,values in expected.items():
        scene,sensor,level,loss,depth=key.split('|')
        subset=[r for r in rows if r['scene']==scene and (sensor=='all' or r['sensor']==sensor)
                and (level=='all' or r['level']==level) and (loss=='all' or r['loss_bin']==loss)]
        if depth=='far_shallow':subset=[r for r in subset if r['camera_to_bottom_m']>=6 and 0<r['truth']['liquid_height_m']<=1]
        elif depth=='far_dry':subset=[r for r in subset if r['camera_to_bottom_m']>=6 and r['truth']['liquid_height_m']==0]
        elif depth!='all':subset=[r for r in subset if r['depth_bin']==depth]
        for quantity,fields in values.items():
            actual=metric(subset,quantity=='highest')
            for field,want in fields.items():
                got=actual[field]
                if want is None:assert got is None,(key,field)
                else:assert np.allclose(got,want,rtol=1e-12,atol=1e-10),(key,quantity,field,got,want)
            counts+=1
    print(json.dumps(dict(frames=len(rows),strata=len(expected),quantity_groups=counts,all_metrics_match=True)))


if __name__=='__main__':main()
