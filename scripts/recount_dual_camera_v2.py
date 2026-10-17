"""Portable verification of published dual-camera metrics (NumPy only)."""
import argparse,gzip,json
from pathlib import Path
import numpy as np


def independent_metrics(rows,peak):
    key='highest_height_m' if peak else 'liquid_depth_m';truth='highest_height_m' if peak else 'liquid_height_m'
    valid=[r for r in rows if r.get(key) is not None and np.isfinite(r[key]) and truth in r['truth']]
    h=np.array([r['truth'][truth] for r in valid]);p=np.array([r[key] for r in valid]);e=abs(p-h)
    rel=e[h>0]/h[h>0];n=len(rows)
    relaxed=peak or (rows and rows[0]['scene']=='uneven_surface')
    floor,ratio=(.05,.05) if relaxed else (.02,.02)
    def histogram(values,edges,denom):
        return [100*int(np.count_nonzero((values>=lo)&(values<hi)))/denom for lo,hi in zip(edges[:-1],edges[1:])] if denom else []
    return dict(n=n,installations=len(set(r['cluster'] for r in rows)),numeric=len(valid),no_output=n-len(valid),
        numeric_coverage_pct=100*len(valid)/n if n else None,pass_pct=100*np.count_nonzero(e<=np.maximum(floor,ratio*h))/n if n else None,
        mae_mm=float(np.mean(e)*1000) if len(e) else None,p95_mm=float(np.quantile(e,.95)*1000) if len(e) else None,
        max_mm=float(max(e)*1000) if len(e) else None,absrel_pct=float(np.mean(rel)*100) if len(rel) else None,
        absolute_bins_pct=histogram(e,[0,.005,.02,.05,.1,np.inf],n),relative_bins_pct=histogram(rel,[0,.01,.02,.05,.1,np.inf],len(rel)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parents[1]/'reports/evidence/dual-camera-v2-20261017');a=p.parse_args()
    with gzip.open(a.evidence/'current-dual-predictions.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(s) for s in f]
    expected=json.loads((a.evidence/'metrics.json').read_text());count=0
    for key,values in expected.items():
        scene,sensor,level,part=key.split('|')
        selected=[r for r in rows if r['scene']==scene and (sensor=='all' or r['sensor']==sensor) and (level=='all' or r['level']==level)]
        if part=='far_shallow':selected=[r for r in selected if r['camera_to_bottom_m']>=6 and 0<r['truth']['liquid_height_m']<=1]
        for name,fields in values.items():
            actual=independent_metrics(selected,name=='highest')
            for k in ('n','installations','numeric','no_output','numeric_coverage_pct','pass_pct','mae_mm','p95_mm','max_mm','absrel_pct','absolute_bins_pct','relative_bins_pct'):
                assert (actual[k] is None and fields[k] is None) or (actual[k] is not None and fields[k] is not None and np.allclose(actual[k],fields[k],rtol=1e-12,atol=1e-10)),(key,name,k)
            count+=1
    print(json.dumps(dict(frames=len(rows),statistic_cells=len(expected),quantity_groups=count,all_match=True)))


if __name__=='__main__':main()
