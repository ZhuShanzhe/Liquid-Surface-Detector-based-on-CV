#!/usr/bin/env python3
"""Aggregate frozen screening with paired-scene, pixel and thickness accounting."""
import argparse,csv,json
from pathlib import Path
import numpy as np
METHODS=['official_sensor_scale','official_dedup_sensor_scale','official_oracle_top_bottom','official_dedup_oracle_top_bottom','v9_1_set','v9_1_zero_raw']
def summarize(folder):
    rows=[]
    for method in METHODS:
        path=folder/(method+'.json')
        if not path.exists():continue
        frames=json.loads(path.read_text())
        for group in ['single','index_matched','weak','strong']:
            for thickness in [None,.002,.01,.03,.05]:
                rs=[r for r in frames if r['group']==group and (thickness is None or r['id']%4==[.002,.01,.03,.05].index(thickness))]
                if group=='single' and thickness is not None:continue
                for tol in ['fixed5mm','relative2pct']:
                    n=sum(r['sampled_pixels'] for r in rs);ss=[r['tolerances'][tol] for r in rs]
                    out=dict(method=method,group=group,thickness_mm=None if thickness is None else thickness*1000,tolerance=tol,frames=len(rs),pixels=n)
                    for k in ['inner_hits','top_hits','both_liquid_hits','all_truth','complete']:
                        out[k+'_pct']=100*sum(s.get(k,0) for s in ss)/n if all(k in s for s in ss) else None
                    out['surface_recall_pct']=100*sum(s['matched'] for s in ss)/sum(s['truth'] for s in ss)
                    out['candidate_count']=float(np.mean(sum([r['candidate_count'] for r in rs],[])))
                    for key in ['heuristic_thickness_errors_mm','oracle_pair_thickness_errors_mm','inner_nearest_errors_mm']:
                        values=sum([r[key] for r in rs],[])
                        out[key+'_mae']=float(np.mean(values)) if values else None
                        out[key+'_p95']=float(np.percentile(values,95)) if values else None
                        out[key+'_coverage_pct']=100*len(values)/n if group!='single' else None
                    rows.append(out)
    (folder/'summary.json').write_text(json.dumps(rows,indent=2))
    with (folder/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for r in rows:
        if r['thickness_mm'] is None and r['tolerance']=='fixed5mm' and r['group'] in ['weak','strong']:
            print(r['method'],r['group'],'inner',round(r['inner_hits_pct'],3),'both',r['both_liquid_hits_pct'],'all',round(r['all_truth_pct'],3),'heurMAE',round(r['heuristic_thickness_errors_mm_mae'] or 0,3),'oracleMAE',round(r['oracle_pair_thickness_errors_mm_mae'] or 0,3),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);summarize(p.parse_args().folder)
