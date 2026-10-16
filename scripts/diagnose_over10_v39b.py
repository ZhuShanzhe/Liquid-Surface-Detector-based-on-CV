"""Read-only metrology diagnostics and scientific figures, no model changes."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation')
OUT=ROOT/'ordinary-v39b-over10-analysis-20261012'
OUT.mkdir(exist_ok=True)

def load(which):
    folder=ROOT/f'ordinary-v39b-{which}-20261012'
    p=folder/'records.jsonl'
    rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else json.loads((folder/'records.json').read_text())
    return [r for r in rows if r['method']=='v39_bias_noisy_echo']

def summary(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[(r['regime'],r['cluster'])].append(r)
    cases=[]
    for (regime,cluster),rs in grouped.items():
        e=np.array([abs(r['predicted_height_m']-r['true_height_m'])*1000 for r in rs])
        cases.append(dict(regime=regime,cluster=cluster,case=rs[0]['case'],group=rs[0]['group'],
            true_height_m=rs[0]['true_height_m'],distance_m=rs[0]['camera_to_surface_m'],mae_mm=float(e.mean()),
            maximum_mm=float(e.max()),error_mm=e.tolist(),over10_frames=int((e>=10).sum())))
    counts=[]
    for regime in ('nominal','adverse'):
      for group in sorted({r['group'] for r in rows}):
        rr=[c for c in cases if c['regime']==regime and c['group']==group]
        counts.append(dict(regime=regime,group=group,cases=len(rr),mae_over10=sum(c['mae_mm']>=10 for c in rr),
            frames_over10=sum(c['over10_frames'] for c in rr),frames=sum(len(c['error_mm']) for c in rr)))
    return dict(counts=counts,top_cases=sorted(cases,key=lambda x:x['mae_mm'],reverse=True)[:20],all_cases=cases)

if __name__=='__main__':
    s={key:summary(load(key)) for key in ('test','replay')}
    (OUT/'error-populations.json').write_text(json.dumps(s,indent=2))
    for key,ss in s.items():
        print(key,json.dumps(dict(counts=ss['counts'],top_cases=ss['top_cases'][:10]),indent=2))
