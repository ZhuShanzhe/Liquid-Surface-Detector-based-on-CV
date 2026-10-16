"""Post-hoc distribution audit of frozen predictions; no inference or tuning."""
import json
import math
import hashlib
from pathlib import Path
import numpy as np

OUT = Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/transparent-focus-v32-20261012')
rawpath = OUT / 'test-raw.json'
raw = json.loads(rawpath.read_text())['records']
policy = json.loads((OUT / 'frozen-policy.json').read_text())
before = [r for r in raw if r['stage'] == 'baseline_v31']
after = [r for r in raw if r['stage'] == policy['selection'][r['condition']]]
assert len(before) == len(after) == 8016, (len(before), len(after), sorted({r['stage'] for r in raw}))
assert {(r['config_id'], r['sample_id']) for r in before} == {(r['config_id'], r['sample_id']) for r in after}

def histogram(values, edges, denominator):
    result = []
    low = -math.inf
    for high in [*edges, math.inf]:
        n = int(sum(low < x <= high for x in values))
        result.append(dict(lower_exclusive=None if not math.isfinite(low) else low,
                           upper_inclusive=None if not math.isfinite(high) else high,
                           count=n, pct=100*n/denominator if denominator else None))
        low = high
    return result

def stats(rows):
    good = [r for r in rows if r['predicted_height_m'] is not None and math.isfinite(r['predicted_height_m'])]
    e = np.array([(r['predicted_height_m'] - r['true_height_m'])*1000 for r in good])
    rel = np.array([abs(r['predicted_height_m']-r['true_height_m'])/r['true_height_m']*100 for r in good if r['true_height_m']>0])
    positive_n = sum(r['true_height_m']>0 for r in rows)
    ratio = [abs(r['predicted_height_m']-r['true_height_m'])/max(.005,.02*r['true_height_m']) for r in good]
    result = dict(n=len(rows), finite=len(good), missing=len(rows)-len(good),
                  absolute_mm=histogram(abs(e), [2,5,10,20,50,100],len(rows)),
                  positive_n=positive_n, positive_finite=len(rel),
                  relative_pct=histogram(rel,[.5,1,2,3,5,10],positive_n),
                  tolerance_ratio=histogram(ratio,[.5,1,2,5,10],len(rows)))
    if len(e):
        result.update(mae_mm=float(abs(e).mean()),bias_mm=float(e.mean()),
                      rmse_mm=float(np.sqrt((e*e).mean())),
                      quantile_mm=dict(zip(['p50','p90','p95','p99','max'],np.percentile(abs(e),[50,90,95,99,100]).tolist())),
                      relative_quantile_pct=dict(zip(['p50','p90','p95','p99','max'],np.percentile(rel,[50,90,95,99,100]).tolist())) if len(rel) else None,
                      overestimate_pct=float(sum(e>0)/len(rows)*100),underestimate_pct=float(sum(e<0)/len(rows)*100))
    return result

def partition(rows):
    result = dict(all=rows,uniform=[r for r in rows if r['population']=='uniform'],
                  zero=[r for r in rows if r['true_height_m']==0],
                  near_zero=[r for r in rows if 0<r['true_height_m']<=.1],
                  low=[r for r in rows if 0<r['true_height_m']<1],
                  mid=[r for r in rows if 2<=r['true_height_m']<3])
    for i in range(9):
        result[f'range_{i}_{i+1}']=[r for r in rows if (0<r['true_height_m']<1 if i==0 else i<=r['true_height_m']<=9 if i==8 else i<=r['true_height_m']<i+1)]
    return result

result = dict(source_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest(),
              analysis_only=True,no_prediction_changes=True,
              bin_definition='lower < error <= upper; first bin includes zero; percentages use all eligible frames including missing',
              subsets={k:dict(before=stats(v),after=stats(partition(after)[k])) for k,v in partition(before).items()},
              materials={})
result['uniform_ranges']={k:dict(before=stats(v),after=stats(partition([r for r in after if r['population']=='uniform'])[k]))
                          for k,v in partition([r for r in before if r['population']=='uniform']).items()}
for m in sorted({r['material_regime'] for r in after}):
    b=[r for r in before if r['material_regime']==m];a=[r for r in after if r['material_regime']==m]
    result['materials'][m]={k:dict(before=stats(v),after=stats(partition(a)[k])) for k,v in partition(b).items()}
result['worst_after'] = sorted([r for r in after if r['predicted_height_m'] is not None],key=lambda r:abs(r['predicted_height_m']-r['true_height_m']),reverse=True)[:15]
for group in [result['subsets'],result['uniform_ranges'],*result['materials'].values()]:
    for item in group.values():
        for s in item.values():
            assert sum(b['count'] for b in s['absolute_mm'])+s['missing']==s['n']
            assert sum(b['count'] for b in s['relative_pct'])==s['positive_finite']
            assert sum(b['count'] for b in s['tolerance_ratio'])==s['finite']
path=OUT/'error-distribution.json'
path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
for label,item in list(result['subsets'].items())+[(m,v['uniform']) for m,v in result['materials'].items()]:
    print(label,json.dumps(item,ensure_ascii=False))
print('TAIL',json.dumps([{k:r[k] for k in ['config_id','sample_id','material_regime','true_height_m','predicted_height_m']} for r in result['worst_after']],ensure_ascii=False))
print('SAVED',path)
