import json
from pathlib import Path
from collections import Counter
import numpy as np

base=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/prior-reliability-v24b-20261011')
raw=json.loads((base/'test-raw.json').read_text())
routes=json.loads((base/'routed-test-raw.json').read_text())
def metrics(rows):
    good=[r for r in rows if r['predicted_height_m'] is not None]
    e=np.array([r['absolute_error_m'] for r in good]);h=np.array([r['true_height_m'] for r in good])
    return dict(n=len(rows),finite=len(good),mae_mm=round(float(e.mean()*1000),3) if len(e) else None,
        rel=round(float(np.mean(e/h)*100),3) if len(e) else None,passed=round(float(np.mean(e<=np.maximum(.005,.02*h))*100),2) if len(e) else None,
        p95_mm=round(float(np.percentile(e,95)*1000),3) if len(e) else None)
for family in ['active_stereo','structured_light']:
    for cond in ['transparent','translucent_mild']:
        for method in ['v23_3','v23_5','affine_3','affine_5','piecewise_5','pchip_5']:
            rs=[r for r in raw['records'] if r['family']==family and r['condition']==cond and r['method']==method]
            print(family,cond,method,metrics(rs))
        r=[r for r in routes if r['family']==family and r['condition']==cond and r['method']=='route_5']
        print('RISK FILTER',Counter(x['source'] for x in r),metrics([x for x in r if x['accepted']]))
    print('FAILURES',Counter(x.get('reason') for x in raw['fits'] if x['config'].startswith(family) and 'strong' not in x['config'] and x['method']=='v23_3' and not x['accepted']))
for family in ['active_stereo','structured_light']:
    for bottom in [.3,.8,2.,6.,9.8]:
        r=[r for r in raw['records'] if r['family']==family and r['condition']!='translucent_strong' and r['method']=='affine_5' and r['bottom_m']==bottom]
        print('AFFINE RANGE',family,bottom,metrics(r))
print('WORST AFFINE')
rs=[r for r in raw['records'] if r['condition']!='translucent_strong' and r['method']=='affine_5' and r['predicted_height_m'] is not None]
for r in sorted(rs,key=lambda r:r['absolute_error_m']/max(.005,.02*r['true_height_m']),reverse=True)[:12]:
    print({k:r[k] for k in ['config_id','sample_id','bottom_m','true_height_m','predicted_height_m','absolute_error_m']},r['diagnostics'])
