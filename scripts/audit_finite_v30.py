"""Post-freeze evaluator-only counterfactual. Never a deployable pipeline."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from experiment_finite_v30 import OUT,DATA
from experiment_transparent_v27 import rows,summary,dump
from liquid_depth import finite_relative_v30 as model
from liquid_depth.refraction_images_v23 import inverse_bottom


def one(c):
    cal=json.loads((__import__('pathlib').Path(c['priors'][0]['folder'])/'calibration.json').read_text())
    hs=[];index=None
    for cap in c['priors']:
        with np.load(cap['evaluator_truth']) as f:hs.append(float(f['height_m']));index=float(f['index'])
    xx,yy=np.meshgrid(np.linspace(-.30,.30,7)*c['container_width_m'],
        np.linspace(-.24,.24,6)*c['container_width_m']);world=np.c_[xx.ravel(),yy.ravel()]
    def pair(a,h):
        p=inverse_bottom(world,hs[a],index,cal['bottom_m']);q=inverse_bottom(world,h,index,cal['bottom_m'])
        return dict(anchor_q=p,current_q=q,weights=np.ones(len(p)),regions=len(p),channel='left')
    pairs=[(a,b,pair(a,hs[b])) for a in range(5) for b in range(a+1,5)]
    labels=[cap['known_liquid_height_m'] for cap in c['priors']]
    prior=model.fit(pairs,labels,cal,c['condition']);ps=[]
    for cap in c['tests']:
        with np.load(cap['evaluator_truth']) as f:h=float(f['height_m'])
        ps.append(model.infer([(a,pair(a,h)) for a in range(5)],prior,cal))
    return rows(c,ps,'evaluator_perfect_correspondences','empty_four')


if __name__=='__main__':
    frozen=json.loads((OUT/'frozen-source.json').read_text())
    assert (OUT/'progress.json').exists() and json.loads((OUT/'progress.json').read_text())['stage']=='complete'
    cs=json.loads((DATA/'test/manifest.json').read_text())['configs']
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,cs))
    records=[r for group in results for r in group]
    dump(OUT/'evaluator-only-perfect-correspondences.json',dict(records=records,summary=summary(records),
        uses_hidden_height_and_index=True,deployable=False,selection_changed=False,
        purpose='Perfect-ray counterfactual with same noisy known liquid-height labels; shared forward/inverse assumptions.'))
    print(summary(records),flush=True)
