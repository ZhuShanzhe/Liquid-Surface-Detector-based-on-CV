"""Paired audit of the opt-in dry-center / weak-mask fallback.

The old evaluation records define eligibility and scoring only. Neither those
records nor the oracle visible mask are passed to the inference route. Oracle
visibility is used solely for controlled deletion, as in V48.
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_liquid_height_v21 import Predictor
from evaluate_shared_empty_v45 import install
from evaluate_uniform_loss_v48 import LEVELS, delete_to_actual_loss
from liquid_depth.depth_failure_recovery_v49 import (prepare_dry_center_reference,
                                                     recover_depth_failure)


@torch.inference_mode()
def probability_batch(model, rgb_bgr, raws):
    color = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    colors = torch.from_numpy(np.stack([color] * len(raws)).transpose(0,3,1,2)).cuda()
    depths = torch.from_numpy(np.stack(raws)[:,None]).cuda()
    _, probability, _ = model(colors, depths)
    return probability[:,0].cpu().numpy()


def summarize(rows, name):
    n = len(rows)
    valid = [r for r in rows if r[name] is not None]
    ae = np.asarray([abs(r[name] - r['truth_m']) for r in valid])
    rel = np.asarray([abs(r[name] - r['truth_m']) / r['truth_m']
                      for r in valid if r['truth_m'] > 0])
    good = sum(abs(r[name] - r['truth_m']) <= max(.005, .02 * r['truth_m']) for r in valid)
    return {'frames':n, 'numeric':len(valid), 'coverage_pct':100*len(valid)/n if n else None,
            'all_frame_pass_pct':100*good/n if n else None,
            'conditional_mae_mm':1000*float(ae.mean()) if len(ae) else None,
            'conditional_p95_mm':1000*float(np.percentile(ae,95)) if len(ae) else None,
            'conditional_absrel_pct':100*float(rel.mean()) if len(rel) else None,
            'absolute_error_bins_all_pct':(100*np.histogram(
                ae,[0,.005,.01,.02,.05,.1,.2,np.inf])[0]/n).tolist() if n else None,
            'relative_error_bins_numeric_pct':(100*np.histogram(
                rel,[0,.005,.01,.02,.03,.05,.1,np.inf])[0]/len(rel)).tolist()
                if len(rel) else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wet-root',type=Path,required=True)
    p.add_argument('--dry-root',type=Path,required=True)
    p.add_argument('--old-records',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--cases',type=int,default=99)
    p.add_argument('--repeats',type=int,default=4)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    model = Predictor('v8')
    old = {(r['case'],r['repeat'],r['target_loss_pct']):r
           for r in map(json.loads,a.old_records.open())}
    rows = []
    for case in range(a.cases):
        if not any(key[0]==case for key in old):
            continue
        item = install(case,a.wet_root,a.dry_root,a.seed)
        reference = prepare_dry_center_reference(item['bundle'],item['dry'])
        folder = a.wet_root / f'{case*6+5:05d}'
        rgb = cv2.imread(str(folder/'rgb.png'))
        with np.load(folder/'geometry.npz') as z:
            oracle_visible = z['mask'].astype(bool)
        for frame in json.loads((folder/'frames.json').read_text()):
            repeat = frame['repeat']
            if repeat >= a.repeats or (case,repeat,0) not in old:
                continue
            base = np.load(frame['raw_path'],allow_pickle=False)
            samples = [base if level==0 else delete_to_actual_loss(
                base,oracle_visible,level,a.seed+case*10007+repeat*509)
                       for level in LEVELS]
            torch.cuda.synchronize()
            model_start = time.perf_counter()
            probabilities = probability_batch(model,rgb,samples)
            torch.cuda.synchronize()
            model_ms = 1000*(time.perf_counter()-model_start)/len(samples)
            for level,raw,probability in zip(LEVELS,samples,probabilities):
                previous = old[case,repeat,level]
                route_start = time.perf_counter()
                result = recover_depth_failure(item['bundle'],rgb,raw,probability,reference)
                route_ms = 1000*(time.perf_counter()-route_start)
                value = result['numeric_candidate_m']
                if value is not None:
                    value = float(value)
                if result['recovery_status']=='baseline_unchanged':
                    before = previous['prediction_m']
                    if before is None or abs(value-before)>1e-5:
                        raise AssertionError(f'baseline_replay_changed:{case}:{repeat}:{level}')
                rows.append({'case':case,'repeat':repeat,'level':level,
                             'sensor':previous['sensor'],'band':previous['band'],
                             'truth_m':previous['true_height_m'],
                             'old_m':previous['prediction_m'],'new_m':value,
                             'recovery_status':result['recovery_status'],
                             'baseline_reasons':result['baseline_reasons'],
                             'model_ms':model_ms,'route_ms':route_ms,
                             'candidate_center_disagreement_m':result.get(
                                 'candidate_center_disagreement_m')})
        print('case_done',case+1,'rows',len(rows),flush=True)
    with (a.output/'records.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row,allow_nan=False)+'\n')
    groups = defaultdict(list)
    for row in rows:
        groups[str(row['level'])].append(row)
        groups[f"{row['level']}/{row['sensor']}"].append(row)
        groups[f"{row['level']}/band_{row['band']}"].append(row)
    summary = {key:{'before':summarize(group,'old_m'),
                    'after':summarize(group,'new_m'),
                    'latency_ms':{'model_mean':float(np.mean([r['model_ms'] for r in group])),
                                  'route_mean':float(np.mean([r['route_ms'] for r in group])),
                                  'combined_p95':float(np.percentile(
                                      [r['model_ms']+r['route_ms'] for r in group],95))},
                    'recovery_status':dict(Counter(r['recovery_status'] for r in group))}
               for key,group in sorted(groups.items())}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (a.output/'protocol.json').write_text(json.dumps({
        'measurand':'liquid height above interior bottom',
        'levels_pct':list(LEVELS),'seed':a.seed,
        'old_records':str(a.old_records),'selected_frames_per_level':len(groups['0']),
        'oracle_uses':'controlled loss and offline score only',
        'route_inputs':'estimated empty vessel geometry, dry depth, current RGB-D, model probability',
        'route_status':'research-only; not independently hardware validated',
        'no_zero_snap_or_height_clipping':True},indent=2))
    print('COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    main()
