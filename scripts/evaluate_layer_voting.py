#!/usr/bin/env python3
"""Frozen V9.1 single-frame voting ablation; labels used only by evaluator.

Reports camera-ray surface depth, NOT calibrated liquid height. Manifest test
split is held out from this experiment's tuning, but is an existing project set.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from liquid_depth.models.universal import UniversalLiquidSurfaceNet
from liquid_depth.training.universal_dataset import UniversalMultiTaskDataset
from liquid_depth.models.layer_voting import (
    SAMPLE_NAMES, sample_prediction, make_ballot, fuse_ballots, predict_liquid_vote,
)


def load_model(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    keys = inspect.signature(UniversalLiquidSurfaceNet).parameters
    options = {k: v for k, v in state.items() if k in keys}
    model = UniversalLiquidSurfaceNet(**options).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    return model, state


def pixel_stats(depth, emitted, truth, valid):
    accepted = emitted & valid
    errors = (depth - truth).abs()
    tol = torch.maximum(truth * .02, torch.full_like(truth, .005))
    n = int(accepted.sum())
    signed = (depth - truth)[accepted]
    # Fixed predicted-support requirement; no label-derived runtime acceptance.
    has_frame = int(emitted.sum()) >= 64
    evaluable = has_frame and n >= 64
    proxy = float(signed.median().abs()) if evaluable else None
    reference = float(truth[accepted].median()) if evaluable else None
    return {
        'possible': int(valid.sum()), 'emitted': int(emitted.sum()), 'n': n,
        'good': int(((errors <= tol) & accepted).sum()),
        'abs_sum_m': float(errors[accepted].sum()),
        'rel_sum': float((errors / truth.clamp_min(1e-6))[accepted].sum()),
        'frame_emitted': has_frame, 'frame_evaluable': evaluable,
        'frame_any': bool(emitted.any()),
        'proxy_abs_m': proxy,
        'proxy_good': proxy <= max(.005, .02 * reference) if evaluable else None,
    }


def summarize(rows):
    sums = {k: sum(r[k] for r in rows) for k in ('possible','emitted','n','good','abs_sum_m','rel_sum',
                                               'frame_emitted','frame_evaluable','frame_any')}
    n, possible = sums['n'], sums['possible']
    proxies = [r['proxy_abs_m'] for r in rows if r['proxy_abs_m'] is not None]
    return {
        'frames': len(rows), 'evaluated_pixels': n,
        'depth_mae_mm': 1000*sums['abs_sum_m']/n if n else None,
        'depth_abs_rel': sums['rel_sum']/n if n else None,
        'pixel_tolerance_pass': sums['good']/n if n else None,
        'pixel_exceedance_rate': 1-sums['good']/n if n else None,
        'valid_pixel_coverage': n/possible if possible else None,
        'within_tolerance_coverage': sums['good']/possible if possible else None,
        'non_target_output_fraction': 1-n/sums['emitted'] if sums['emitted'] else None,
        'frame_coverage_64_predicted_points': sums['frame_emitted']/len(rows),
        'frame_coverage_any_pixel_legacy': sums['frame_any']/len(rows),
        'evaluable_frame_rate': sums['frame_evaluable']/sums['frame_emitted'] if sums['frame_emitted'] else None,
        'signed_depth_median_proxy_mae_mm': float(np.mean(proxies)*1000) if proxies else None,
        'signed_depth_median_proxy_p95_mm': float(np.percentile(proxies,95)*1000) if proxies else None,
        'signed_depth_median_proxy_pass': sum(r['proxy_good'] is True for r in rows)/len(proxies) if proxies else None,
    }


def audit_layers(prediction, target):
    truth = target['layer_depths_m']
    valid = target['layer_valid'] > 0
    depth = prediction['layer_depths_m']
    presence = prediction['layer_presence_logits'].sigmoid()
    hard = presence >= .5
    count = valid.sum(1)
    mask = count > 0
    hard_count, soft_count = hard.sum(1), presence.sum(1)
    distance = (depth[:, :, None] - truth[:, None]).abs()
    target_tol = torch.maximum(truth*.02, torch.full_like(truth,.005))
    matched = ((distance <= target_tol[:,None]) & valid[:,None]).any(2)
    # Count close predicted components once, sorted in metric depth.
    order = depth.argsort(1)
    ds = depth.gather(1,order); hs = hard.gather(1,order)
    last = torch.full_like(ds[:,:1], -100.)
    dedup = torch.zeros_like(count)
    for k in range(ds.shape[1]):
        current = ds[:,k:k+1]
        tol = torch.maximum(current*.01, torch.full_like(current,.005))
        keep = hs[:,k:k+1] & ((current-last).abs() > tol)
        dedup += keep[:,0]
        last = torch.where(keep, current, last)
    return {
        'pixels': int(mask.sum()), 'gt_count_sum': float(count[mask].sum()),
        'soft_count_sum': float(soft_count[mask].sum()),
        'hard_count_sum': int(hard_count[mask].sum()),
        'hard_over': int(((hard_count > count) & mask).sum()),
        'hard_under': int(((hard_count < count) & mask).sum()),
        'hard_equal': int(((hard_count == count) & mask).sum()),
        'dedup_over': int(((dedup > count) & mask).sum()),
        'dedup_under': int(((dedup < count) & mask).sum()),
        'dedup_count_sum': int(dedup[mask].sum()),
        'active_components': int((hard & mask[:,None]).sum()),
        'unsupported_components': int((hard & ~matched & mask[:,None]).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['val','test'], default='val')
    parser.add_argument('--limit-per-scenario', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.manual_seed(20260914)
    model, state = load_model(args.checkpoint)
    dataset = UniversalMultiTaskDataset(args.manifest, args.split,
        tuple(state['image_size']), augment=False,
        min_depth_m=state['min_depth_m'], max_depth_m=state['max_depth_m'])
    scope = set(state['route_scope'])
    dataset.rows = [r for r in dataset.rows if r['scenario'] in scope and r.get('layer_depths_path')]
    if args.limit_per_scenario:
        seen = defaultdict(int); selected = []
        for row in dataset.rows:
            if seen[row['scenario']] < args.limit_per_scenario:
                selected.append(row); seen[row['scenario']] += 1
        dataset.rows = selected
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, shuffle=False)
    results = []; audits = []; offset = 0
    start = time.monotonic()
    with torch.inference_mode():
        for inputs, target in loader:
            inputs = inputs.cuda()
            target = {k: v.cuda() for k,v in target.items()}
            predictions = [sample_prediction(model, inputs, i) for i in range(7)]
            ballots = [make_ballot(p) for p in predictions]
            methods = {'single': ballots[0]}
            for samples in (3,5,7):
                methods[f'plurality{samples}'] = fuse_ballots(ballots[:samples], predictions[0]['depth_m'], min_fraction=0)
                methods[f'majority{samples}'] = fuse_ballots(ballots[:samples], predictions[0]['depth_m'])
            for samples in (5,7):
                methods[f'weighted{samples}'] = fuse_ballots(ballots[:samples], predictions[0]['depth_m'], weighted=True)
            audits.append(audit_layers(predictions[0], target))
            for b in range(inputs.shape[0]):
                meta = dataset.rows[offset+b]
                entry = {'sequence_id': meta['sequence_id'], 'rgb_path': meta['rgb_path'],
                         'scenario': meta['scenario'], 'sensor_model': meta.get('sensor_model'),
                         'methods': {}, 'common_with_single': {}}
                truth, valid = target['depth_m'][b], target['valid'][b] > 0
                base = methods['single']
                for name, m in methods.items():
                    entry['methods'][name] = pixel_stats(m['depth_m'][b], m['accepted'][b], truth, valid)
                    common = m['accepted'][b] & base['accepted'][b]
                    entry['common_with_single'][name] = {
                        'single': pixel_stats(base['depth_m'][b], common, truth, valid),
                        'vote': pixel_stats(m['depth_m'][b], common, truth, valid),
                    }
                results.append(entry)
            offset += inputs.shape[0]
            if offset % 40 == 0 or offset == len(dataset):
                print(json.dumps({'done':offset,'total':len(dataset),
                                  'elapsed_s':round(time.monotonic()-start,1)}), flush=True)
    names = list(results[0]['methods'])
    groups = {}
    for scene in ['all'] + sorted(scope):
        selected = [r for r in results if scene == 'all' or r['scenario'] == scene]
        if selected:
            groups[scene] = {name: summarize([r['methods'][name] for r in selected]) for name in names}
    common_summary = {name:{side:summarize([r['common_with_single'][name][side] for r in results])
                           for side in ('single','vote')} for name in names if name != 'single'}
    audit = {k:sum(a[k] for a in audits) for k in audits[0]}
    n = audit['pixels']
    audit['mean_gt_count'] = audit['gt_count_sum']/n
    audit['mean_soft_count'] = audit['soft_count_sum']/n
    audit['mean_hard_count'] = audit['hard_count_sum']/n
    for key in ('hard_over','hard_under','hard_equal','dedup_over','dedup_under'):
        audit[key+'_fraction'] = audit[key]/n
    audit['unsupported_active_fraction'] = audit['unsupported_components']/max(1,audit['active_components'])
    latency = {}
    if args.benchmark:
        x = inputs[:1]
        for name, samples, weighted, fraction in [('single',1,False,0),('majority3',3,False,.6),
                ('majority5',5,False,.6),('weighted5',5,True,.6),('weighted7',7,True,.6)]:
            durations = []
            for rep in range(25):
                torch.cuda.synchronize(); t = time.perf_counter()
                if samples == 1:
                    with torch.inference_mode(): make_ballot(sample_prediction(model,x,0))
                else:
                    predict_liquid_vote(model,x,samples=samples,weighted=weighted,min_fraction=fraction)
                torch.cuda.synchronize()
                if rep >= 5: durations.append((time.perf_counter()-t)*1000)
            latency[name] = {'p50_ms':float(np.median(durations)), 'p95_ms':float(np.percentile(durations,95)),
                             'max_ms':max(durations), 'repetitions':20, 'warmup':5}
    report = {'split':args.split,'sample_count':len(dataset), 'limit_per_scenario':args.limit_per_scenario,
              'checkpoint':str(args.checkpoint), 'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'manifest':str(args.manifest), 'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'protocol':{'samples':SAMPLE_NAMES,'selector_confidence':.5,'selector_relative_tolerance':.008,'cluster_absolute_m':.005,
                          'cluster_relative':.01,'majority_fraction':.6,'min_margin':1,
                          'metric':'camera-ray surface depth, not calibrated liquid height',
                          'proxy':'absolute median signed camera-depth residual, evaluation-only',
                          'latency_scope':'resident GPU model, TTA and voting; excludes camera IO, plane fitting and UI',
                          'dependence':'TTA ballots correlated; agreement is not probability of correctness'},
              'summary':groups,'common_support':common_summary,'layer_count_audit':audit,
              'latency':latency,'frames':results,'runtime_s':time.monotonic()-start,
              'production_promoted':False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'audit':audit,'overall':groups['all'],'latency':latency}),flush=True)


if __name__ == '__main__':
    main()
