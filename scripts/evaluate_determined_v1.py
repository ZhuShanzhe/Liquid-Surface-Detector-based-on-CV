"""Replay the fixed facade; keep historical evidence cohorts separate."""
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from evaluate_shared_empty_v45 import install
from experiment_range_v31 import load
from experiment_sr_v33 import configs

from liquid_depth.determined_scene_meter import DeterminedSceneMeter

BASE = Path('/root/autodl-tmp')
REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OUT = BASE/'liquid-depth-artifacts/determined-all-scenes-v1-20261016'
ROOT = BASE/'liquid-depth-data/synthetic/adaptive-evidence-v2-confirmation-20261015'
OLD = BASE/'liquid-depth-artifacts/adaptive-route-v2-20261015/confirmation-with-real'
REF = BASE/'liquid-depth-artifacts/evaluation/transparent-sr-v33-20261012/test-raw.json'
FLOAT = BASE/'liquid-depth-artifacts/floating-joint-v9-20261016/independent-test'
SCENE = {'ordinary': 'ordinary_diffuse', 'depth_failure': 'partial_depth_loss'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False))


def metric(rows, key, peak=False):
    truth_key = 'highest_height_m' if peak else 'liquid_height_m'
    good = [r for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    e = np.asarray([1000*abs(r[key]-r['truth'][truth_key]) for r in good])
    relative = np.asarray([100*abs(r[key]-r['truth'][truth_key])/r['truth'][truth_key]
                           for r in good if r['truth'][truth_key] > 0])
    passing = sum(abs(r[key]-r['truth'][truth_key]) <= max(.05 if peak else .02,
                   (.05 if peak else .02)*r['truth'][truth_key]) for r in good)
    n = len(rows)
    return {'frames': n, 'numeric': len(good), 'numeric_coverage_pct': 100*len(good)/n,
            'all_frame_pass_pct': 100*passing/n,
            'mae_mm': float(e.mean()) if len(e) else None,
            'p95_mm': float(np.percentile(e, 95)) if len(e) else None,
            'max_mm': float(e.max()) if len(e) else None,
            'absolute_bins_pct_all_frames': (100*np.histogram(e, [0, 5, 20, 50, 100, np.inf])[0]/n).tolist(),
            'relative_bins_pct_nonzero_numeric': (100*np.histogram(relative, [0, 1, 2, 5, 10, np.inf])[0]/len(relative)).tolist() if len(relative) else [],
            'relative_frames': len(relative)}


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        if not r.get('scope_included', True):
            continue
        h = r['truth']['liquid_height_m']
        distance = r['camera_to_bottom_m']
        subset = 'far_shallow_wet' if distance >= 6 and 0 < h <= 1 else 'far_dry' if distance >= 6 and h == 0 else 'other'
        for part in ['all_frames', 'wet' if h > 0 else 'dry', subset]:
            for layer in ['all', 'sensor='+r.get('sensor', 'not_applicable'), 'band='+str(r['band'])]:
                groups[r['scene']+'|'+part+'|'+layer].append(r)
    result = {}
    for k, group in groups.items():
        elapsed = [r['algorithm_ms'] for r in group if r.get('algorithm_ms') is not None]
        result[k] = {'liquid': metric(group, 'liquid_depth_m'),
                     'highest': metric(group, 'highest_height_m', True) if any('highest_height_m' in r for r in group) else None,
                     'latency_ms': {'median': float(np.median(elapsed)), 'p95': float(np.percentile(elapsed, 95)),
                                    'max': max(elapsed)} if elapsed else None}
    return result


def opaque():
    prior = [json.loads(line) for line in (OLD/'records.jsonl').read_text().splitlines()]
    lookup = {(r['case'], r['repeat']): r for r in prior if r['mode'] == 'off'}
    controls = {(r['case'], r['repeat']): r for r in prior if r['mode'] == 'geometric_peak_control'}
    rows, audits = [], []
    with (OUT/'opaque-records.partial.jsonl').open('w') as stream:
        for slot, case in enumerate(sorted({r['case'] for r in lookup.values()})):
            item = install(case, ROOT, ROOT/'dry', 202610154229, max_annotation_attempts=6, source_slot_stride=1)
            audits.append(item['audit'])
            meter = DeterminedSceneMeter.from_empty_geometry(item['bundle'], item['dry'])
            folder = ROOT/f'{case:05d}'
            scene = SCENE.get(item['meta']['scenario'], item['meta']['scenario'])
            rgb = cv2.imread(str(folder/'rgb.png')) if scene == 'floating_objects' else None
            for frame in json.loads((folder/'frames.json').read_text()):
                raw = np.load(frame['raw_path'])
                start = time.perf_counter()
                value = meter.measure(scene, depth_m=raw, rgb_bgr=rgb, visual_enabled=scene == 'floating_objects')
                elapsed = 1000*(time.perf_counter()-start)
                old = lookup[case, frame['repeat']]
                assert value['liquid_depth_m'] == old['liquid_depth_m']
                assert not value['liquid_visual_guidance_used']
                if scene != 'floating_objects':
                    assert value['highest_height_m'] == controls[case, frame['repeat']]['highest_height_m']
                    assert not value['visual_module_executed']
                loss = old['actual_liquid_missing_fraction_offline']
                included = scene != 'partial_depth_loss' or loss is None or loss <= .5
                row = {**old, 'scene': scene, 'mode': 'determined_v1', 'cohort': 'opaque_202610154229',
                       'scope_included': included, 'liquid_depth_m': value['liquid_depth_m'],
                       'highest_height_m': value['highest_height_m'], 'algorithm_ms': elapsed,
                       'diagnostics': value['diagnostics']}
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
            if slot % 20 == 0:
                print('OPAQUE_REPLAY', slot+1, 198, flush=True)
    (OUT/'opaque-records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    dump('opaque-calibration-audit.json', audits)
    dump('opaque-summary.json', summarize(rows))
    return rows


def transparent():
    cached = json.loads(REF.read_text())['records']
    selected = [r for r in cached if r['stage'] == ('baseline_v32' if r['condition'] == 'transparent' else 'midpoint_dense')]
    ix = {(r['config_id'], r['sample_id']): r for r in selected}
    assert len(ix) == len(selected) == 8016
    replay = []
    examples = []
    for condition in ['transparent', 'translucent_mild']:
        pool = sorted((c for c in configs('test') if c['condition'] == condition), key=lambda c: c['id'])
        for j in np.rint(np.linspace(0, len(pool)-1, 6)).astype(int):
            config = pool[j]
            images, masks, cal, heights = load(config)
            scene = 'homogeneous_transparent' if condition == 'transparent' else 'homogeneous_translucent'
            prior = [{ch: images[i, ch] for ch in ('left', 'right')} for i in range(5)]
            meter = DeterminedSceneMeter.from_five_priors(cal, masks, prior, heights, scene=scene)
            for index in np.rint(np.linspace(0, len(config['tests'])-1, 24)).astype(int):
                start = time.perf_counter()
                result = meter.measure(scene, stereo_images={ch: images[index+5, ch] for ch in ('left', 'right')}, visual_enabled=True)
                elapsed = 1000*(time.perf_counter()-start)
                old = ix[config['id'], int(index)]
                assert abs(result['liquid_depth_m']-old['predicted_height_m']) < 1e-7
                replay.append({'scene': scene, 'config_id': config['id'], 'sample_id': int(index),
                    'band': 'dry' if old['true_height_m'] == 0 else int(old['true_height_m']),
                    'sensor': 'stereo_IR_image_proxy', 'camera_to_bottom_m': old['bottom_m'],
                    'truth': {'liquid_height_m': old['true_height_m']}, 'liquid_depth_m': result['liquid_depth_m'],
                    'algorithm_ms': elapsed, 'cohort': 'five_prior_live_replay',
                    'difference_to_frozen_m': result['liquid_depth_m']-old['predicted_height_m']})
            if not any(e['scene'] == scene for e in examples):
                examples.append({'scene': scene, 'config': config})
            print('REFRACTION_REPLAY', config['id'], flush=True)
    full = [{'scene': 'homogeneous_transparent' if r['condition'] == 'transparent' else 'homogeneous_translucent',
             'band': 'dry' if r['true_height_m'] == 0 else int(r['true_height_m']),
             'sensor': 'stereo_IR_image_proxy', 'camera_to_bottom_m': r['bottom_m'],
             'truth': {'liquid_height_m': r['true_height_m']}, 'liquid_depth_m': r['predicted_height_m'],
             'algorithm_ms': None, 'cohort': 'frozen_refraction_8016_recount',
             'config_id': r['config_id'], 'sample_id': r['sample_id'], 'method': r['stage']}
            for r in selected]
    dump('refraction-live-records.json', replay)
    dump('refraction-live-summary.json', summarize(replay))
    dump('refraction-frozen-records.json', full)
    dump('refraction-frozen-summary.json', summarize(full))
    dump('refraction-example-configs.json', examples)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    sources = list((REPO/'src/liquid_depth').glob('*.py')) + [REPO/'scripts/evaluate_determined_v1.py',
        REPO/'scripts/measure_determined.py', REPO/'configs/determined_scene_routes_v1.json']
    dump('protocol.json', {'source_sha256': {str(p): sha(p) for p in sources},
        'opaque_source': str(ROOT), 'opaque_seed': 202610154229, 'opaque_installations': 198,
        'refraction_source': str(REF), 'refraction_source_sha256': sha(REF),
        'same_existing_holdouts_not_new_blinded_capture': True,
        'liquid_tolerance': 'max(20mm,2% true liquid height)', 'highest_tolerance': 'max(50mm,5% true highest height)',
        'no_training_or_algorithm_tuning': True, 'no_zero_snap_or_output_clipping': True,
        'timing_includes_first_call_loads_for_floating': True})
    opaque()
    transparent()
    rows = [json.loads(s) for s in (FLOAT/'records.jsonl').read_text().splitlines()]
    selected = [{**r, 'scene': 'floating_objects', 'cohort': 'dedicated_float_202610164913'}
                for r in rows if r['mode'] == 'v8']
    dump('dedicated-floating-v8-summary.json', summarize(selected))
    dump('dedicated-floating-v8-records.json', selected)
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert all(sha(Path(p)) == h for p, h in protocol['source_sha256'].items())
    print('DETERMINED_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
