#!/usr/bin/env python3
"""Summarize frozen V13 experiments without selecting a test-time threshold."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path('/root/autodl-tmp/liquid-depth-artifacts/training/candidate-ranker-v13-20260915')
SCENES = ('all','transparent','translucent','multilayer','compound')


def read(path):
    return json.loads(path.read_text())


def paired_interval(candidate, reference, scenario, repetitions=2000):
    a = {f['rgb_path']: f for f in candidate['frames'] if scenario=='all' or f['scenario']==scenario}
    b = {f['rgb_path']: f for f in reference['frames'] if scenario=='all' or f['scenario']==scenario}
    if a.keys()!=b.keys():
        raise ValueError('Unpaired frames')
    grouped = {}
    for key, f in a.items():
        r = b[key]
        if f['n'] != r['n'] or f['oracle_good'] != r['oracle_good']:
            raise ValueError('Frozen candidate pool or valid pixel count changed')
        sums = grouped.setdefault(f['sequence_id'], np.zeros(3))
        sums += [f['n'], f['top1_good'], r['top1_good']]
    arr = np.asarray(list(grouped.values()))
    rng = np.random.default_rng(20260919)
    samples = rng.integers(0,len(arr),(repetitions,len(arr)))
    counts = arr[samples].sum(1)
    delta = (counts[:,1]-counts[:,2])/np.maximum(counts[:,0],1)
    return {'sequences': len(arr), 'top1_delta_pp': 100*(arr[:,1].sum()-arr[:,2].sum())/arr[:,0].sum(),
            'ci95_pp': (100*np.quantile(delta,[.025,.975])).tolist(),
            'scope': 'paired sequence bootstrap vs V12; exploratory multiple comparisons, not industrial qualification'}


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=ROOT)
    ROOT = parser.parse_args().run_dir
    frozen = read(ROOT/'frozen_selection.json')
    out = {'scope': 'axial camera depth, not liquid height', 'datasets': {}}
    for dataset in ('legacy','fresh'):
        reference = read(ROOT/f'{dataset}_v12_False.json')
        baseline = read(ROOT/f'{dataset}_baseline_False.json')
        result = {}
        for name, report, threshold in [('V9.1', baseline,.5),('V12',reference,.4)]:
            result[name] = {s: {'top1': report['summary'][s]['top1_recall_all_valid'],
                              'oracle': report['summary'][s]['oracle_recall'],
                              'ece': report['summary'][s]['confidence_ece'],
                              **report['summary'][s]['selected'][str(threshold)]} for s in SCENES}
        for arm, decision in frozen['arms'].items():
            clean, stress = [read(ROOT/arm/f'{dataset}_{flag}.json') for flag in (False,True)]
            t = decision['threshold'] if decision['threshold'] is not None else frozen['diagnostic_threshold_for_unqualified']
            result[arm] = {'qualified': decision['threshold'] is not None, 'threshold': t,
                           'best_epoch': decision['best_epoch'],
                           'groups': {s: {'top1': clean['summary'][s]['top1_recall_all_valid'],
                            'oracle': clean['summary'][s]['oracle_recall'],
                            'ece': clean['summary'][s]['confidence_ece'],
                            'ece_stress': stress['summary'][s]['confidence_ece'],
                            'none_false_accept': clean['summary'][s]['accepted_when_no_good_candidate'][str(t)],
                            'paired_vs_v12': paired_interval(clean, reference, s),
                            **clean['summary'][s]['selected'][str(t)]} for s in SCENES}}
        out['datasets'][dataset] = result
    (ROOT/'comparison.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    for dataset, groups in out['datasets'].items():
        print(dataset)
        for name, record in groups.items():
            all_group = record['all'] if name in ('V9.1','V12') else record['groups']['all']
            print(name, {k:round(all_group[k],5) if all_group[k] is not None else None for k in
                         ('top1','ece','depth_mae_mm','pixel_tolerance_pass','valid_pixel_coverage')},
                         record.get('qualified'), record.get('best_epoch'))


if __name__=='__main__':
    main()
