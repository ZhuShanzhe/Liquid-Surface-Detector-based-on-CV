"""Development-only policy comparison; no held-out result is used for selection."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_illumination_v2 import score

ROOT = Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation')


def load(name):
    rows = [json.loads(s) for s in (ROOT/name/'records.jsonl').read_text().splitlines()]
    result = defaultdict(dict)
    for r in rows:
        result[(r['case'], r['slot'], r['repeat'])][r['method']] = r
    return result


def main():
    sensor = load('illumination-v2-tail-model-development-20261015')
    changed = load('illumination-v2-dry-change-development-20261015')
    groups, histories = defaultdict(list), defaultdict(list)
    selected = defaultdict(int)
    for key in sorted(sensor):
        b = sensor[key]['v1']
        c = sensor[key]['sensor_full']
        d = changed[key]['changed_layer']
        for threshold in (.03, .1, .2):
            value, source = b['prediction_m'], 'v1'
            ev = c['evidence']
            if (b['scene'] == 'low_light' and b['sensor'] != 'tof' and
                    ev.get('height_m') is not None and ev.get('layer_pixels', 0) >= 64 and
                    ev.get('layer_tiles', 0) >= 4 and ev.get('median_noise_m', 99) <= threshold and
                    (value is None or abs(c['prediction_m']-value) <= max(.03, 2*ev['median_noise_m']))):
                value, source = c['prediction_m'], 'sensor'
            if (b['sensor'] == 'tof' and d['evidence'].get('height_m') is not None and
                    d['evidence']['dry_reference_pixels'] >= 24):
                value, source = d['prediction_m'], 'dry_changed'
            selected[(threshold, source)] += 1
            for use_temporal in (False, True):
                method = f'hybrid_{threshold}' + ('_temporal' if use_temporal else '')
                hkey = (key[0], key[1], method)
                histories[hkey].append(value)
                output = value
                severe = (b['luma'] <= 2 if b['scene'] == 'low_light' else b['saturated_fraction'] >= .5)
                history = [x for x in histories[hkey][-4:] if x is not None]
                if use_temporal and severe and len(history) >= 3:
                    output = float(np.median(history))
                far = b['camera_to_bottom_m'] >= 6 and b['truth_m'] <= 1
                if not far:
                    groups[f"{b['scene']}|{b['level']}|{method}"].append({**b, 'prediction_m': output})
        if not (b['camera_to_bottom_m'] >= 6 and b['truth_m'] <= 1):
            groups[f"{b['scene']}|{b['level']}|v1"].append(b)
    result = {k: score(v) for k, v in sorted(groups.items())}
    (ROOT/'illumination-v2-policy-development-20261015.json').write_text(json.dumps(result, indent=2))
    for k, v in result.items():
        print(k, 'pass', round(v['pass'], 2), 'mae', round(v['mae'], 2), 'p95', round(v['p95'], 2), 'max', round(v['max'], 2))
    print('SELECTION', dict(selected))


if __name__ == '__main__':
    main()
