"""Index far/shallow evidence by cohort; never pool unlike datasets."""
import json
from collections import defaultdict
from pathlib import Path

from evaluate_determined_v1 import summarize

OUT = Path('/root/autodl-tmp/liquid-depth-artifacts/determined-all-scenes-v1-20261016')


def main():
    sources = [OUT/'opaque-records.jsonl', OUT/'depth-grid/records.jsonl',
               OUT/'refraction-frozen-records.json', OUT/'dedicated-floating-v8-records.json']
    groups = defaultdict(list)
    far = []
    for path in sources:
        rows = [json.loads(s) for s in path.read_text().splitlines()] if path.suffix == '.jsonl' else json.loads(path.read_text())
        for r in rows:
            if not r.get('scope_included', True):
                continue
            h = r['truth']['liquid_height_m']
            if r['camera_to_bottom_m'] < 6 or not 0 <= h <= 1:
                continue
            part = 'dry' if h == 0 else '0-0.1m' if h <= .1 else '0.1-0.5m' if h <= .5 else '0.5-1m'
            r = {**r, 'far_depth_band': part, 'source_record': str(path)}
            if 'cohort' not in r:
                r['cohort'] = 'dedicated_float_202610164913'
            far.append(r)
            for label in dict.fromkeys([part, 'dry' if h == 0 else 'wet']):
                groups[(r['cohort'], r['scene'], label)].append(r)
    result = {}
    for (cohort, scene, part), rows in groups.items():
        summary = summarize(rows)
        result[cohort+'|'+scene+'|'+part] = {'rows': len(rows),
            'installations_or_configurations': len({r.get('case', r.get('config_id')) for r in rows}),
            'statistics': summary[scene+'|all_frames|all']}
    (OUT/'far-shallow-records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in far))
    (OUT/'far-shallow-index.json').write_text(json.dumps(result, indent=2))
    print('FAR_SHALLOW_INDEXED', len(far), len(result), flush=True)


if __name__ == '__main__':
    main()
