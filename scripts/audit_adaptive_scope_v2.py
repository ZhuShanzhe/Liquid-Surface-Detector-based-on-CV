"""Additional scope/rate records; never rewrites the original inference records."""
import importlib.metadata
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

from evaluate_visual_route_v1 import metrics

BASE=Path('/root/autodl-tmp');OUT=BASE/'liquid-depth-artifacts/adaptive-route-v2-20261015'
ROOT=BASE/'liquid-depth-data/synthetic/adaptive-evidence-v2-confirmation-20261015'


def main():
    environment={'python':sys.version,'platform':platform.platform(),
        'packages':{**{k:importlib.metadata.version(k) for k in ['numpy','scipy','torch','Pillow']},'opencv':cv2.__version__}}
    archive=BASE/'liquid-depth-archives/determined-symmetric-geometry-v1-20261015/environment.json'
    if not archive.exists():
        archive.write_text(json.dumps(environment,indent=2));archive.chmod(0o444)
    for name in ['confirmation-synthetic','confirmation-with-real']:
        p=OUT/name;rows=[json.loads(s) for s in (p/'records.jsonl').read_text().splitlines()]
        groups=defaultdict(list);losses=[]
        for r in rows:
            if r['scene']!='depth_failure':continue
            numeric_loss=r['actual_liquid_missing_fraction_offline']
            if numeric_loss is None:continue
            folder=ROOT/f"{r['case']:05d}"
            sem=np.load(folder/'labels_offline.npz')['semantic'];water=sem==2
            source=np.load(folder/f"origin_offline_{r['repeat']:02d}.npy")
            native_loss=1-float((source[water]==1).mean())
            band='above_50pct' if numeric_loss>.5 else f'{min(int(numeric_loss*10),4)*10}-{min(int(numeric_loss*10),4)*10+10}pct'
            groups[band+'|'+r['mode']].append(r)
            if r['mode']=='off':losses.append({'case':r['case'],'repeat':r['repeat'],
                'missing_numeric_depth_fraction':numeric_loss,'missing_native_liquid_source_fraction':native_loss})
        (p/'partial-loss-scope.json').write_text(json.dumps({'definitions':{
            'numeric_loss':'zero/nonpositive raw depth within visible liquid pixels',
            'native_source_loss':'pixels without native liquid-origin return; includes wrong echoes and numerical holes',
            'no_truth_for_runtime_gate':True},'frames':losses,
            'by_numeric_loss':{k:metrics(v) for k,v in groups.items()}},indent=2))
    print('SCOPE_AUDIT_COMPLETE')


if __name__=='__main__':main()
