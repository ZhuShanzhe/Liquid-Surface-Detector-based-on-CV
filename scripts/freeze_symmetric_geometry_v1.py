"""Freeze the user's selected deterministic geometric baseline without modifying it."""
import hashlib
import json
import shutil
from pathlib import Path

BASE=Path('/root/autodl-tmp');REPO=BASE/'Liquid-Surface-Detector-based-on-CV'
OUT=BASE/'liquid-depth-archives/determined-symmetric-geometry-v1-20261015'
SOURCE=BASE/'liquid-depth-artifacts/visual-route-v1-20261015'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(SOURCE/'experiment-source.tar',OUT/'source.tar')
    rows=[json.loads(s) for s in (SOURCE/'final-model-v1/records.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['method']=='symmetric_depth']
    (OUT/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary=json.loads((SOURCE/'final-model-v1/summary.json').read_text())
    summary={k:v for k,v in summary.items() if k.endswith('|symmetric_depth')}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    shutil.copyfile(SOURCE/'final-model-v1/remaining-far-shallow-summary.json',OUT/'all-methods-strata-for-context.json')
    files=['src/liquid_depth/symmetric_visual_meter.py','src/liquid_depth/supported_scene_meter.py',
           'src/liquid_depth/surface_refinement.py','src/liquid_depth/empty_geometry_v41.py']
    manifest={'user_selected_baseline':'symmetric geometry without RGB or learned probabilities',
              'configuration':{'predictor':None,'probabilities':None,'rgb_not_used':True},
              'status':'determined_backup_baseline_not_claim_of_all_scene_optimality',
              'limitations':['glare and partial-loss regressions retained','transparent and translucent retain original five-prior refraction path'],
              'source_sha256':{f:hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in files},
              'archive_sha256':hashlib.sha256((OUT/'source.tar').read_bytes()).hexdigest(),
              'reference_unique_frames':len(rows)}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for p in OUT.iterdir():p.chmod(0o444)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
