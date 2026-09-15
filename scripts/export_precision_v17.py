#!/usr/bin/env python3
"""Collect exact frozen reference/locator dependencies for portable V17 inference."""
import hashlib,json,shutil
from pathlib import Path
from train_precision_v17 import OUT
from train_observable_v16 import OUT as V16
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    frozen=json.loads((OUT/'frozen-experiment.json').read_text())
    assert digest(V16/'locator.pth')==frozen['locator_sha256']
    (OUT/'reference').mkdir(exist_ok=True)
    for source,dest in [(V16/'locator.pth',OUT/'locator.pth'),(V16/'locator-config.json',OUT/'locator-config.json'),(V16/'gated/best.pth',OUT/'reference/best.pth')]:
        if dest.exists():assert digest(source)==digest(dest)
        else:shutil.copyfile(source,dest)
    profile=json.loads((OUT/'runtime-profile.json').read_text())
    for c in frozen['candidates']:
        assert digest(OUT/profile['variants'][c['name']]['checkpoint'])==c['checkpoint_sha256']
    phase_b=OUT/'phase-b/frozen.json'
    if phase_b.exists():
        b=json.loads(phase_b.read_text())
        assert digest(OUT/'reference/best.pth')==b['near_sha256']
        assert digest(OUT/'r4/best.pth')==b['far_sha256']
        profile['variants']['distance_blend_diagnostic']=dict(kind='distance_blend',near='v16_plane',far='mae_r4_plane',window_m=b['diagnostic']['window_m'],radius=4,surface_fit=False,diagnostic_only=True)
        (OUT/'runtime-profile.json').write_text(json.dumps(profile,indent=2))
    print('PORTABLE_WEIGHTS_VERIFIED',frozen['selected'])
if __name__=='__main__':main()
