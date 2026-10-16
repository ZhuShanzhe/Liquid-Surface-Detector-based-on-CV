"""Portable resource wrapper for the unchanged determined scene algorithms."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from measure_determined import image,initialize,resolve
from liquid_depth.determined_scene_meter import DeterminedSceneMeter,SCENES

ROOT=Path(__file__).resolve().parents[1]


def validated_profile(root=ROOT):
    manifest=json.loads((root/'configs/release_assets.json').read_text())
    profile={}
    for key,item in manifest['models'].items():
        path=(root/item['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('model_asset_outside_release_root')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('model_asset_hash_mismatch:'+key)
        profile[key]=str(path)
    return profile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=sorted(SCENES),required=True)
    p.add_argument('--initialization',type=Path,required=True)
    p.add_argument('--depth',type=Path);p.add_argument('--rgb',type=Path)
    p.add_argument('--left',type=Path);p.add_argument('--right',type=Path)
    p.add_argument('--enable-floating-peak-vision',action='store_true')
    a=p.parse_args()
    visual=a.scene=='floating_objects' and a.enable_floating_peak_vision
    if visual:
        import torch
        torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix='liquid-release-profile-') as tmp:
        if visual:
            spec=json.loads(a.initialization.read_text());base=a.initialization.parent
            if spec['kind']!='empty_geometry':raise ValueError('floating_requires_empty_geometry')
            profile=Path(tmp)/'profile.json'
            profile.write_text(json.dumps(validated_profile()))
            meter=DeterminedSceneMeter.from_empty_geometry(
                json.loads(resolve(base,spec['bundle']).read_text()),
                np.load(resolve(base,spec['dry_depth']),allow_pickle=False),
                profile_path=profile,max_return_depth_m=spec.get('max_return_depth_m',10.5))
        else:
            meter=initialize(a.initialization,a.scene)
        raw=np.load(a.depth,allow_pickle=False) if a.depth else None
        rgb=image(a.rgb) if visual and a.rgb else None
        stereo={'left':image(a.left,True),'right':image(a.right,True)} if a.left and a.right else None
        value=meter.measure(a.scene,depth_m=raw,rgb_bgr=rgb,stereo_images=stereo,
                            visual_enabled=a.enable_floating_peak_vision)
    print(json.dumps(value,allow_nan=False))


if __name__=='__main__':main()
