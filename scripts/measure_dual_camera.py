"""Dual-camera public entry: ToF/SL only, raw phase priors for uniform refraction."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
from measure_determined import image,resolve
from measure_release import validated_profile
from liquid_depth.dual_camera_scene_meter_v2 import DualCameraSceneMeter,SCENES


def initialize(path,scene,profile_path=None):
    path=Path(path);spec=json.loads(path.read_text());base=path.parent
    sensor=spec['sensor_family']
    if spec['kind']=='empty_geometry':
        bundle=json.loads(resolve(base,spec['bundle']).read_text())
        if bundle['sensor_family']!=sensor:raise ValueError('initialization_camera_family_mismatch')
        options={'max_return_depth_m':spec.get('max_return_depth_m',10.5)}
        if profile_path:options['profile_path']=profile_path
        return DualCameraSceneMeter.from_empty_geometry(bundle,np.load(resolve(base,spec['dry_depth']),allow_pickle=False),**options)
    if spec['kind']=='five_phase_priors':
        if spec.get('scene')!=scene:raise ValueError('phase_initialization_scene_mismatch')
        with np.load(resolve(base,spec['rays_roi']),allow_pickle=False) as f:q=f['ray_coordinates'];roi=f['roi']
        captures=[]
        for filename in spec['prior_captures']:
            with np.load(resolve(base,filename),allow_pickle=False) as f:captures.append(f['correlations'])
        return DualCameraSceneMeter.from_phase_priors(sensor=sensor,scene=scene,profile=spec['profile'],ray_coordinates=q,roi=roi,
            prior_correlations=captures,known_heights_m=spec['known_heights_m'])
    raise ValueError('empty_geometry_or_five_phase_priors_required')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',choices=sorted(SCENES),required=True)
    p.add_argument('--initialization',type=Path,required=True);p.add_argument('--depth',type=Path);p.add_argument('--rgb',type=Path)
    p.add_argument('--phase-capture',type=Path);p.add_argument('--enable-floating-peak-vision',action='store_true');a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='liquid-dual-assets-') as temp:
        profile=None
        if a.scene=='floating_objects' and a.enable_floating_peak_vision:
            import torch
            torch.set_num_threads(4)
            profile=Path(temp)/'models.json';profile.write_text(json.dumps(validated_profile()))
        meter=initialize(a.initialization,a.scene,profile_path=profile)
        correlation=None
        if a.phase_capture:
            with np.load(a.phase_capture,allow_pickle=False) as f:correlation=f['correlations']
        result=meter.measure(a.scene,depth_m=np.load(a.depth,allow_pickle=False) if a.depth else None,
            rgb_bgr=image(a.rgb) if a.rgb else None,correlations=correlation,visual_enabled=a.enable_floating_peak_vision)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
