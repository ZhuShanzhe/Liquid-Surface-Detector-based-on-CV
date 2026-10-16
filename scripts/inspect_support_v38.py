"""Offline attribution of erroneous surface support; does not alter estimates."""
import json
from pathlib import Path
from dataclasses import replace
import cv2,numpy as np,torch
from evaluate_liquid_height_v21 import Predictor
from evaluate_device_letterbox_v38 import infer_letterbox
from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.surface_refinement import support
ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/device-v38-test-20261012')
DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/device-v38-test-20261012')
def main():
    cv2.setNumThreads(1);torch.set_num_threads(2);model=Predictor('v8')
    rows=json.loads((ROOT/'letterbox/test-raw.json').read_text())['records'];out=[]
    for profile in ['blaze101_long','gemini2xl_dense']:
      for scene in ['ordinary','low_light','glare','floating_objects','depth_failure','uneven_surface']:
        candidates=[r for r in rows if (r['profile'],r['scenario'],r['regime'],r['variant'],r['method'],r['population'])==
                    (profile,scene,'nominal','base','native_refined','uniform') and r['predicted_height_m'] is not None]
        if not candidates:continue
        row=max(candidates,key=lambda r:r['absolute_error_m']);folder=DATA/f"{row['slot']:05d}"
        meta=json.loads((folder/'metadata.json').read_text());rgb=cv2.imread(str(folder/'rgb.png'))
        raw=np.load(folder/f"raw_{row['repeat']:02d}_nominal_base.npy")
        p=infer_letterbox(model,rgb,raw);cal=replace(HeightCalibration.from_simulation(meta),bottom_z_m=meta['interior_bottom_z_m'])
        data,reason=support(raw,p['mask'],cal.intrinsics,cal.camera_to_world_cv,cal.radii_m)
        if data is None:continue
        points,pixels,z,weights,_=data
        outside=np.sum(abs(points[:,:2]/cal.radii_m)**cal.footprint_exponent,axis=1)>1
        bottom_return=abs(points[:,2]-cal.bottom_z_m)<.02
        # Evaluator-only liquid surface discrepancy, never used to reroute an output.
        wrong=abs(points[:,2]-cal.bottom_z_m-row['true_height_m'])>max(.02,.02*row['true_height_m'])
        out.append(dict(profile=profile,scenario=scene,slot=row['slot'],repeat=row['repeat'],
          actual_liquid_height_m=row['true_height_m'],predicted_liquid_height_m=row['predicted_height_m'],
          absolute_error_mm=1000*row['absolute_error_m'],mask_iou=row['mask_iou'],
          selected_points=len(z),outside_footprint_point_pct=float(outside.mean()*100),
          outside_footprint_weight_pct=float(weights[outside].sum()*100),
          bottom_return_weight_pct=float(weights[bottom_return].sum()*100),
          evaluator_wrong_surface_point_pct=float(wrong.mean()*100),
          evaluator_wrong_surface_weight_pct=float(weights[wrong].sum()*100)))
    (ROOT/'support-attribution.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
