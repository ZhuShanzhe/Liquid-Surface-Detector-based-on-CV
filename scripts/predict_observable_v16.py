#!/usr/bin/env python3
"""Research inference from camera-exported observations; no labels or scene metadata."""
import argparse,importlib,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from liquid_depth.models.universal import UniversalLiquidSurfaceNet
from liquid_depth.models.liquid_adaptation_v15 import LiquidDFNet
from liquid_depth.models.observable_v16 import ObservableLiquidPipeline
from liquid_depth.observation_v16 import prepare_observation

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rgb',type=Path,required=True);p.add_argument('--raw-depth',type=Path,required=True)
    p.add_argument('--camera-json',type=Path,required=True,help='SDK export: explicit depth unit scale and registration/calibration')
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--route',choices=['full','gated'],default='gated')
    p.add_argument('--transcg-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rgb_bgr=cv2.imread(str(a.rgb))
    if rgb_bgr is None:raise FileNotFoundError(a.rgb)
    raw=np.load(a.raw_depth,allow_pickle=False)
    camera=json.loads(a.camera_json.read_text())
    rgb,depth=prepare_observation(rgb_bgr,raw,camera)
    rgb=cv2.resize(rgb,(320,180),interpolation=cv2.INTER_LINEAR)
    depth=cv2.resize(depth,(320,180),interpolation=cv2.INTER_NEAREST)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
    config=json.loads((a.weights/'locator-config.json').read_text())
    locator=UniversalLiquidSurfaceNet(**config)
    locator.load_state_dict(torch.load(a.weights/'locator.pth',map_location='cpu',weights_only=True)['model'])
    sys.path.insert(0,str(a.transcg_source.resolve()))
    network=importlib.import_module('models.DFNet').DFNet(in_channels=4,hidden_channels=64,L=5,k=12)
    model=LiquidDFNet(network)
    model.load_state_dict(torch.load(a.weights/a.route/'best.pth',map_location='cpu',weights_only=True)['depth'])
    pipe=ObservableLiquidPipeline(locator,model,a.route).to(device).eval()
    rgb=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].to(device)
    depth=torch.from_numpy(depth[None,None].copy()).to(device)
    start=time.perf_counter()
    with torch.inference_mode():result=pipe(rgb,depth)
    if device.type=='cuda':torch.cuda.synchronize()
    z=result['depth_m'][0,0].cpu().numpy();prob=result['mask_probability'][0,0].cpu().numpy()
    np.save(a.output/'depth_m.npy',z)
    np.save(a.output/'mask_probability.npy',prob)
    np.save(a.output/'masked_depth_m.npy',np.where(prob>=.5,z,0))
    status={'research_only':True,'depth_confidence_qualified':False,'liquid_height_available':False,
            'has_depth_anchor':bool(result['has_depth_anchor'][0]),'predicted_region_raw_points':int(result['anchor_count'][0]),
            'seconds_including_first_call':time.perf_counter()-start,'input_resolution':[320,180],
            'input_parameters_used':['RGB','raw depth','SDK depth units','SDK alignment or factory calibration'],
            'mask_probability_is_not_depth_confidence':True}
    (a.output/'status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status))
if __name__=='__main__':main()
