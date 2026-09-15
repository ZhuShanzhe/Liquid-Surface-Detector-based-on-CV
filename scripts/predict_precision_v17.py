#!/usr/bin/env python3
"""Portable precision pipeline: camera observations and frozen research weights only."""
import argparse,importlib,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from liquid_depth.models.universal import UniversalLiquidSurfaceNet
from liquid_depth.models.liquid_adaptation_v15 import LiquidDFNet
from liquid_depth.models.precision_v17 import PrecisionPipeline,fit_surface,DistanceBlendPipeline
from liquid_depth.observation_v16 import prepare_observation

class SurfacePrediction(torch.nn.Module):
    def __init__(self,pipe):
        super().__init__();self.pipe=pipe
    def forward(self,rgb,raw):
        result=self.pipe(rgb,raw)
        result['depth_m']=fit_surface(result['depth_m'],result['mask_probability'])
        return result

def load_pipeline(weights,source,variant='selected',device='cuda'):
    profile=json.loads((weights/'runtime-profile.json').read_text())
    name=profile['selected'] if variant=='selected' else variant
    config=profile['variants'][name]
    if config.get('kind')=='distance_blend':
        near,ncfg,_=load_pipeline(weights,source,config['near'],device)
        far,fcfg,_=load_pipeline(weights,source,config['far'],device)
        if ncfg['surface_fit']:near=SurfacePrediction(near)
        if fcfg['surface_fit']:far=SurfacePrediction(far)
        return DistanceBlendPipeline(near,far,*config['window_m']).to(device).eval(),config,name
    locator=UniversalLiquidSurfaceNet(**json.loads((weights/'locator-config.json').read_text()))
    locator.load_state_dict(torch.load(weights/'locator.pth',map_location='cpu',weights_only=True)['model'])
    sys.path.insert(0,str(source.resolve()))
    network=importlib.import_module('models.DFNet').DFNet(in_channels=4,hidden_channels=64,L=5,k=12)
    model=LiquidDFNet(network);model.load_state_dict(torch.load(weights/config['checkpoint'],map_location='cpu',weights_only=True)['depth'])
    return PrecisionPipeline(locator,model,config['radius']).to(device).eval(),config,name

def main():
    p=argparse.ArgumentParser()
    for key in ('rgb','raw-depth','camera-json','weights','transcg-source','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--variant',default='selected');a=p.parse_args()
    bgr=cv2.imread(str(a.rgb))
    if bgr is None:raise FileNotFoundError(a.rgb)
    rgb,raw=prepare_observation(bgr,np.load(a.raw_depth,allow_pickle=False),json.loads(a.camera_json.read_text()))
    rgb=cv2.resize(rgb,(320,180),interpolation=cv2.INTER_LINEAR);raw=cv2.resize(raw,(320,180),interpolation=cv2.INTER_NEAREST)
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    pipe,config,name=load_pipeline(a.weights,a.transcg_source,a.variant,device)
    rgb=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].to(device);raw=torch.from_numpy(raw[None,None]).to(device)
    start=time.perf_counter()
    with torch.inference_mode():
        result=pipe(rgb,raw)
        if config['surface_fit']:result['depth_m']=fit_surface(result['depth_m'],result['mask_probability'])
    if device=='cuda':torch.cuda.synchronize()
    a.output.mkdir(parents=True,exist_ok=False)
    z=result['depth_m'][0,0].cpu().numpy();prob=result['mask_probability'][0,0].cpu().numpy()
    np.save(a.output/'depth_m.npy',z);np.save(a.output/'mask_probability.npy',prob);np.save(a.output/'masked_depth_m.npy',np.where(prob>=.5,z,0))
    status={'variant':name,'radius':config['radius'],'surface_fit':config['surface_fit'],'research_only':True,'depth_confidence_qualified':False,'liquid_height_available':False,'has_depth_anchor':bool(result['has_depth_anchor'][0]),'source_points':int(result['anchor_count'][0]),'seconds_first_call':time.perf_counter()-start,'input_keys':['RGB','raw metric depth','SDK unit/registration configuration']}
    if 'estimated_distance_m' in result:
        estimate=float(result['estimated_distance_m'][0])
        status['estimated_distance_m']=estimate if np.isfinite(estimate) else None
        status['far_weight']=float(result['far_weight'][0])
        status['diagnostic_only']=True
    (a.output/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))
if __name__=='__main__':main()
