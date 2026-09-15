"""Observable-only liquid recovery. No scene metadata or target enters inference."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .liquid_adaptation_v15 import prepare_depth,encode_depth

def sanitize_raw(raw):
    return torch.where(torch.isfinite(raw)&(raw>=.1)&(raw<=10),raw,0)

def encode_rgbd(rgb,raw):
    raw=sanitize_raw(raw)
    mean=rgb.new_tensor([.485,.456,.406])[None,:,None,None]
    std=rgb.new_tensor([.229,.224,.225])[None,:,None,None]
    return torch.cat([(rgb-mean)/std,torch.where(raw>0,encode_depth(raw),0),(raw>0).float()],1)

def gated_measurements(raw,probability,threshold=.5,radius=4,min_points=32):
    """The support is estimated from RGB-D, never supplied by a human or label."""
    raw=sanitize_raw(raw)
    keep=F.max_pool2d((probability>=threshold).float(),2*radius+1,stride=1,padding=radius)>0
    filtered=torch.where(keep,raw,0)
    count=(filtered>0).flatten(1).sum(1)
    return torch.where((count>=min_points)[:,None,None,None],filtered,0),count

def prepare_batch(rgb,raw):
    rows=[prepare_depth(a,b) for a,b in zip(rgb.detach().cpu().permute(0,2,3,1).numpy(),raw.detach().cpu()[:,0].numpy())]
    return [torch.as_tensor(np.stack([r[i] for r in rows]),device=rgb.device) for i in range(5)]

class ObservableLiquidPipeline(nn.Module):
    """RGB [0,1], aligned metric depth in meters; outputs are research predictions."""
    def __init__(self,locator,depth_model,route='gated'):
        super().__init__()
        if route not in ('full','gated'):raise ValueError(route)
        self.locator=locator
        self.depth_model=depth_model
        self.route=route
    def forward(self,rgb,raw):
        if rgb.ndim!=4 or rgb.shape[1]!=3 or raw.shape!=(rgb.shape[0],1,*rgb.shape[2:]):
            raise ValueError('Expected aligned BCHW RGB and B1HW metric depth')
        if not torch.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>1:
            raise ValueError('RGB must be finite in [0,1]')
        raw=sanitize_raw(raw)
        with torch.no_grad():
            observation=self.locator(encode_rgbd(rgb,raw))
            probability=observation['mask_logits'].sigmoid()
            prior=observation['depth_m']
            selected,count=gated_measurements(raw,probability)
            source=selected if self.route=='gated' else raw
            rgb240,dn,low,span,anchor=prepare_batch(rgb,source)
        depth=self.depth_model(rgb240,dn,low,span,rgb,source,prior,probability,anchor)
        depth=torch.where(torch.isfinite(depth)&(depth>=.1)&(depth<=10),depth,0)
        return {'depth_m':depth,'mask_probability':probability,'anchor_count':count,
                'has_depth_anchor':anchor>0,'source_depth_m':source}
