"""Separate semantic, return-origin and metric-reliability heads from observed RGB-D."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .empty_geometry_v41 import height_calibration
from .liquid_height_metrics import world_points
from .visual_region_model import block


def observed_features(rgb_bgr,depth_m,dry_depth_m,roi,bundle):
    depth=np.asarray(depth_m,dtype=np.float32);dry=np.asarray(dry_depth_m,dtype=np.float32)
    rgb=np.asarray(rgb_bgr)
    if depth.shape!=dry.shape or roi.shape!=depth.shape or rgb.shape!=(*depth.shape,3):raise ValueError('aligned_rgbd_dry_roi_required')
    valid=np.isfinite(depth)&(depth>0);dv=np.isfinite(dry)&(dry>0)
    clean=np.where(valid,depth,0);dc=np.where(dv,dry,0)
    cal=height_calibration(bundle)
    height=world_points(clean,cal)[...,2]-cal.bottom_z_m
    local=cv2.medianBlur(clean,5)
    sigma=.002+.0015*clean**2
    residual=np.where(valid,(clean-local)/np.maximum(sigma,.002),0)
    channels=[cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB).astype(np.float32)/255.,roi[...,None].astype(np.float32),
       clean[...,None]/10,dc[...,None]/10,np.where(valid&dv,dry-depth,0)[...,None],
       valid[...,None],dv[...,None],np.tanh(residual)[...,None],sigma[...,None],np.where(valid,height,0)[...,None]/10]
    out=np.concatenate(channels,axis=-1).transpose(2,0,1).astype(np.float32)
    if not np.isfinite(out).all():raise ValueError('nonfinite_observed_features')
    return out


class RGBDEvidenceNet(nn.Module):
    def __init__(self):
        super().__init__();self.e1=block(12,16);self.e2=block(16,32);self.e3=block(32,64)
        self.d2=block(96,32);self.d1=block(48,16)
        self.semantic=nn.Conv2d(16,4,1);self.origin=nn.Conv2d(16,7,1);self.reliability=nn.Conv2d(16,1,1)

    def forward(self,x):
        a=self.e1(x);b=self.e2(F.avg_pool2d(a,2));c=self.e3(F.avg_pool2d(b,2))
        d=self.d2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        z=self.d1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return {'semantic':self.semantic(z),'origin':self.origin(z),'reliability':self.reliability(z)}


class RGBDEvidencePredictor:
    def __init__(self,checkpoint,device='cpu'):
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        self.device=device;self.net=RGBDEvidenceNet().to(device).eval();self.net.load_state_dict(state['model'])
        self.temperature=state.get('temperature',{'semantic':1.,'origin':1.,'reliability':1.})

    @torch.inference_mode()
    def predict(self,features):
        x=torch.from_numpy(np.asarray(features,dtype=np.float32)[None]).to(self.device)
        output=self.net(x)
        return {k:((v[0]/self.temperature.get(k,1.)).sigmoid() if k=='reliability' else
                    (v[0]/self.temperature.get(k,1.)).softmax(0)).cpu().numpy() for k,v in output.items()}
