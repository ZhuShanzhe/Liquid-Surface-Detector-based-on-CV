"""Calibrated dual-ROI highlight head (128px); no liquid estimate or truth input."""
import cv2,numpy as np,torch
from torch import nn
from .new_view_features import aperture,bottom_roi,height_calibration
def glare_tensor(rgb,depth,bundle):
    cal=height_calibration(bundle);wide=aperture(bundle,depth.shape)
    bottom=bottom_roi(bundle,depth.shape)
    tensors=[]
    for roi in (wide,bottom):
        yy,xx=np.nonzero(roi)
        if not len(xx):raise ValueError('no_calibrated_aperture')
        box=np.s_[yy.min():yy.max()+1,xx.min():xx.max()+1];r=roi[box]
        color=rgb[box].astype(np.float32)/255;color=np.where(r[...,None],color,0)
        z=depth[box];valid=(z>=.08)&np.isfinite(z)&r
        local=cv2.medianBlur(np.asarray(z,np.float32),5)
        residual=np.where(valid,np.tanh((z-local)/np.maximum(.0015+.0007*z,.0015)),0)
        crop=np.concatenate([color,valid[...,None].astype(np.float32),residual[...,None]],axis=-1)
        tensors.append(cv2.resize(crop,(128,128),interpolation=cv2.INTER_LINEAR))
    return np.concatenate(tensors,axis=-1).transpose(2,0,1).astype(np.float32)
class GlareNet(nn.Module):
    def __init__(self):
        super().__init__()
        def conv(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
        self.encoder=nn.Sequential(conv(10,16),nn.AvgPool2d(2),conv(16,24),nn.AvgPool2d(2),conv(24,32),nn.AvgPool2d(2),conv(32,48))
        self.head=nn.Sequential(nn.Linear(96,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,x):
        z=self.encoder(x);return self.head(torch.cat([z.mean((2,3)),z.amax((2,3))],1))[:,0]
class GlarePredictor:
    def __init__(self,path):
        self.net=GlareNet().cuda().eval();self.net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'])
    @torch.inference_mode()
    def predict(self,rgb,depth,bundle):
        x=torch.from_numpy(glare_tensor(rgb,depth,bundle))[None].cuda();return float(self.net(x).sigmoid()[0].cpu())
