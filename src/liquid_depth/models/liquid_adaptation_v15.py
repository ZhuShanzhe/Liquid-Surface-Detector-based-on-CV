"""Opt-in DFNet liquid adaptation and learned selective prediction."""
from pathlib import Path
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from scipy.interpolate import NearestNDInterpolator

def prepare_depth(rgb,raw):
 """Official wide-range DFNet normalization; no target or mask input."""
 image=cv2.resize(rgb,(320,240),interpolation=cv2.INTER_NEAREST)
 depth=cv2.resize(raw,(320,240),interpolation=cv2.INTER_NEAREST).copy()
 valid=np.isfinite(depth)&(depth>=.1)&(depth<=10)
 depth=np.where(valid,depth,0).astype(np.float32)
 available=depth[depth>0]
 if available.size:
  mean=float(available.mean());radius=3*float(available.std())
  depth=np.where((depth>=mean-radius)&(depth<=mean+radius),depth,0)
 coords=np.where(depth>0);anchor=bool(coords[0].size)
 if anchor:
  depth=NearestNDInterpolator(np.transpose(coords),depth[coords])(*np.indices(depth.shape)).astype(np.float32)
 low=float(depth.min()-.5*depth.std()-1e-6)
 high=float(depth.max()+.5*depth.std()+1e-6)
 return image.transpose(2,0,1).copy(),(depth-low)/(high-low),np.float32(low),np.float32(high-low),np.float32(anchor)

def encode_depth(raw):
 return torch.log(raw.clamp(.1,10)/.1)/np.log(100)

def evidence(rgb,raw,depth,prior,mask):
 return torch.cat([rgb,encode_depth(raw), (raw>0).float(),encode_depth(depth),encode_depth(prior),mask,((depth-prior)/prior.clamp_min(.1)).clamp(-1,1),((depth-raw)/depth.clamp_min(.1)).clamp(-1,1)*(raw>0)],1)

class LiquidDFNet(nn.Module):
 def __init__(self,network):
  super().__init__();self.network=network
  self.correction=nn.Sequential(nn.Conv2d(10,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,1,1))
  nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)
 def forward(self,rgb240,dnorm,low,span,rgb,raw,prior,mask,anchor):
  normalized=self.network(rgb240,dnorm)
  base=normalized[:,None]*span[:,None,None,None]+low[:,None,None,None]
  base=F.interpolate(base,raw.shape[-2:],mode='nearest')
  features=evidence(rgb,raw,base,prior,mask)
  depth=base*torch.exp(.2*torch.tanh(self.correction(features)))
  return torch.where(anchor[:,None,None,None]>0,depth,torch.zeros_like(depth))

class QualityNet(nn.Module):
 def __init__(self):
  super().__init__()
  self.net=nn.Sequential(nn.Conv2d(10,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,1,1))
 def forward(self,rgb,raw,depth,prior,mask):
  f=evidence(rgb,raw,torch.nan_to_num(depth,nan=0,posinf=0,neginf=0),prior,mask)
  return self.net(f)

def depth_loss(depth,truth,valid):
 tol=(truth*.02).clamp_min(.005)
 error=(depth-truth).abs()/tol
 loss=torch.log1p(error)
 sums=(loss*valid).sum((1,2,3));counts=valid.sum((1,2,3))
 return (sums/counts.clamp_min(1))[counts>0].mean()

class ContextQualityNet(nn.Module):
 """Correctness prediction with frozen semantic features and whole-frame context."""
 def __init__(self,feature_provider):
  super().__init__();self.feature_provider=feature_provider
  self.local=nn.Sequential(nn.Conv2d(34,32,3,padding=1),nn.SiLU())
  self.global_context=nn.Sequential(nn.Conv2d(24,32,1),nn.SiLU(),nn.Conv2d(32,32,1))
  self.head=nn.Sequential(nn.Conv2d(32,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,1,1))
 def forward(self,rgb,raw,depth,prior,mask):
  mean=rgb.new_tensor([.485,.456,.406])[None,:,None,None];std=rgb.new_tensor([.229,.224,.225])[None,:,None,None]
  x=torch.cat([(rgb-mean)/std,torch.where(raw>0,encode_depth(raw),0),(raw>0).float()],1)
  with torch.no_grad():features=self.feature_provider(x)
  depth=torch.nan_to_num(depth,nan=0,posinf=0,neginf=0)
  local=self.local(torch.cat([evidence(rgb,raw,depth,prior,mask),features],1))
  global_features=self.global_context(features.mean((2,3),keepdim=True))
  return self.head(local+global_features)
