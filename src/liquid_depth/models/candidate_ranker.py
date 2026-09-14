"""Opt-in V12 candidate ranking. Frozen depths, learned quality and abstention.

Geometry uses axial camera depth and Blender-camera rays/normals (x right,
y up, z backward), matching this simulator. Other camera conventions must be
converted explicitly. Intrinsics are measured inputs, never pose/level labels.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def shift_replicate(x, dy, dx):
    h, w = x.shape[-2:]
    yy = (torch.arange(h, device=x.device)+dy).clamp(0,h-1)
    xx = (torch.arange(w, device=x.device)+dx).clamp(0,w-1)
    return x.index_select(-2,yy).index_select(-1,xx)


def plane_support(depths, normal, rays):
    """Soft local tangent-plane residual, minimized over unordered neighbors.

    Not an independent measurement or a hard planarity assumption. A coherent
    wrong plane can also score well. Replicated boundaries have neutral residual.
    """
    center=(normal*rays).sum(1,keepdim=True)*depths
    tol=(depths*.02).clamp_min(.005)
    features=[]
    for step in (2,8):
        residuals=[]
        for dy,dx in ((step,0),(-step,0),(0,step),(0,-step)):
            neighbor=shift_replicate(depths,dy,dx)
            projection=(normal*shift_replicate(rays,dy,dx)).sum(1,keepdim=True)
            residual=(neighbor[:,None]*projection[:,None]-center[:,:,None]).abs().amin(2)
            residuals.append(torch.log1p((residual/tol).clamp_max(100)))
        stack=torch.stack(residuals,2)
        features.extend((stack.mean(2),stack.amax(2)))
    return torch.stack(features,2)


class CandidateRanker(nn.Module):
    def __init__(self, geometry=True, context_features=0):
        super().__init__()
        self.geometry=bool(geometry)
        self.context_features=int(context_features)
        self.global_context=nn.Conv2d(self.context_features,16,1) if self.context_features else None
        self.context=nn.Sequential(nn.Conv2d(10+self.context_features,16,3,padding=1),nn.SiLU(),
                                   nn.Conv2d(16,16,3,padding=1),nn.SiLU())
        self.scorer=nn.Sequential(nn.Conv2d(28,24,1),nn.SiLU(),nn.Conv2d(24,1,1))
        self.none_head=nn.Sequential(nn.Conv2d(18,16,1),nn.SiLU(),nn.Conv2d(16,1,1))

    def forward(self, x, p, rays):
        d=p['layer_depths_m'].detach().float()
        if d.shape[1]!=4 or x.shape[1]!=5 or rays.shape[1]!=3:
            raise ValueError('Expected four candidates, five input channels and three ray channels')
        prior=p['depth_m'].detach().float().clamp_min(.1)
        normal=p['normal'].detach().float()
        inputs=[x.float(),torch.log(prior/.1)/math.log(100),
                p['mask_logits'].detach().float().sigmoid(),normal]
        if self.context_features:
            features=p['rank_features'].detach().float()
            if features.shape[1]!=self.context_features:raise ValueError('Wrong frozen feature channel count')
            inputs.append(features)
        context=self.context(torch.cat(inputs,1))
        if self.global_context is not None:
            context=context+self.global_context(features.mean((2,3),keepdim=True))
        pair=(d[:,:,None]-d[:,None]).abs()/d[:,:,None].clamp_min(.1)
        eye=torch.eye(4,device=d.device,dtype=torch.bool)[None,:,:,None,None]
        separation=pair.masked_fill(eye,float('inf')).amin(2)
        raw=.1*torch.exp(x[:,3:4].float()*math.log(100))
        features=[torch.log(d.clamp_min(.1)/.1)/math.log(100),
                  torch.log(d.clamp_min(.1)/prior).clamp(-4,4),
                  (p['layer_scales_m'].detach()/d).clamp(0,10),
                  p['layer_presence_logits'].detach().sigmoid(),
                  p['liquid_interface_probability'].detach(),p['layer_confidence'].detach(),
                  separation.clamp_max(10),
                  ((d-raw)/d).clamp(-10,10)*x[:,4:5]]
        local=plane_support(d,normal,rays.float()) if self.geometry else d.new_zeros((*d.shape[:2],4,*d.shape[2:]))
        features=torch.cat([torch.stack(features,2),local],2)
        b,k,_,h,w=features.shape
        shared=context[:,None].expand(-1,k,-1,-1,-1)
        logits=self.scorer(torch.cat([shared,features],2).reshape(b*k,28,h,w)).reshape(b,k,h,w)
        pooled=torch.cat([context,logits.mean(1,keepdim=True),logits.amax(1,keepdim=True)],1)
        none=self.none_head(pooled)
        index=logits.argmax(1,keepdim=True)
        quality=logits.sigmoid().gather(1,index)
        return {'quality_logits':logits,'none_logits':none,'index':index,
                'depth_m':d.gather(1,index),'raw_confidence':quality*(1-none.sigmoid())}


def ranking_loss(result, candidates, y):
    valid=y['valid']>0
    tolerance=(y['depth_m']*.02).clamp_min(.005)
    good=((candidates-y['depth_m']).abs()<=tolerance)&valid
    exists=good.any(1,keepdim=True)
    weight=torch.where(valid,1.,.1)
    quality=F.binary_cross_entropy_with_logits(result['quality_logits'],good.float(),reduction='none')
    quality=(quality*weight).sum()/(weight.sum()*candidates.shape[1]).clamp_min(1)
    none=F.binary_cross_entropy_with_logits(result['none_logits'],(~exists).float(),reduction='none')
    none=(none*weight).sum()/weight.sum().clamp_min(1)
    # All qualifying candidates are positives; no forced "least-wrong" positive.
    soft=good.float()/good.sum(1,keepdim=True).clamp_min(1)
    rank=-(soft*result['quality_logits'].log_softmax(1)).sum(1,keepdim=True)
    rank=(rank*exists).sum()/exists.sum().clamp_min(1)
    return quality+.5*none+.5*rank


class MonotoneCalibrator:
    """Fixed-bin weighted PAVA on disjoint calibration sequences; no extrapolation claim."""
    def __init__(self, values=None, bins=32):
        self.bins=bins
        self.values=np.asarray(values,dtype=np.float64) if values is not None else None

    def fit_histogram(self, count, positive):
        count=np.asarray(count,dtype=np.float64); positive=np.asarray(positive,dtype=np.float64)
        if len(count)!=self.bins or count.sum()==0:raise ValueError('Empty or incompatible calibration histogram')
        blocks=[]
        for i in range(self.bins):
            if count[i]==0:continue
            blocks.append([i,i,count[i],positive[i]])
            while len(blocks)>1 and blocks[-2][3]/blocks[-2][2]>blocks[-1][3]/blocks[-1][2]:
                right=blocks.pop(); left=blocks.pop()
                blocks.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
        self.values=np.empty(self.bins)
        first=blocks[0]; self.values[:first[0]]=first[3]/first[2]
        for j,block in enumerate(blocks):
            end=blocks[j+1][0] if j+1<len(blocks) else self.bins
            self.values[block[0]:end]=block[3]/block[2]
        return self

    def __call__(self, probability):
        if self.values is None:raise RuntimeError('Calibration has not been fitted')
        values=torch.as_tensor(self.values,device=probability.device,dtype=probability.dtype)
        index=(probability.clamp(0,1)*self.bins).long().clamp_max(self.bins-1)
        return values[index]
