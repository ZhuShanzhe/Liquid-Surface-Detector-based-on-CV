"""V17 precision-focused observation routing and prediction-only surface fitting."""
import torch
from torch.nn import functional as F
from .observable_v16 import ObservableLiquidPipeline,sanitize_raw,encode_rgbd,prepare_batch

def route_measurements(raw,probability,radius=0,min_points=32):
    keep=(probability>=.5).float()
    if radius>0:keep=F.max_pool2d(keep,2*radius+1,1,radius)
    elif radius<0:
        r=-radius;keep=-F.max_pool2d(-keep,2*r+1,1,r)
    filtered=torch.where(keep>0,sanitize_raw(raw),0)
    count=(filtered>0).flatten(1).sum(1)
    return torch.where((count>=min_points)[:,None,None,None],filtered,0),count

class PrecisionPipeline(ObservableLiquidPipeline):
    def __init__(self,locator,depth_model,radius=0):
        super().__init__(locator,depth_model,'gated');self.radius=radius
    def forward(self,rgb,raw):
        if rgb.ndim!=4 or rgb.shape[1]!=3 or raw.shape!=(rgb.shape[0],1,*rgb.shape[2:]):
            raise ValueError('Expected aligned BCHW RGB and B1HW depth')
        if not torch.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>1:raise ValueError('Invalid RGB')
        raw=sanitize_raw(raw)
        with torch.no_grad():
            p=self.locator(encode_rgbd(rgb,raw));mask=p['mask_logits'].sigmoid()
            source,count=route_measurements(raw,mask,self.radius)
            rgb240,dn,low,span,anchor=prepare_batch(rgb,source)
        depth=self.depth_model(rgb240,dn,low,span,rgb,source,p['depth_m'],mask,anchor)
        unclipped=depth
        depth=torch.where(torch.isfinite(depth)&(depth>=.1)&(depth<=10),depth,0)
        return {'depth_m':depth,'mask_probability':mask,'anchor_count':count,'has_depth_anchor':anchor>0,'source_depth_m':source,'unclipped_depth_m':unclipped}

def precision_loss(depth,truth,valid):
    error=(depth-truth).abs()
    counts=valid.sum((1,2,3))
    absolute=(error*valid).sum((1,2,3))/counts.clamp_min(1)
    relative=(error/truth.clamp_min(.1)*valid).sum((1,2,3))/counts.clamp_min(1)
    return (absolute+.1*relative)[counts>0].mean()

def fit_surface(depth,probability,min_points=64):
    """Robust inverse-depth plane in normalized image coordinates. No K/pose/GT.
    Selects support using estimated liquid identity; cannot resolve a wrong layer
    or a constant scale bias. Unsupported/degenerate frames preserve prediction.
    """
    h,w=depth.shape[-2:]
    y,x=torch.meshgrid(torch.linspace(-1,1,h,device=depth.device),torch.linspace(-1,1,w,device=depth.device),indexing='ij')
    A=torch.stack([x,y,torch.ones_like(x)],-1).reshape(-1,3).double()
    outputs=[]
    for d,mask in zip(depth,probability):
        flat=d.flatten();ok=(mask.flatten()>=.5)&torch.isfinite(flat)&(flat>=.1)&(flat<=10)
        ids=torch.where(ok)[0][::4]
        if len(ids)<min_points:outputs.append(d);continue
        a=A[ids];b=(1/flat[ids]).double()
        if torch.linalg.eigvalsh(a.T@a).min()<1e-5:outputs.append(d);continue
        weight=torch.ones_like(b)
        for _ in range(12):
            # Small normal system with float64; normalized coordinates and rank check.
            coef=torch.linalg.solve((a*weight[:,None]).T@a+torch.eye(3,device=a.device,dtype=a.dtype)*1e-8,(a*weight[:,None]).T@b)
            residual=(a@coef-b).abs();scale=residual.median().clamp_min(1e-6)*1.4826
            weight=(1.345*scale/residual.clamp_min(1e-8)).clamp_max(1)
        inv=A@coef;plane=(1/inv.clamp_min(1e-8)).float().reshape_as(d)
        good=torch.isfinite(plane)&(plane>=.1)&(plane<=10)&(d>0)
        outputs.append(torch.where(good,plane,d))
    return torch.stack(outputs)

class DistanceBlendPipeline(torch.nn.Module):
    """Blend frozen predictors using a distance estimated from their own outputs."""
    def __init__(self,near_pipeline,far_pipeline,low_m=1.,high_m=2.):
        super().__init__()
        if not 0<low_m<high_m:raise ValueError('Invalid transition limits')
        self.near=near_pipeline;self.far=far_pipeline;self.low_m=low_m;self.high_m=high_m
    def forward(self,rgb,raw):
        near=self.near(rgb,raw);far=self.far(rgb,raw)
        a=near['depth_m'];b=far['depth_m'];mask=near['mask_probability']
        support=(mask>=.5)&torch.isfinite(a)&(a>0)
        z=torch.nanmedian(torch.where(support,a,float('nan')).flatten(1),dim=1).values
        # No usable near prediction: preserve the near pipeline's abstention.
        weight=((z-self.low_m)/(self.high_m-self.low_m)).clamp(0,1)
        weight=torch.nan_to_num(weight,nan=0.)[:,None,None,None]
        valid_a=torch.isfinite(a)&(a>0);valid_b=torch.isfinite(b)&(b>0)
        output=(1-weight)*a+weight*b
        # Preserve valid near output; never fill locations missing in the near route.
        output=torch.where(valid_a&valid_b,output,torch.where(valid_a,a,0))
        result=dict(near);result['depth_m']=output;result['estimated_distance_m']=z
        result['far_weight']=weight[:,0,0,0]
        return result
