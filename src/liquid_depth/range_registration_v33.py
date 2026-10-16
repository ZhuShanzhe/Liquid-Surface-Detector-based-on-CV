"""Research image registration: resolution controls and symmetric dense fits.

Only observed images, a fixed operator ROI and camera geometry are inputs.
Interpolated/SR pixels are explicitly correlated, never independent evidence.
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt,spline_filter,map_coordinates
from scipy.optimize import least_squares
from . import finite_relative_v30 as base
from . import range_relative_v31 as old

OPTIONS={
 'baseline_v32':None,
 'bicubic_x2':dict(mode='forward',factor=2,samples=2600,core=.30,sigma=3.,order=3),
 'swinir_x2':dict(mode='forward',factor=2,samples=2600,core=.30,sigma=3.,order=3),
 'native_dense':dict(mode='forward',factor=1,samples=2000,core=.30,sigma=3.,order=3),
 'midpoint_650':dict(mode='midpoint',factor=1,samples=650,core=.36,sigma=3.,order=3),
 'midpoint_dense':dict(mode='midpoint',factor=1,samples=2000,core=.36,sigma=3.,order=3),
 'midpoint_sigma2':dict(mode='midpoint',factor=1,samples=2000,core=.36,sigma=2.,order=3),
 'midpoint_sigma4':dict(mode='midpoint',factor=1,samples=2000,core=.36,sigma=4.,order=3),
 'midpoint_spline5':dict(mode='midpoint',factor=1,samples=2000,core=.36,sigma=3.,order=5),
 'midpoint_core30':dict(mode='midpoint',factor=1,samples=2000,core=.30,sigma=3.,order=3),
 'midpoint_bicubic2':dict(mode='midpoint',factor=2,samples=2000,core=.36,sigma=3.,order=3),
 'midpoint_swinir2':dict(mode='midpoint',factor=2,samples=2000,core=.36,sigma=3.,order=3),
}

def upscale_geometry(cal,factor):
    out=dict(cal)
    for k in ('cx','cy'):out[k]=(cal[k]+.5)*factor-.5
    out['fx']=cal['fx']*factor
    for k in ('width','height'):
        if k in cal:out[k]=cal[k]*factor
    return out

def native_coordinates(points,factor):return (np.asarray(points)+.5)/factor-.5

def _one(a,b,mask,opt):
    factor=opt['factor'];support=base.eroded(mask,3*factor)
    yy,xx=np.where(support)
    if len(xx)<60:return None
    centre=np.array([xx.mean(),yy.mean()])
    use=(abs(xx-centre[0])<opt['core']*np.ptp(xx))&(abs(yy-centre[1])<opt['core']*np.ptp(yy))
    xx,yy=xx[use],yy[use]
    # Even spatial sampling avoids aliasing a deterministic raster stride.
    take=np.linspace(0,len(xx)-1,min(len(xx),opt['samples'])).astype(int)
    xx,yy=xx[take].astype(float),yy[take].astype(float)
    if len(xx)<30:return None
    nearest=distance_transform_edt(mask==0,return_distances=False,return_indices=True)
    aa=cv2.GaussianBlur(a[tuple(nearest)].astype(float)/255,(0,0),opt['sigma']*factor)
    bb=cv2.GaussianBlur(b[tuple(nearest)].astype(float)/255,(0,0),opt['sigma']*factor)
    ref=aa[yy.astype(int),xx.astype(int)]
    if ref.std()<.003:return None
    order=opt['order'];target=spline_filter(bb,order=order)
    anchor=spline_filter(aa,order=order) if opt['mode']=='midpoint' else None
    p0=np.c_[xx,yy];delta=p0-centre
    def sample(im,p):return map_coordinates(im,[p[:,1],p[:,0]],order=order,prefilter=False,mode='nearest')
    def values(v):
        if anchor is None:
            p=p0;q=centre+delta*np.exp(v[0])+v[1:]
            r=ref-ref.mean()
        else:
            p=centre+delta*np.exp(-v[0]/2)-v[1:]/2
            q=centre+delta*np.exp(v[0]/2)+v[1:]/2
            r=sample(anchor,p);r=r-r.mean()
        z=sample(target,q);z=z-z.mean()
        return p,q,r,z
    def residual(v):
        p,q,r,z=values(v)
        if anchor is None:
            gain=max(0.,float(z@r/max(z@z,1e-12)))
            return gain*z-r
        # Both images have noise: neither is treated as an exact intensity target.
        # Unit-variance symmetric residual removes exposure gain and offset.
        return .05*(z/max(float(z.std()),1e-6)-r/max(float(r.std()),1e-6))
    starts=[np.array([s,0.,0.]) for s in np.linspace(-.48,.48,25)]
    starts.sort(key=lambda s:float(np.mean(residual(s)**2)))
    fits=[]
    for start in starts[:2]:
        fit=least_squares(residual,start,bounds=([-.5,-5*factor,-5*factor],[.5,5*factor,5*factor]),
             loss='soft_l1',f_scale=.012,max_nfev=65,diff_step=1e-4,xtol=1e-9,ftol=1e-9,gtol=1e-9)
        if np.isfinite(fit.x).all():fits.append(fit)
    if not fits:return None
    fit=min(fits,key=lambda f:np.mean(residual(f.x)**2));p,q,r,z=values(fit.x)
    valid=np.ones(len(p),bool)
    for pts in (p,q):
        pi=np.round(pts).astype(int)
        inside=(pi[:,0]>=0)&(pi[:,0]<mask.shape[1])&(pi[:,1]>=0)&(pi[:,1]<mask.shape[0])
        ix=np.where(inside)[0];inside[ix]&=support[pi[ix,1],pi[ix,0]]
        valid &= inside
    corr=float(z@r/max(np.linalg.norm(z)*np.linalg.norm(r),1e-12))
    return p[valid],q[valid],dict(correlation=corr,photometric_rms=float(np.sqrt(np.mean(residual(fit.x)**2))),
         log_scale=float(fit.x[0]),sample_count=int(valid.sum()),mode=opt['mode'])

def pair(a,b,cal,mask,channel,name):
    if name=='baseline_v32':return old.wide_pair(a,b,cal,mask,channel)
    opt=OPTIONS[name];first=_one(a,b,mask,opt)
    if opt['mode']=='midpoint':chosen=first
    else:
        second=_one(b,a,mask,opt)
        if first is None:chosen=None if second is None else (second[1],second[0],second[2])
        elif second is None or first[2]['correlation']>=second[2]['correlation']:chosen=first
        else:chosen=(second[1],second[0],second[2])
    if chosen is None:return base.package([],[],cal,channel)
    p,q,e=chosen
    return base.package(native_coordinates(p,opt['factor']),native_coordinates(q,opt['factor']),cal,channel,
                        dict(e,method=name,correlated_dense_samples=True,independent_samples_claimed=False))

class SwinIRX2:
    """Official frozen PSNR-oriented lightweight model, no GAN fine-tuning."""
    def __init__(self,root='/root/autodl-tmp/liquid-depth-artifacts/pretrained/swinir_v33'):
        from pathlib import Path
        import importlib.util
        import torch
        self.torch=torch;root=Path(root)
        spec=importlib.util.spec_from_file_location('official_swinir_v33',root/'network_swinir.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.model=module.SwinIR(upscale=2,in_chans=3,img_size=64,window_size=8,img_range=1.,
             depths=[6]*4,embed_dim=60,num_heads=[6]*4,mlp_ratio=2,upsampler='pixelshuffledirect',resi_connection='1conv')
        weights=torch.load(root/'swinir_x2.pth',map_location='cpu',weights_only=True)
        self.model.load_state_dict(weights['params'] if 'params' in weights else weights,strict=True)
        self.model.eval().cuda()
    def __call__(self,images):
        torch=self.torch;stack=np.stack(images).astype(np.float32)/255
        x=torch.from_numpy(stack[:,None].repeat(3,axis=1)).cuda()
        h,w=x.shape[-2:];ph=(-h)%8;pw=(-w)%8
        x=torch.nn.functional.pad(x,(0,pw,0,ph),mode='reflect')
        with torch.inference_mode():y=self.model(x)[...,:h*2,:w*2].mean(1)
        # Preserve float precision; clamp image intensities, never liquid heights.
        return (y.clamp(0,1)*255).cpu().numpy()
