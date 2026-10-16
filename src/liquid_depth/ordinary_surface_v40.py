"""Opt-in diffuse flat-liquid route. No 10-m software cutoff or truth inputs.

Prepared installation geometry and ONE dry depth frame. Denoising is in disparity
space, inside a conservatively eroded vessel ROI; holes are never made valid.
No liquid-height clipping, zero snap, echo-layer hypothesis or output quality gate.
"""
from dataclasses import dataclass
import hashlib
import cv2
import numpy as np
from .ordinary_surface_v35 import interior_ray_roi
from .ordinary_surface_v39 import _installation


@dataclass
class DiffuseContext:
    shape: tuple
    digest: str
    roi: np.ndarray
    rays_cv: np.ndarray
    cosine: np.ndarray
    fb: float
    bottom: float
    pose: np.ndarray


@dataclass
class DiffuseReference:
    digest: str
    residual: np.ndarray
    valid: np.ndarray
    sigma: float
    observed_pixels: int


def prepare(cal, rim, shape, baseline_m):
    if baseline_m<=0 or not np.isfinite(baseline_m):raise ValueError('positive finite baseline required')
    bottom=float(cal.camera_to_world_cv[2,3]-cal.bottom_z_m)
    if bottom<=0:raise ValueError('camera must be above calibrated bottom')
    roi=interior_ray_roi(cal,rim,shape,.65)
    roi=cv2.erode(roi.astype('uint8'),np.ones((5,5),'uint8')).astype(bool)
    y,x=np.indices(shape,dtype=np.float32);k=cal.intrinsics
    rays=np.stack(((x-k[0,2])/k[0,0],(y-k[1,2])/k[1,1],np.ones(shape)),axis=-1)
    cosine=-(rays@cal.camera_to_world_cv[:3,:3].T)[...,2]
    digest=hashlib.sha256((_installation(cal,rim)+repr((tuple(shape),float(baseline_m)))).encode()).hexdigest()
    return DiffuseContext(tuple(shape),digest,roi,rays,
        cosine,float(k[0,0]*baseline_m),bottom,cal.camera_to_world_cv.copy())


def disparity(raw,ctx):
    if tuple(raw.shape)!=ctx.shape:raise ValueError('installation shape mismatch')
    valid=ctx.roi&np.isfinite(raw)&(raw>0)&(ctx.cosine>0)
    d=np.zeros(ctx.shape,np.float32)
    np.divide(ctx.fb,raw,out=d,where=valid)
    return d,valid


def sigma_from_neighbors(d,valid):
    good=valid[:,1:]&valid[:,:-1]
    delta=np.diff(d,axis=1)[good]
    if len(delta)<24:return .18
    # Adjacent differences mostly cancel smooth/persistent common noise.
    return max(.025,float(np.median(abs(delta-np.median(delta)))/(.67448975*np.sqrt(2))))


def robust_plane(d,valid,ctx,sigma):
    y,x=np.nonzero(valid)
    if len(x)<24:return None
    # All supported pixels, not a raster-strided sparse alias of spatial noise.
    a=ctx.rays_cv[y,x].astype(float);v=d[y,x].astype(float)
    p=np.linalg.lstsq(a,v,rcond=None)[0]
    for _ in range(3):
        res=v-a@p
        w=np.minimum(1.,2.5*max(sigma,.025)/np.maximum(abs(res),1e-9))
        p=np.linalg.solve(a.T@(a*w[:,None]),a.T@(w*v))
    return p


def smooth_residual(d,valid,ctx,sigma,kernel=17):
    p=robust_plane(d,valid,ctx,sigma)
    if p is None:return d.copy(),valid.copy(),0
    trend=np.einsum('ijk,k->ij',ctx.rays_cv,p).astype(np.float32)
    residual=d-trend
    keep=valid&(abs(residual)<=3.5*max(sigma,.025))
    count=cv2.boxFilter(keep.astype(np.float32),-1,(kernel,kernel),normalize=False,borderType=cv2.BORDER_CONSTANT)
    total=cv2.boxFilter(np.where(keep,residual,0).astype(np.float32),-1,(kernel,kernel),normalize=False,borderType=cv2.BORDER_CONSTANT)
    filtered=trend+total/np.maximum(count,1.)
    # Preserve original validity: never fill holes or bridge the vessel boundary.
    enough=valid&(count>=12)
    return np.where(enough,filtered,d).astype(np.float32),valid.copy(),int(np.count_nonzero(valid&~keep))


def calibrate(raw,ctx):
    d,valid=disparity(raw,ctx)
    if valid.sum()<24:raise ValueError('insufficient independently observed dry depth')
    sigma=sigma_from_neighbors(d,valid)
    filtered,_,_=smooth_residual(d,valid,ctx,sigma)
    residual=filtered-ctx.fb*ctx.cosine/ctx.bottom
    return DiffuseReference(ctx.digest,residual.astype(np.float32),valid,sigma,int(valid.sum()))


def estimate(raw,ctx,reference=None,mode='paired'):
    if mode not in ('dense','denoise','scalar','paired'):raise ValueError('unknown mode')
    if reference is not None and reference.digest!=ctx.digest:raise ValueError('reference installation mismatch')
    d,valid=disparity(raw,ctx)
    out=dict(statistics=None,qualified=False,reasons=['diffuse_flat_fixed_installation_only','research_not_hardware_qualified'],route='ordinary_diffuse_v40_'+mode)
    if valid.sum()<24:
        out['reasons'].append('insufficient_current_depth');return out
    sigma=sigma_from_neighbors(d,valid);alpha=0.;discarded=0
    if mode!='dense':d,valid,discarded=smooth_residual(d,valid,ctx,sigma)
    if mode in ('scalar','paired'):
        if reference is None:raise ValueError('independent dry reference required')
        common=valid&reference.valid
        if common.sum()>=24:
            if mode=='paired':
                # Estimate transferable spatial-noise correlation after removing
                # any free plane. Constant level and tilt cannot inflate alpha.
                y,x=np.nonzero(common);a=ctx.rays_cv[y,x].astype(float)
                dry=reference.residual[y,x].astype(float);wet=d[y,x].astype(float)
                dry_hp=dry-a@np.linalg.lstsq(a,dry,rcond=None)[0]
                wet_hp=wet-a@np.linalg.lstsq(a,wet,rcond=None)[0]
                den=float(dry_hp@dry_hp)
                alpha=float(np.clip((dry_hp@wet_hp)/max(den,1e-12),0,1))
            constant=float(np.mean(reference.residual[common]/ctx.cosine[common]))
            correction=(1-alpha)*constant*ctx.cosine+alpha*reference.residual
            d=d.copy();d[common]-=correction[common]
            valid=common  # actual observed dry AND current points, never GT-selected
        else:out['reasons'].append('dry_overlap_insufficient_uncorrected')
    p=robust_plane(d,valid,ctx,sigma)
    if p is None:out['reasons'].append('insufficient_plane_support');return out
    # Horizontal gravity-constrained plane is the reporting model for this route.
    c=ctx.cosine[valid].astype(float);v=d[valid].astype(float)
    s=float(np.dot(c,v)/np.dot(c,c))
    for _ in range(3):
        w=np.minimum(1.,2.5*max(sigma,.025)/np.maximum(abs(v-s*c),1e-9))
        s=float(np.dot(w*c,v)/np.dot(w*c,c))
    if not np.isfinite(s) or s<=0:
        out['reasons'].append('nonpositive_inverse_clearance');return out
    height=ctx.bottom-ctx.fb/s
    n=ctx.pose[:3,:3]@(p/ctx.fb)
    if n[2]<0:n=-n
    angle=float(np.degrees(np.arccos(np.clip(n[2]/np.linalg.norm(n),-1,1))))
    # Free-plane center height: derive before normal reorientation.
    nw=ctx.pose[:3,:3]@(p/ctx.fb)
    free_z=float((1+nw@ctx.pose[:3,3])/nw[2])
    out.update(statistics={k:float(height) for k in ('mean_m','min_m','max_m','p05_m','p95_m')},
        selected_points=int(valid.sum()),sigma_px=sigma,spatial_transfer=alpha,
        free_plane_angle_deg=angle,free_plane_height_m=free_z-(ctx.pose[2,3]-ctx.bottom),
        denoise_downweighted_points=discarded)
    return out
