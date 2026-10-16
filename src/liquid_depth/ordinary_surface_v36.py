"""Quantization-aware ordinary liquid height. Experimental calibrated-sensor route.

Simulation presets are not hardware specifications. Neither liquid height nor
surface truth is an input. Signed liquid height is never clipped or snapped.
"""
from dataclasses import replace
import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize_scalar
from .ordinary_surface_v35 import interior_ray_roi,estimate_ordinary
from .surface_refinement import support,weighted_median,StereoNoiseModel


def normal_mass(a,b):
    return np.maximum(np.where(a>0,ndtr(-a)-ndtr(-b),ndtr(b)-ndtr(a)),0.)


def discrete_clearance(depth,cosine,weights,fx,model,*,bottom_clearance=None,
                       disparity_step=1/16,depth_step=.001,echo=False):
    """Probability of the actual quantized depth cell, conditional on range.

    Range selection occurs on quantized disparity *centres*, so conditioning
    extends half a disparity bin beyond the first/last valid centre.
    Echo mode separately models a calibrated bottom-return atom, not a liquid
    height prior. The uniform term keeps unexpected returns in the likelihood.
    """
    fb=fx*model.baseline_m;q=disparity_step
    min_index=int(np.ceil(fb/model.max_depth_m/q-1e-8))
    max_index=int(np.floor(fb/model.min_depth_m/q+1e-8))
    lower=(min_index-.5)*q;upper=(max_index+.5)*q
    # Account for final metric-depth rounding after disparity quantization.
    ilo=np.maximum(np.ceil(fb/(depth+depth_step*.501)/q-1e-7),min_index)
    ihi=np.minimum(np.floor(fb/np.maximum(depth-depth_step*.501,1e-6)/q+1e-7),max_index)
    possible=ihi>=ilo
    if np.count_nonzero(possible)<24:return None
    left=(ilo-.5)*q;right=(ihi+.5)*q
    weights=weights/weights.sum();initial=weighted_median(depth*cosine,weights)
    floor_mass=np.zeros(len(depth))
    if echo and bottom_clearance is not None:
        expected=np.round(bottom_clearance/cosine/depth_step)*depth_step
        floor_mass=(abs(depth-expected)<depth_step*.501).astype(float)
    def loss(clearance):
        z=clearance/cosine
        sz=model.depth_noise_floor_m+model.depth_noise_quadratic*z*z
        mu=fb/z*(1+(sz/z)**2)
        sd=np.sqrt(model.disparity_sigma_px**2+(fb*sz/z**2)**2)
        norm=normal_mass((lower-mu)/sd,(upper-mu)/sd)
        mass=normal_mass((left-mu)/sd,(right-mu)/sd)/np.maximum(norm,1e-300)
        mass=np.where(possible,mass,0.)
        density=(.945*mass+.05*floor_mass if echo else .995*mass)+.005*depth_step/(model.max_depth_m-model.min_depth_m)
        return float(-np.dot(weights,np.log(np.maximum(density,1e-300))))
    bounds=(max(.02,initial*.6),max(.04,initial*1.5))
    grid=np.unique(np.r_[np.linspace(*bounds,41),initial]);cost=np.array([loss(x) for x in grid]);i=int(np.argmin(cost))
    if i==0 or i==len(grid)-1:return None
    fit=minimize_scalar(loss,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-7})
    return float(fit.x) if fit.success else None


def estimate_v36(rgb,raw,cal,rim_z_m,sensor,*,mode='quant_echo',wide_near=True,uniform=False,
                 noise_model=None,disparity_step=1/16,depth_step=.001):
    if mode not in ('quant','quant_echo') or not np.isfinite([disparity_step,depth_step]).all() or disparity_step<=0 or depth_step<=0:
        raise ValueError('Valid likelihood mode and calibrated quantization steps required')
    if sensor=='tof':return estimate_ordinary(rgb,raw,cal,rim_z_m,sensor,0)
    roi=interior_ray_roi(cal,rim_z_m,raw.shape,.65);fallback=roi.sum()<256
    if fallback:roi=interior_ray_roi(replace(cal,bottom_z_m=rim_z_m-1e-5),rim_z_m,raw.shape,.65)
    data,reason=support(raw,roi,cal.intrinsics,cal.camera_to_world_cv,cal.radii_m,max_points=8192 if uniform else 4096)
    out=dict(statistics=None,qualified=False,reasons=[],route='quantized_ordinary_v36')
    if data is None:out['reasons']=[reason];return out
    points,pixels,depth,weights,_=data
    # More area only when the near-range preliminary measurement and vessel
    # geometry permit it. No true-depth bin or known liquid level is consulted.
    if wide_near and np.median(depth)<3:
        candidate=interior_ray_roi(cal,rim_z_m,raw.shape,.9)
        if candidate.sum()<256:candidate=interior_ray_roi(replace(cal,bottom_z_m=rim_z_m-1e-5),rim_z_m,raw.shape,.9)
        data2,reason2=support(raw,candidate,cal.intrinsics,cal.camera_to_world_cv,cal.radii_m,max_points=8192)
        if data2 is not None:
            points,pixels,depth,weights,_=data2;out['reasons'].append('near_range_wider_geometry_support')
    k=cal.intrinsics;t=cal.camera_to_world_cv
    original_effective=float(1/np.sum(weights**2))
    if uniform:
        # Equal angular tiles can give a few border pixels the same weight as
        # hundreds in a central tile. For this flat-surface-only route, equal
        # independent observations reduce variance without a wave-area bias.
        weights=np.full(len(depth),1/len(depth))
    rays=np.column_stack(((pixels[:,0]-k[0,2])/k[0,0],(pixels[:,1]-k[1,2])/k[1,1],np.ones(len(depth))))
    cosine=-(rays@t[:3,:3].T)[:,2]
    model=noise_model if noise_model is not None else StereoNoiseModel.simulation_proxy(sensor)
    clearance=discrete_clearance(depth,cosine,weights,k[0,0],model,
        bottom_clearance=t[2,3]-cal.bottom_z_m,echo=mode=='quant_echo',
        disparity_step=disparity_step,depth_step=depth_step)
    if clearance is None:out['reasons'].append('quantized_likelihood_unobservable');return out
    h=float(t[2,3]-cal.bottom_z_m-clearance)
    out.update(statistics={k:h for k in ('mean_m','min_m','max_m','p05_m','p95_m')},
        selected_points=len(depth),effective_points_before_uniform=original_effective,
        effective_points=float(1/np.sum(weights**2)))
    out['reasons']+=['requires_calibrated_raw_disparity_and_depth_quantization','systematic_echo_not_independently_verified']
    return out
