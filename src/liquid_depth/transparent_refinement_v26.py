"""Bounded correction and periodic pixel-bias calibration for target-only research."""
import numpy as np
from scipy.optimize import least_squares,minimize_scalar
from . import transparent_diagnostic_v25 as base


def bounded_height(raw,prior,cal,shrink=1.,slope_cap=.2,tail='constant'):
    """Piecewise residual with a bounded derivative; tail never learns a secant slope."""
    x=np.asarray(prior['x']);delta=np.asarray(prior['y'])-x
    slopes=np.clip(np.diff(delta)/np.diff(x),-slope_cap,slope_cap)
    reconstructed=np.r_[0.,np.cumsum(slopes*np.diff(x))]
    reconstructed+=float(np.mean(delta-reconstructed))
    correction=float(np.interp(raw,x,reconstructed))
    distance=max(x[0]-raw,raw-x[-1],0.)
    if tail=='taper':correction*=max(0.,1-distance/max(.08*cal['bottom_m'],1e-9))
    elif tail=='zero' and distance>0:correction=0.
    elif tail!='constant' and tail!='zero':raise ValueError('unsupported_tail')
    return float(raw+shrink*correction)


def periodic_basis(obs,harmonics):
    p=np.asarray(obs['right_pixels'])[:,0]
    return np.column_stack([f(2*np.pi*k*p) for k in range(1,harmonics+1) for f in (np.sin,np.cos)])


def pixel_residual(h,index,bias,coefficients,obs,cal):
    residual=base.pixel_residual(h,index,bias,obs,cal)
    if len(coefficients):residual[:,0]-=periodic_basis(obs,len(coefficients)//2)@coefficients
    return residual


def solve_height(obs,prior,cal):
    if not base.usable(obs):return dict(height_m=None,finite=False,reasons=['insufficient_or_invalid_matches'])
    coefficients=np.asarray(prior.get('pixel_bias_coefficients',[]))
    weights=np.maximum(obs['weights'],.02)
    def objective(h):
        r=pixel_residual(h,prior['index'],prior['bias_px'],coefficients,obs,cal)
        return float(np.average((np.sqrt(1+(r/.03)**2)-1).sum(1),weights=weights))
    fit=minimize_scalar(objective,bounds=(0.,.98*cal['bottom_m']),method='bounded',
                        options={'xatol':max(1e-9,cal['bottom_m']*1e-8)})
    reasons=[]
    if not fit.success:reasons.append('optimizer_failed')
    h=float(fit.x)
    if h<1e-5*cal['bottom_m'] or h>(.98-1e-5)*cal['bottom_m']:reasons.append('geometry_boundary')
    residual=pixel_residual(h,prior['index'],prior['bias_px'],coefficients,obs,cal)
    return dict(height_m=h,finite=bool(np.isfinite(h)),reasons=reasons,regions=obs['regions'],
                residual_px=float(np.median(np.linalg.norm(residual,axis=1))))


def fit_prior(observations,heights,cal,condition,harmonics=0,ridge=.02):
    base.require_target(condition)
    if harmonics==0:return base.fit_prior(observations,heights,cal,condition,'none')
    hs=np.asarray(heights,float)
    if len(hs)!=len(observations) or len(hs)<3:raise ValueError('at_least_three_known_levels')
    if not np.isfinite(hs).all() or min(hs)<=0 or max(hs)>=cal['bottom_m']:raise ValueError('invalid_known_levels')
    ids=[i for i,o in enumerate(observations) if base.usable(o)]
    if len(ids)<3:raise ValueError('fewer_than_three_usable_levels')
    obs=[observations[i] for i in ids];hs=hs[ids]
    if np.ptp(hs)<max(.03,.15*max(hs)):raise ValueError('insufficient_level_span')
    def objective(p):
        values=[(pixel_residual(h,p[0],p[1],p[2:],o,cal)*np.sqrt(np.maximum(o['weights'],.02))[:,None]/np.sqrt(len(o['weights']))).ravel()
                for h,o in zip(hs,obs)]
        return np.r_[np.concatenate(values),np.sqrt(ridge)*p[2:]]
    count=2*harmonics
    fit=least_squares(objective,np.r_[1.37,0.,np.zeros(count)],
        bounds=(np.r_[1.2,-2.,np.full(count,-.15)],np.r_[1.6,2.,np.full(count,.15)]),
        loss='soft_l1',f_scale=.005,max_nfev=180)
    if not fit.success:raise ValueError('prior_optimizer_failed')
    prior=dict(index=float(fit.x[0]),bias_px=float(fit.x[1]),pixel_bias_coefficients=fit.x[2:].tolist(),
        condition=condition,correction='none',used_frames=ids,research_only=True,production_enabled=False)
    x=np.array([solve_height(o,prior,cal)['height_m'] for o in obs]);order=np.argsort(x)
    if np.any(np.diff(hs[order])<=0) or np.min(np.diff(x[order]))<=1e-9:raise ValueError('nonmonotonic_calibration')
    prior.update(x=x[order].tolist(),y=hs[order].tolist(),prior_residual_m=(x-hs).tolist())
    return prior


def infer(obs,prior,cal,correction):
    base.require_target(prior['condition'])
    r=solve_height(obs,prior,cal)
    if not r['finite']:return dict(r,diagnostic_only=True,qualified=False)
    raw=r['height_m']
    value=raw if correction['kind']=='none' else bounded_height(raw,prior,cal,
        correction['shrink'],correction['slope_cap'],correction['tail'])
    if raw<min(prior['x']) or raw>max(prior['x']):r['reasons'].append('outside_prior_span')
    if value<=0 or value>=cal['bottom_m']:r['reasons'].append('outside_physical_height')
    return dict(r,height_m=value,uncorrected_height_m=raw,diagnostic_only=True,qualified=False,
                confidence_gate_enabled=False,coverage_gate_enabled=False)
