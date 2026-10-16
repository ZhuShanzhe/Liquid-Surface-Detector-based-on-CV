"""Five-known-level calibration of residual pixel/phase bias; research only."""
import numpy as np
from scipy.optimize import least_squares,minimize_scalar
from . import transparent_diagnostic_v25 as base
from . import transparent_refinement_v26 as v26

def bias_basis(obs,setting,axis=0):
    count=int(setting.get('harmonics',0));period=float(setting.get('period_px',1.))
    if period<=0 or count<0 or count>4:raise ValueError('invalid_bias_basis')
    phase=np.asarray(obs['right_pixels'])[:,axis]*2*np.pi/period
    terms=[f(k*phase) for k in range(1,count+1) for f in (np.sin,np.cos)]
    if setting.get('spatial',False):
        q=np.asarray(obs['left_q']);x=q[:,0]/.5;y=q[:,1]/.35
        terms.extend([x,y,x*x+y*y-.5])
    if axis==1:terms.insert(0,np.ones(len(phase)))
    return np.column_stack(terms) if terms else np.empty((len(phase),0))

def solve_height(obs,prior,cal):
    if not base.usable(obs):return dict(height_m=None,finite=False,reasons=['insufficient_or_invalid_matches'])
    setting=prior['setting'];cx=prior['coeff_x'];cy=prior['coeff_y']
    bx=bias_basis(obs,setting,0);by=bias_basis(obs,setting,1)
    xshift=bx@cx if len(cx) else np.zeros(obs['regions'])
    yshift=by@cy if len(cy) else np.zeros(obs['regions'])
    weights=np.maximum(obs['weights'],.02)
    def errors(h):return base.pixel_residual(h,prior['index'],prior['bias_px'],obs,cal)-np.column_stack((xshift,yshift))
    def objective(h):return float(np.average((np.sqrt(1+(errors(h)/.03)**2)-1).sum(1),weights=weights))
    fit=minimize_scalar(objective,bounds=(0.,.98*cal['bottom_m']),method='bounded',
        options={'xatol':max(1e-9,cal['bottom_m']*1e-8)})
    value=float(fit.x);warnings=[]
    if not fit.success:warnings.append('optimizer_failed')
    if value<1e-5*cal['bottom_m'] or value>(.98-1e-5)*cal['bottom_m']:warnings.append('geometry_boundary')
    return dict(height_m=value,finite=bool(np.isfinite(value)),reasons=warnings,regions=obs['regions'],
        residual_px=float(np.median(np.linalg.norm(errors(value),axis=1))))

def fit_prior(observations,heights,cal,condition,setting):
    base.require_target(condition)
    if len(observations)!=5 or len(heights)!=5:raise ValueError('exactly_five_known_levels')
    hs=np.asarray(heights,float)
    if not np.isfinite(hs).all() or min(hs)<=0 or max(hs)>=cal['bottom_m']:raise ValueError('invalid_known_levels')
    ids=[i for i,o in enumerate(observations) if base.usable(o)]
    if len(ids)<3:raise ValueError('fewer_than_three_usable_levels')
    obs=[observations[i] for i in ids];hs=hs[ids]
    if np.ptp(hs)<max(.03,.15*max(hs)):raise ValueError('insufficient_level_span')
    bx=[bias_basis(o,setting,0) for o in obs];by=[bias_basis(o,setting,1) for o in obs]
    nx=bx[0].shape[1];ny=by[0].shape[1] if setting.get('fit_y',False) else 0
    ridge=float(setting.get('ridge',.1))
    if ridge<0:raise ValueError('invalid_ridge')
    def objective(p):
        values=[]
        for i,(o,h) in enumerate(zip(obs,hs)):
            r=base.pixel_residual(h,p[0],p[1],o,cal)
            if nx:r[:,0]-=bx[i]@p[2:2+nx]
            if ny:r[:,1]-=by[i]@p[2+nx:]
            values.append((r*np.sqrt(np.maximum(o['weights'],.02))[:,None]/np.sqrt(len(o['weights']))).ravel())
        return np.r_[np.concatenate(values),np.sqrt(ridge)*p[2:]]
    count=nx+ny
    fit=least_squares(objective,np.r_[1.37,0.,np.zeros(count)],
        bounds=(np.r_[1.2,-2.,np.full(count,-.15)],np.r_[1.6,2.,np.full(count,.15)]),
        loss='soft_l1',f_scale=.005,max_nfev=240)
    if not fit.success:raise ValueError('prior_optimizer_failed')
    singular=np.linalg.svd(fit.jac,compute_uv=False)
    prior=dict(index=float(fit.x[0]),bias_px=float(fit.x[1]),coeff_x=fit.x[2:2+nx].tolist(),
        coeff_y=fit.x[2+nx:].tolist(),setting=dict(setting),condition=condition,used_frames=ids,
        parameter_condition_number=float(singular[0]/max(singular[-1],1e-12)),
        research_only=True,production_enabled=False,calibration_budget=5)
    x=np.array([solve_height(o,prior,cal)['height_m'] for o in obs]);order=np.argsort(x)
    if np.any(np.diff(hs[order])<=0) or np.min(np.diff(x[order]))<=1e-9:raise ValueError('nonmonotonic_calibration')
    prior.update(x=x[order].tolist(),y=hs[order].tolist(),prior_residual_m=(x-hs).tolist())
    return prior

def infer(obs,prior,cal,correction):
    base.require_target(prior['condition']);r=solve_height(obs,prior,cal)
    if not r['finite']:return dict(r,diagnostic_only=True,qualified=False)
    raw=r['height_m'];kind=correction['kind']
    if kind=='none':value=raw
    elif kind=='bounded':value=v26.bounded_height(raw,prior,cal,correction['shrink'],correction['slope_cap'],correction['tail'])
    elif kind=='affine':value=base._map(raw,np.array(prior['x']),np.array(prior['y']),'affine')
    else:raise ValueError('unsupported_correction')
    if raw<min(prior['x']) or raw>max(prior['x']):r['reasons'].append('outside_prior_span')
    if value<=0 or value>=cal['bottom_m']:r['reasons'].append('outside_physical_height')
    return dict(r,height_m=float(value),uncorrected_height_m=raw,diagnostic_only=True,qualified=False,
        confidence_gate_enabled=False,coverage_gate_enabled=False)
