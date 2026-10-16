"""Five-prior reference-specific response calibration; research only.

Known heights are calibration inputs. Unknown height, index, extinction and
test labels are never consumed. No zero snapping, capacity clipping or gates.
"""
import numpy as np
from functools import lru_cache
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from . import range_relative_v31 as old
from . import finite_relative_v30 as base


@lru_cache(maxsize=1024)
def _pchip(x,y):
    return PchipInterpolator(x,y,extrapolate=False)


def inverse_curve(s,x,y,shape='pchip'):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if shape=='linear':return old._inverse(s,x,y)
    f=_pchip(tuple(x),tuple(y))
    if x[0]<=s<=x[-1]:return float(f(s))
    # Extrapolate with a positive endpoint tangent; do not clip the output.
    j=0 if s<x[0] else -1
    tangent=float(f.derivative()(x[j]))
    secant=(y[1]-y[0])/(x[1]-x[0]) if j==0 else (y[-1]-y[-2])/(x[-1]-x[-2])
    tangent=float(np.clip(tangent,.25*secant,4*secant))
    return float(y[j]+(s-x[j])*tangent)


def fit_local(pairs,heights,cal,condition,shape='pchip'):
    base.old.v25.require_target(condition);hs=base.old.validate_heights(heights,cal)
    observations={};channels=set()
    for a,b,p in pairs:
        if not base.valid(p):continue
        s=base.observed_log_scale(p)
        if s is None:continue
        ch=p['channel'];channels.add(ch)
        observations[a,b,ch]=(s,old.quality(p))
        observations[b,a,ch]=(-s,old.quality(p))
    curves={}
    for a in range(5):
        for ch in sorted(channels):
            nodes=[b for b in range(5) if b==a or (a,b,ch) in observations]
            if len(nodes)<3:continue
            hh=hs[nodes];anchor=nodes.index(a)
            ss=np.array([0. if b==a else observations[a,b,ch][0] for b in nodes])
            ww=np.array([1. if b==a else observations[a,b,ch][1] for b in nodes])
            lower=.005/cal['bottom_m']*np.diff(hh)
            # Self-image identity is an exact coordinate origin, not a rule
            # that forces an unknown measurement to zero.
            def response(d):
                z=np.r_[0.,np.cumsum(d)];return z-z[anchor]
            if np.all(np.diff(ss)>lower):z=ss
            else:
                initial=np.maximum(np.diff(ss),lower*1.01)
                result=least_squares(lambda d:(response(d)-ss)*np.sqrt(ww),initial,
                    bounds=(lower,np.inf),loss='soft_l1',f_scale=.0005,max_nfev=100,
                    xtol=1e-10,ftol=1e-10,gtol=1e-10)
                if not result.success:continue
                z=response(result.x)
            curves[f'{a}:{ch}']=dict(scale=z.tolist(),heights=hh.tolist(),anchor=a,channel=ch,
                calibration_residual=float(np.sqrt(np.mean((z-ss)**2))))
    if not curves:raise ValueError('no_reference_specific_calibration')
    return dict(curves=curves,heights=hs.tolist(),condition=condition,shape=shape,
        mode='reference_specific',physical_index_identified=False,research_only=True,production_enabled=False)


def _fuse(values,weights,kind):
    x=np.asarray(values);w=np.asarray(weights)
    if kind=='median':return old._weighted_median(x,w)
    if kind=='mean':return float(np.sum(x*w)/np.sum(w))
    # Soft robust aggregation retains every positive-weight observation.
    location=float(np.sum(x*w)/np.sum(w))
    for _ in range(8):
        residual=x-location
        scale=max(.003,1.4826*old._weighted_median(abs(residual),w))
        rw=w/np.sqrt(1+(residual/(2*scale))**2)
        location=float(np.sum(x*rw)/np.sum(rw))
    return location


def infer(groups,prior,cal,fusion='robust',band=None):
    values=[];weights=[];sources=[]
    for a,p in groups:
        if not base.valid(p):continue
        s=base.observed_log_scale(p)
        if s is None:continue
        if prior['mode']=='reference_specific':
            key=f"{a}:{p['channel']}"
            if key not in prior['curves']:continue
            curve=prior['curves'][key]
            h=inverse_curve(s,curve['scale'],curve['heights'],prior['shape'])
        else:
            mapping=dict(zip(prior['nodes'],prior['latent']))
            if a not in mapping:continue
            h=inverse_curve(mapping[a]+s,prior['latent'],prior['node_heights'],prior['shape'])
        w=old.quality(p)
        if band is not None:w*=1e-6+np.exp(-.5*(s/band)**2)
        if np.isfinite(h):values.append(h);weights.append(w);sources.append(a)
    if not values:return dict(height_m=None,reasons=['no_usable_finite_bottom_observations'],qualified=False)
    h=_fuse(values,weights,fusion);reasons=[]
    if h<0 or h>cal['container_max_height_m']:reasons.append('outside_container_capacity')
    if h<prior['heights'][0] or h>prior['heights'][-1]:reasons.append('outside_known_prior_span')
    return dict(height_m=float(h),reasons=reasons,qualified=False,diagnostic_only=True,
        confidence_gate_enabled=False,coverage_gate_enabled=False,capacity_clipping=False,zero_snapping=False,
        track_groups=len(values),height_dispersion_m=float(np.std(values)),
        min_reference_height_m=float(min(values)),max_reference_height_m=float(max(values)),
        effective_reference_groups=float(np.sum(weights)**2/np.sum(np.square(weights))))


def fit(pairs,heights,cal,condition,options):
    if options['calibration']=='local':
        return fit_local(pairs,heights,cal,condition,options['shape'])
    prior=old.fit_curve(pairs,heights,cal,condition)
    prior.update(mode='global',shape=options['shape'])
    return prior
