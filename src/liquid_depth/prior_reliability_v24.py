"""Observable liquid-height calibration from matched refractive bottom rays.

The effective optical coordinate need not identify a unique physical IOR.
Only supplied known-level frames fit the mapping; no current frame truth is used.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator


def optical_coordinate(obs,cal):
    """Robust linear Snell height at a fixed reference index, without clipping."""
    if obs['regions']<20:raise ValueError('insufficient_matched_regions')
    ql=np.asarray(obs['left_q']);qr=np.asarray(obs['right_q']);n=1.37
    a=lambda q:np.sqrt(n*n+(n*n-1)*np.sum(q*q,axis=1))
    slope=ql*(1/a(ql)-1)[:,None]-qr*(1/a(qr)-1)[:,None]
    intercept=(ql-qr)*cal['bottom_m']-np.array([cal['baseline_m'],0.])
    denom=np.sum(slope*slope,axis=1)
    height=-np.sum(slope*intercept,axis=1)/np.maximum(denom,1e-15)
    valid=np.isfinite(height)&(denom>1e-12)
    if valid.sum()<20:raise ValueError('unobservable_geometry')
    weights=np.maximum(np.asarray(obs['weights'])[valid],.01)
    hh=height[valid];cells=np.floor(np.asarray(obs['left_pixels'])[valid]/64).astype(int)
    # Aggregate spatial blocks first: more pixels do not imply independence.
    block=[];block_weights=[]
    for cell in np.unique(cells,axis=0):
        ids=(cells==cell).all(1)
        if ids.sum()<2:continue
        values=hh[ids];w=weights[ids];order=np.argsort(values)
        median=values[order[np.searchsorted(np.cumsum(w[order]),w.sum()/2)]]
        block.append(median);block_weights.append(np.mean(w))
    if len(block)<6:raise ValueError('insufficient_spatial_blocks')
    block=np.asarray(block);scale=max(.0002*cal['bottom_m'],1.4826*np.median(abs(block-np.median(block))))
    result=least_squares(lambda p:(block-p[0])/scale,[np.median(block)],loss='soft_l1',f_scale=1.)
    value=float(result.x[0]);scatter=float(1.4826*np.median(abs(block-value)))
    return dict(value=value,noise=max(.00005*cal['bottom_m'],scatter/np.sqrt(len(block))),
        block_count=len(block),scatter=scatter,regions=obs['regions'])


def _line(x,y):
    x=np.asarray(x);y=np.asarray(y)
    slope,offset=np.polyfit(x,y,1)
    return np.array([slope,offset])


def _map(x,xp,yp,mode):
    if mode=='affine':return float(np.polyval(_line(xp,yp),x))
    if x<xp[0]:return float(yp[0]+(x-xp[0])*(yp[1]-yp[0])/(xp[1]-xp[0]))
    if x>xp[-1]:return float(yp[-1]+(x-xp[-1])*(yp[-1]-yp[-2])/(xp[-1]-xp[-2]))
    if mode=='pchip':return float(PchipInterpolator(xp,yp)(x))
    return float(np.interp(x,xp,yp))


def fit_prior(observations,heights,cal,mode='affine'):
    if len(observations)!=len(heights) or len(heights)<3:raise ValueError('at_least_three_known_levels')
    hs=np.asarray(heights,float)
    if not np.isfinite(hs).all() or min(hs)<=0 or max(hs)>=cal['bottom_m']:raise ValueError('invalid_known_levels')
    coords=[];usable=[];discarded=[]
    for i,obs in enumerate(observations):
        try:coords.append(optical_coordinate(obs,cal));usable.append(i)
        except ValueError as exc:discarded.append(dict(frame=i,reason=str(exc)))
    if len(coords)<3:raise ValueError('fewer_than_three_usable_levels')
    x=np.array([c['value'] for c in coords]);y=hs[usable]
    order=np.argsort(x);x=x[order];y=y[order];usable=np.asarray(usable)[order]
    if np.ptp(y)<max(.03,.15*max(y)) or np.ptp(x)<.01*cal['bottom_m']:raise ValueError('insufficient_level_span')
    if np.any(np.diff(y)<=0):raise ValueError('nonmonotonic_prior')
    coefficients=_line(x,y)
    if not .25<coefficients[0]<4:raise ValueError('invalid_optical_gain')
    loo=[]
    for i in range(len(x)):
        ids=np.arange(len(x))!=i
        loo.append(_map(x[i],x[ids],y[ids],mode)-y[i])
    # Calibration errors are output-domain quantities, not IOR parameter SD.
    normalized=np.abs(loo)/np.maximum(.005,.02*y)
    if mode=='affine' and np.max(normalized)>12:raise ValueError('inconsistent_prior_predictions')
    residual=y-np.polyval(coefficients,x)
    label_floor=float(np.median(.0005+.001*y))
    return dict(version=24,mode=mode,coefficients=coefficients.tolist(),x=x.tolist(),y=y.tolist(),
        used_frames=usable.tolist(),discarded=discarded,loo_errors_m=np.asarray(loo).tolist(),
        calibration_error_m=float(max(label_floor,np.sqrt(np.mean(np.square(loo))))),
        fit_error_m=float(np.sqrt(np.mean(residual**2))),span_m=float(np.ptp(x)),
        label_floor_m=label_floor,research_only=True,production_enabled=False)


def infer(obs,prior,cal):
    try:coordinate=optical_coordinate(obs,cal)
    except ValueError as exc:return dict(height_m=None,observable=False,uncertainty_m=None,reasons=[str(exc)])
    x=coordinate['value'];xp=np.asarray(prior['x']);yp=np.asarray(prior['y']);p=np.asarray(prior['coefficients'])
    h=_map(x,xp,yp,prior['mode'])
    extrap=max(min(xp)-x,x-max(xp),0)/prior['span_m']
    local_risk=(prior['calibration_error_m'] if prior['mode']=='affine' else
                max(prior['label_floor_m'],float(np.interp(x,xp,np.abs(prior['loo_errors_m'])))))
    dx=max(prior['span_m']*1e-5,1e-7)
    local_gain=(_map(x+dx,xp,yp,prior['mode'])-_map(x-dx,xp,yp,prior['mode']))/(2*dx)
    uncertainty=float(np.sqrt((local_gain*coordinate['noise'])**2+local_risk**2)*(1+3*extrap))
    reasons=[]
    if h<=0 or h>=.85*cal['bottom_m']:reasons.append('geometry_limit')
    if extrap>.25:reasons.append('outside_calibrated_span')
    return dict(height_m=h,observable=not reasons,uncertainty_m=uncertainty,reasons=reasons,
        optical_coordinate_m=x,spatial_noise_m=coordinate['noise'],extrapolation_fraction=extrap,
        research_only=True,hardware_qualified=False)


def accepted(prediction,multiplier):
    h=prediction.get('height_m');u=prediction.get('uncertainty_m')
    return bool(multiplier is not None and prediction.get('observable') and h is not None
        and u is not None and np.isfinite(u) and u<=multiplier*max(.005,.02*max(h,0)))


def route(primary,fallback,primary_multiplier,fallback_multiplier):
    """Preserve every accepted V23 output; qualify added outputs separately."""
    if accepted(primary,primary_multiplier):return dict(primary,accepted=True,source='v23_preserved')
    if accepted(fallback,fallback_multiplier):return dict(fallback,accepted=True,source='calibrated_fallback')
    return dict(fallback,accepted=False,source='rejected')
