"""Target-only diagnostic inverse. Finite estimates are NOT qualified outputs.

Fits known-level priors in observed pixel space, avoiding inversion of each noisy
ray pair into a height before aggregation. No test truth, coverage target, or
confidence rejection participates in fitting or prediction.
"""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from .refraction_images_v23 import bottom_map, inverse_bottom
from .prior_reliability_v24 import _map

TARGET_CONDITIONS = frozenset(('transparent', 'translucent_mild'))


def require_target(condition):
    if condition not in TARGET_CONDITIONS:
        raise ValueError('outside_transparent_diagnostic_scope')


def usable(obs):
    return (obs['regions'] >= 20 and
            np.isfinite(obs['left_q']).all() and np.isfinite(obs['right_q']).all())


def pixel_residual(h, index, bias_px, obs, cal):
    world = bottom_map(obs['left_q'], h, index, cal['bottom_m'])
    predicted = inverse_bottom(world, h, index, cal['bottom_m'], (cal['baseline_m'], 0))
    return (predicted - obs['right_q']) * cal['fx'] - np.array([bias_px, 0.])


def solve_height(obs, prior, cal):
    if not usable(obs):
        return dict(height_m=None, reasons=['insufficient_or_invalid_matches'], finite=False)
    weights = np.maximum(np.asarray(obs['weights']), .02)
    def objective(h):
        r = pixel_residual(h, prior['index'], prior['bias_px'], obs, cal)
        return float(np.average((np.sqrt(1+(r/.03)**2)-1).sum(1), weights=weights))
    upper = cal['bottom_m'] * .98
    fit = minimize_scalar(objective, bounds=(0., upper), method='bounded',
                          options={'xatol': max(1e-9, cal['bottom_m']*1e-8)})
    h = float(fit.x)
    reasons = []
    if not fit.success: reasons.append('optimizer_failed')
    if h < cal['bottom_m']*1e-5 or h > upper-cal['bottom_m']*1e-5:
        reasons.append('geometry_boundary')
    residual = pixel_residual(h, prior['index'], prior['bias_px'], obs, cal)
    return dict(height_m=h, finite=bool(np.isfinite(h)), reasons=reasons,
                residual_px=float(np.median(np.linalg.norm(residual, axis=1))),
                regions=obs['regions'])


def fit_prior(observations, heights, cal, condition, correction='none'):
    require_target(condition)
    if correction not in ('none', 'affine', 'pchip'):
        raise ValueError('unsupported_correction')
    heights = np.asarray(heights, float)
    if len(heights) != len(observations) or len(heights) < 3:
        raise ValueError('at_least_three_known_levels')
    if not np.isfinite(heights).all() or min(heights)<=0 or max(heights)>=cal['bottom_m']:
        raise ValueError('invalid_known_levels')
    ids = [i for i,o in enumerate(observations) if usable(o)]
    if len(ids)<3: raise ValueError('fewer_than_three_usable_levels')
    obs = [observations[i] for i in ids]; hs = heights[ids]
    if np.ptp(hs)<max(.03,.15*max(hs)): raise ValueError('insufficient_level_span')
    def objective(p):
        return np.concatenate([(pixel_residual(h,p[0],p[1],o,cal)*
            np.sqrt(np.maximum(o['weights'],.02))[:,None]/np.sqrt(len(o['weights']))).ravel()
            for o,h in zip(obs,hs)])
    fit=least_squares(objective,[1.37,0.],bounds=([1.2,-2.],[1.6,2.]),
                      loss='soft_l1',f_scale=.005,max_nfev=150)
    if not fit.success: raise ValueError('prior_optimizer_failed')
    singular=np.linalg.svd(fit.jac,compute_uv=False)
    prior=dict(index=float(fit.x[0]),bias_px=float(fit.x[1]),condition=condition,
               correction=correction,used_frames=ids,research_only=True,
               production_enabled=False,parameter_condition_number=float(singular[0]/max(singular[-1],1e-12)))
    xp=np.array([solve_height(o,prior,cal)['height_m'] for o in obs]);order=np.argsort(xp)
    if np.any(np.diff(hs[order])<=0) or np.min(np.diff(xp[order]))<=1e-9:
        raise ValueError('nonmonotonic_calibration')
    prior.update(x=xp[order].tolist(),y=hs[order].tolist(),
        prior_residual_m=(xp-hs).tolist(),calibration_floor_m=float(np.median(.0005+.001*hs)))
    return prior


def infer(obs,prior,cal):
    require_target(prior['condition'])
    result=solve_height(obs,prior,cal)
    if not result['finite']: return dict(result,diagnostic_only=True,qualified=False)
    raw=result['height_m'];xp=np.asarray(prior['x']);yp=np.asarray(prior['y'])
    h=_map(raw,xp,yp,prior['correction']) if prior['correction']!='none' else raw
    warnings=list(result['reasons'])
    if raw<xp.min() or raw>xp.max(): warnings.append('outside_prior_span')
    if h<=0 or h>=cal['bottom_m']: warnings.append('outside_physical_height')
    if prior['index']<=1.205 or prior['index']>=1.595: warnings.append('effective_index_at_bound')
    return dict(result,height_m=float(h),uncorrected_height_m=raw,reasons=warnings,
                diagnostic_only=True,qualified=False,uncertainty_calibrated=False)
