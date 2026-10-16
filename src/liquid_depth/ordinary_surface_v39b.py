"""Observable-return mixture: prevent a censored-away surface degeneracy.

Mixture fraction describes VALID returns, not latent pre-range-cutoff events.
An invisible surface must not turn all visible samples into bottom echoes.
No liquid-height bound, capacity clipping, forced zero or confidence rejection.
"""
import numpy as np
from scipy.optimize import minimize
from .ordinary_surface_v39 import _observations, _cells, _prob, _installation


def estimate_v39(raw, cal, rim_z_m, model, reference=None, *, noisy_echo=True):
    out=dict(statistics=None,qualified=False,reasons=[],route='ordinary_observed_return_mixture_v39b')
    d,reason=_observations(raw,cal,rim_z_m)
    if d is None:
        out['reasons']=[reason];return out
    z,c=d;fb=float(cal.intrinsics[0,0]*model.baseline_m)
    bottom=float(cal.camera_to_world_cv[2,3]-cal.bottom_z_m)
    if reference is not None and (reference.shape!=tuple(raw.shape) or not np.isclose(reference.fb,fb) or reference.installation_digest!=_installation(cal,rim_z_m)):
        raise ValueError('Dry reference belongs to a different installation')
    bias=reference.disparity_bias_px if reference else 0.
    sigma=reference.disparity_sigma_px if reference else model.disparity_sigma_px
    cells=_cells(z,fb,model)
    if np.count_nonzero(cells[2])<24:
        out['reasons']=['unsupported_depth_quantization'];return out
    floor,fnorm=_prob(cells,fb*c/bottom+bias,sigma)
    floor=floor/np.maximum(fnorm,1e-300)
    if not noisy_echo:floor=(abs(z-np.round(bottom/c/.001)*.001)<.000501).astype(float)
    start=float(np.median((fb/z-bias)/c))
    def objective(p):
        mass,norm=_prob(cells,p[0]*c+bias,sigma)
        surface=mass/np.maximum(norm,1e-300)
        prob=(1-p[1])*surface+p[1]*floor
        return float(-np.mean(np.log(np.maximum(.995*prob+.005/4096,1e-300))))
    bounds=[(max(.01,start-max(3*sigma,start*.15)),start+max(3*sigma,start*.15)),(0.,.35)]
    fits=[minimize(objective,[start,a],method='L-BFGS-B',bounds=bounds,options={'ftol':1e-12,'maxiter':100}) for a in (.04,.15)]
    fits=[f for f in fits if f.success and np.isfinite(f.fun)]
    if not fits:
        out['reasons']=['mixture_optimizer_failed'];return out
    fit=min(fits,key=lambda f:f.fun);h=float(bottom-fb/fit.x[0])
    out.update(statistics={k:h for k in ('mean_m','min_m','max_m','p05_m','p95_m')},selected_points=len(z),
        fitted_echo_fraction=float(fit.x[1]),calibrated_bias_px=bias,calibrated_sigma_px=sigma)
    out['reasons']=['valid_return_mixture_not_latent_echo_rate','relaxed_gate_signed_height',
        'fixed_pose_flat_ordinary_only','research_not_hardware_qualified']
    if reference is None:out['reasons'].append('no_independent_bias_reference')
    if fit.x[1]>.34:out['reasons'].append('echo_fraction_weakly_identified_or_at_model_boundary')
    return out
