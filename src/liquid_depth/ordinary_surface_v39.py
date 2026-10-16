"""Opt-in ordinary flat-liquid metrology with an independently observed dry reference.

Signed height is never zeroed, capacity-clipped, or rejected by confidence. This
does not support transparent refraction, moving cameras, or an uncalibrated bottom.
"""
from dataclasses import dataclass, replace
import hashlib
import numpy as np
from scipy.optimize import minimize
from .ordinary_surface_v35 import interior_ray_roi
from .ordinary_surface_v36 import normal_mass, estimate_v36
from .surface_refinement import support


@dataclass(frozen=True)
class DryReference:
    disparity_bias_px: float
    disparity_sigma_px: float
    samples: int
    bottom_clearance_m: float
    fb: float
    shape: tuple
    installation_digest: str


def _installation(cal, rim):
    values = np.r_[cal.intrinsics.ravel(), cal.camera_to_world_cv.ravel(), cal.bottom_z_m,
                   cal.radii_m, cal.footprint_exponent, rim]
    return hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()


def _observations(raw, cal, rim, *, max_points=8192, wide=True):
    roi = interior_ray_roi(cal, rim, raw.shape, .65)
    if roi.sum() < 256:
        roi = interior_ray_roi(replace(cal, bottom_z_m=rim-1e-5), rim, raw.shape, .65)
    d, reason = support(raw, roi, cal.intrinsics, cal.camera_to_world_cv, cal.radii_m, max_points=max_points)
    if d is None:
        return None, reason
    if wide and np.median(d[2]) < 3:
        candidate = interior_ray_roi(cal, rim, raw.shape, .9)
        more, _ = support(raw, candidate, cal.intrinsics, cal.camera_to_world_cv, cal.radii_m, max_points=max_points)
        if more is not None:
            d = more
    _, pixels, z, _, _ = d
    k, t = cal.intrinsics, cal.camera_to_world_cv
    rays = np.column_stack(((pixels[:, 0]-k[0, 2])/k[0, 0], (pixels[:, 1]-k[1, 2])/k[1, 1], np.ones(len(z))))
    return (z.astype(float), -(rays @ t[:3, :3].T)[:, 2]), None


def _cells(depth, fb, model, q=1/16, dz=.001):
    imin = int(np.ceil(fb/model.max_depth_m/q-1e-8))
    imax = int(np.floor(fb/model.min_depth_m/q+1e-8))
    ilo = np.maximum(np.ceil(fb/(depth+dz*.501)/q-1e-7), imin)
    ihi = np.minimum(np.floor(fb/np.maximum(depth-dz*.501, 1e-6)/q+1e-7), imax)
    return (ilo-.5)*q, (ihi+.5)*q, ihi >= ilo, (imin-.5)*q, (imax+.5)*q


def _prob(cells, mu, sigma):
    left, right, possible, low, high = cells
    mass = np.where(possible, normal_mass((left-mu)/sigma, (right-mu)/sigma), 0.)
    norm = normal_mass((low-mu)/sigma, (high-mu)/sigma)
    return mass, norm


def calibrate_dry_reference(raw, cal, rim_z_m, model):
    """One real empty-container frame + independently calibrated bottom geometry.

    The empty designation is operator input, never inferred from test truth.
    Fit in quantized disparity, conditional on sensor valid range.
    """
    d, reason = _observations(raw, cal, rim_z_m, max_points=16384, wide=False)
    if d is None:
        raise ValueError('Unobservable dry reference: '+str(reason))
    z, c = d
    fb = float(cal.intrinsics[0, 0]*model.baseline_m)
    bottom = float(cal.camera_to_world_cv[2, 3]-cal.bottom_z_m)
    if bottom <= 0:
        raise ValueError('Camera must be above the bottom')
    cells = _cells(z, fb, model)
    expected = fb*c/bottom
    residual = fb/z-expected
    b0 = float(np.median(residual))
    s0 = max(.04, float(1.4826*np.median(abs(residual-b0))))
    def objective(p):
        mass, norm = _prob(cells, expected+p[0], np.exp(p[1]))
        return float(-np.mean(np.log(np.maximum(.998*mass/np.maximum(norm, 1e-300)+.002/4096, 1e-300))))
    fit = minimize(objective, [b0, np.log(s0)], method='L-BFGS-B', bounds=[(b0-2, b0+2), (np.log(.015), np.log(2.))], options={'ftol':1e-12})
    if not fit.success or not np.isfinite(fit.fun):
        raise ValueError('Dry reference fit failed: '+fit.message)
    return DryReference(float(fit.x[0]), float(np.exp(fit.x[1])), len(z), bottom, fb, tuple(raw.shape), _installation(cal, rim_z_m))


def estimate_v39(raw, cal, rim_z_m, model, reference=None, *, noisy_echo=True):
    """Use observed dry bias/noise; fit noisy bottom+surface quantized mixture.

    Without a dry reference, bias stays zero (explicit ablation). Noise and bias
    cannot be independently identified from a single unknown flat liquid level.
    """
    out = dict(statistics=None, qualified=False, reasons=[], route='ordinary_dry_bias_noisy_echo_v39')
    d, reason = _observations(raw, cal, rim_z_m)
    if d is None:
        out['reasons'] = [reason]
        return out
    z, c = d
    fb = float(cal.intrinsics[0, 0]*model.baseline_m)
    bottom = float(cal.camera_to_world_cv[2, 3]-cal.bottom_z_m)
    if reference is not None and (reference.shape != tuple(raw.shape) or not np.isclose(reference.fb, fb) or reference.installation_digest != _installation(cal, rim_z_m)):
        raise ValueError('Dry reference belongs to a different installation')
    bias = reference.disparity_bias_px if reference else 0.
    sigma = reference.disparity_sigma_px if reference else model.disparity_sigma_px
    cells = _cells(z, fb, model)
    if np.count_nonzero(cells[2]) < 24:
        out['reasons'] = ['unsupported_depth_quantization']
        return out
    floor, fnorm = _prob(cells, fb*c/bottom+bias, sigma)
    if not noisy_echo:
        floor = (abs(z-np.round(bottom/c/.001)*.001)<.000501).astype(float)
        fnorm = np.ones(len(z))
    start = float(np.median((fb/z-bias)/c))
    def objective(p):
        mass, norm = _prob(cells, p[0]*c+bias, sigma)
        alpha = p[1]
        prob = ((1-alpha)*mass+alpha*floor)/np.maximum((1-alpha)*norm+alpha*fnorm, 1e-300)
        return float(-np.mean(np.log(np.maximum(.995*prob+.005/4096, 1e-300))))
    bounds = [(max(.01, start-max(3*sigma, start*.15)), start+max(3*sigma, start*.15)), (0., .35)]
    fits = [minimize(objective, [start, a], method='L-BFGS-B', bounds=bounds, options={'ftol':1e-12,'maxiter':100}) for a in (.04,.15)]
    fits = [f for f in fits if f.success and np.isfinite(f.fun)]
    if not fits:
        out['reasons'] = ['mixture_optimizer_failed']
        return out
    fit = min(fits, key=lambda f:f.fun)
    h = float(bottom-fb/fit.x[0])
    out.update(statistics={k:h for k in ('mean_m','min_m','max_m','p05_m','p95_m')},
               selected_points=len(z), fitted_echo_fraction=float(fit.x[1]),
               calibrated_bias_px=bias, calibrated_sigma_px=sigma)
    out['reasons'] = ['relaxed_gate_signed_height','fixed_pose_flat_ordinary_only','research_not_hardware_qualified']
    if reference is None: out['reasons'].append('no_independent_bias_reference')
    if fit.x[1] > .34: out['reasons'].append('echo_fraction_weakly_identified_or_at_model_boundary')
    return out
