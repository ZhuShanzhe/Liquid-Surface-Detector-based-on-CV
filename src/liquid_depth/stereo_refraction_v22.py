"""Finite-baseline Snell inverse for a homogeneous flat liquid, research only.

Rectified horizontal epipolar-plane rays; phase refractive index. This is not
the ToF optical-length equation. Real 2-D correspondences may acquire vertical
disparity through refraction and need explicit ray tracing and match checks.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

FAMILIES = ('active_stereo', 'structured_light')


def apparent_contraction(height, index, qx, bottom, baseline):
    """B - Z_triangulated, for bottom-return correspondences in the central row.

    Air -> homogeneous liquid -> Lambertian bottom, with reciprocal paths.
    For structured light the second centre is the calibrated projector.
    No assumption that the measured points are on the free surface.
    """
    q = np.asarray(qx, float)
    if not 0 <= height < bottom or not 1.2 <= index <= 1.6 or baseline <= 0:
        raise ValueError('Invalid liquid/baseline geometry')
    if not np.isfinite(q).all() or (np.abs(q) > 1).any():
        raise ValueError('Unsupported ray slopes')
    d = bottom-height
    a = np.sqrt(index**2 + (index**2-1)*q*q)
    x_bottom = q*(d+height/a)
    right = (x_bottom-baseline)/max(d+height/index, 1e-6)
    for _ in range(12):
        ar = np.sqrt(index**2+(index**2-1)*right*right)
        length = d+height/ar
        residual = right*length-(x_bottom-baseline)
        derivative = length-height*(index**2-1)*right*right/ar**3
        right -= residual/derivative
    disparity = q-right
    if (disparity <= 0).any():
        raise ValueError('Invalid stereo correspondence')
    return bottom-baseline/disparity


def observations(frame):
    q = np.asarray(frame['qx'], float).ravel()
    v = np.asarray(frame['value_m'], float).ravel()
    if q.shape != v.shape:
        raise ValueError('Aligned measured correspondence samples required')
    good = np.isfinite(q) & np.isfinite(v) & (np.abs(q) <= .8)
    return v[good], q[good]


@dataclass(frozen=True)
class StereoMaterialPrior:
    sensor_family: str
    index: float
    bias_m: float
    bottom_m: float
    baseline_m: float
    sigma_m: float
    index_jackknife_sd: float
    bias_jackknife_sd_m: float
    height_min_m: float
    height_max_m: float
    calibration_frames: int
    production_enabled: bool = False
    hardware_qualified: bool = False

    def to_dict(self):
        return asdict(self)


def fit_stereo_prior(frames, known_heights_m, *, sensor_family, bottom_m, baseline_m, sigma_m):
    if sensor_family not in FAMILIES:
        raise ValueError('Stereo/projector triangulation only; ToF is different')
    hs = np.asarray(known_heights_m, float)
    if len(frames) != len(hs) or len(hs) < 3 or not np.isfinite(hs).all():
        raise ValueError('Three independently labelled prior frames required')
    if min(hs) <= 0 or max(hs) >= bottom_m or baseline_m <= 0 or sigma_m <= 0:
        raise ValueError('Invalid calibration geometry/noise')
    if np.ptp(hs) < max(.03, .15*max(hs)):
        raise ValueError('Insufficient known liquid-level diversity')
    obs = [observations(f) for f in frames]
    if any(len(v) < 16 for v, q in obs):
        raise ValueError('Insufficient independent bottom-match regions')
    def fit(ids):
        def residual(p):
            return np.concatenate([(v-apparent_contraction(hs[i], p[0], q, bottom_m, baseline_m)-p[1])
                /sigma_m/np.sqrt(len(v)) for i, (v, q) in enumerate(obs) if i in ids])
        result = least_squares(residual, [1.333, 0], bounds=([1.2, -.1], [1.6, .1]),
            loss='soft_l1', f_scale=1/8, max_nfev=100)
        if not result.success:
            raise ValueError('Stereo calibration failed to converge')
        return result.x
    n, bias = fit(set(range(len(hs))))
    leaveout = np.array([fit(set(range(len(hs)))-{i}) for i in range(len(hs))])
    sd = np.sqrt((len(hs)-1)/len(hs)*np.sum((leaveout-leaveout.mean(0))**2, 0))
    if n <= 1.205 or n >= 1.595 or sd[0] > .03:
        raise ValueError('Unstable material index or wrong optical model')
    for i, (v, q) in enumerate(obs):
        ni, bi = leaveout[i]
        result = minimize_scalar(lambda h: np.median(np.abs(v-apparent_contraction(h, ni, q, bottom_m, baseline_m)-bi)),
                                 bounds=(.001, .85*bottom_m), method='bounded')
        if not result.success or abs(result.x-hs[i]) > max(.005, .03*hs[i]):
            raise ValueError('Held-out prior liquid-level validation failed')
    return StereoMaterialPrior(sensor_family, float(n), float(bias), float(bottom_m),
        float(baseline_m), float(sigma_m), float(sd[0]), float(sd[1]),
        float(min(hs)), float(max(hs)), len(hs))


def infer_stereo_height(frame, prior):
    v, q = observations(frame)
    out = dict(accepted=False, height_m=None, reasons=[], independent_regions=len(v),
               research_only=True, hardware_qualified=False)
    if len(v) < 16:
        out['reasons'] = ['insufficient_independent_bottom_matches']
        return out
    def objective(h):
        r = (v-apparent_contraction(h, prior.index, q, prior.bottom_m, prior.baseline_m)-prior.bias_m)/prior.sigma_m
        return float(np.mean(np.sqrt(1+r*r)-1))
    result = minimize_scalar(objective, bounds=(.001, .85*prior.bottom_m), method='bounded',
                             options={'xatol': 1e-7})
    h = float(result.x)
    out['height_m'] = h
    pred = apparent_contraction(h, prior.index, q, prior.bottom_m, prior.baseline_m)+prior.bias_m
    inliers = np.abs(v-pred) <= 3*prior.sigma_m
    dh = max(1e-6, h*1e-4)
    dn = 1e-4
    sensitivity_h = np.median((apparent_contraction(h+dh, prior.index, q, prior.bottom_m, prior.baseline_m)
        -apparent_contraction(h-dh, prior.index, q, prior.bottom_m, prior.baseline_m))/(2*dh))
    sensitivity_n = np.median((apparent_contraction(h, prior.index+dn, q, prior.bottom_m, prior.baseline_m)
        -apparent_contraction(h, prior.index-dn, q, prior.bottom_m, prior.baseline_m))/(2*dn))
    # A conservative common-error floor; not sigma/sqrt(number of pixels).
    width = 1.96*np.sqrt(prior.sigma_m**2+prior.bias_jackknife_sd_m**2
                         +(sensitivity_n*prior.index_jackknife_sd)**2)/abs(sensitivity_h)
    reasons = []
    if not result.success:
        reasons.append('height_optimizer_failed')
    if inliers.mean() < .60:
        reasons.append('incompatible_matches_or_mixed_paths')
    if width > max(.005, .02*h):
        reasons.append('disparity_noise_exceeds_height_tolerance')
    if h < .5*prior.height_min_m or h > 1.5*prior.height_max_m:
        reasons.append('outside_prior_height_envelope')
    if h >= .849*prior.bottom_m:
        reasons.append('height_at_geometry_limit')
    out.update(accepted=not reasons, reasons=reasons, model_inlier_fraction=float(inliers.mean()),
               approximate_95_halfwidth_m=float(width))
    return out
