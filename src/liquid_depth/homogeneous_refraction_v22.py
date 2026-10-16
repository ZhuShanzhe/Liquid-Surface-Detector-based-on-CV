"""Research-only homogeneous, horizontal free-surface inverse optics.

Only explicit nondispersive coaxial ToF or measured RGB-reference ray
correspondences are supported. NOT a correction for commodity stereo depth.
Indices at different wavelengths and group/phase indices are assumed equal
in the ToF proxy. A real sensor requires its own validated forward model.
Calibration sees known heights; inference has no current height/event labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.optimize import least_squares

MODELS = ('nondispersive_coaxial_tof_proxy', 'rgb_reference_snell')


def coefficient(index, q2, model):
    """value = height * coefficient + bias; q2 is squared air-ray slope.

    ToF value: measured *axial* optical depth minus empty bottom axial depth.
    RGB value: radial bottom correspondence contraction in metric units,
    dot(q, q*B - X_bottom)/dot(q,q), not an RGB-D depth-map value.
    """
    if model not in MODELS:
        raise ValueError('Unsupported sensor model; stereo depth is not ToF')
    q2 = np.asarray(q2, float)
    if not np.isfinite(q2).all() or (q2 < 0).any() or not 1.2 <= index <= 1.6:
        raise ValueError('Invalid refractive index or ray slope')
    a = np.sqrt(index**2 + (index**2-1)*q2)
    if model == 'rgb_reference_snell':
        return 1 - 1/a
    # n*c0/cos(theta_liquid), derived from Snell plus axial depth conversion.
    return index**2/a - 1


def samples(frame, sigma):
    """Non-oracle bottom-path candidate selection; no surface/event masks.

    One robust observation per independent spatial tile is expected. The
    raw-pixel adapter must establish correspondences/ROI and aggregate tiles.
    Positive optical excess selects bottom candidates, but not their identity.
    """
    value = np.asarray(frame['value_m'], float).ravel()
    q2 = np.asarray(frame['q2'], float).ravel()
    if value.shape != q2.shape:
        raise ValueError('Aligned ray observations required')
    good = np.isfinite(value) & np.isfinite(q2) & (q2 >= .01) & (q2 <= 1.)
    good &= value >= -3*sigma
    return value[good], q2[good]


@dataclass(frozen=True)
class MaterialPrior:
    sensor_model: str
    index: float
    bias_m: float
    bottom_distance_m: float
    observation_sigma_m: float
    index_jackknife_sd: float
    bias_jackknife_sd_m: float
    calibration_residual_mad_m: float
    calibrated_height_min_m: float
    calibrated_height_max_m: float
    calibration_frames: int
    validated_holdout_frames: int
    hardware_qualified: bool = False
    production_enabled: bool = False

    def to_dict(self):
        return asdict(self)


def fit_material_prior(frames, known_heights_m, *, sensor_model, bottom_distance_m,
                       observation_sigma_m=.002):
    """Fit n and residual offset from >=3 independently known liquid levels.

    The third level permits leave-one-frame-out instability diagnostics.
    Repeating one liquid level deliberately fails observability, even if many
    pixels produce a deceptively small optimizer covariance.
    """
    if sensor_model not in MODELS:
        raise ValueError('Explicit supported forward model required')
    heights = np.asarray(known_heights_m, float)
    if len(frames) != len(heights) or len(frames) < 3:
        raise ValueError('At least three labelled calibration frames required')
    if not np.isfinite(heights).all() or np.any(heights <= 0) or bottom_distance_m <= max(heights):
        raise ValueError('Invalid bottom reference or calibration levels')
    if not .00001 <= observation_sigma_m <= .1:
        raise ValueError('Invalid observation noise scale')
    if np.ptp(heights) < max(.03, .15*max(heights)):
        raise ValueError('Unobservable index/bias: insufficient known height span')
    observations = [samples(f, observation_sigma_m) for f in frames]
    if any(len(v) < 16 for v, q in observations):
        raise ValueError('Insufficient independent spatial tiles in calibration')

    def fit(selected):
        def residual(x):
            return np.concatenate([
                (v - (heights[i]*coefficient(x[0], q, sensor_model)+x[1]))
                / observation_sigma_m / np.sqrt(len(v))
                for i, (v, q) in enumerate(observations) if i in selected])
        # Equal total weight per frame, not per correlated pixel.
        scale = 1/np.sqrt(np.median([len(v) for v, q in observations]))
        r = least_squares(residual, [1.33, 0.], bounds=([1.2, -.1], [1.6, .1]),
                          loss='soft_l1', f_scale=scale, max_nfev=120)
        if not r.success:
            raise ValueError('Material calibration did not converge')
        return r.x

    n, bias = fit(set(range(len(frames))))
    estimates = np.array([fit(set(range(len(frames))) - {i}) for i in range(len(frames))])
    sd = np.sqrt((len(frames)-1)/len(frames)*np.sum((estimates-estimates.mean(0))**2, 0))
    errors = [v - (h*coefficient(n, q, sensor_model)+bias)
              for h, (v, q) in zip(heights, observations)]
    mad = float(np.median(np.abs(np.concatenate(errors))))
    if n < 1.205 or n > 1.595 or sd[0] > .03 or mad > 3*observation_sigma_m:
        raise ValueError('Unstable material prior or incompatible forward model')
    # Held-out frame checks are performed with priors fit WITHOUT that frame.
    holdout = []
    for i, (v, q) in enumerate(observations):
        ni, bi = estimates[i]
        hi = np.median((v-bi)/coefficient(ni, q, sensor_model))
        holdout.append(abs(hi-heights[i]) <= max(.005, .03*heights[i]))
    if not all(holdout):
        raise ValueError('Leave-one-prior-frame liquid-height validation failed')
    return MaterialPrior(sensor_model, float(n), float(bias), float(bottom_distance_m),
        float(observation_sigma_m), float(sd[0]), float(sd[1]), mad,
        float(min(heights)), float(max(heights)), len(frames), int(sum(holdout)))


def infer_height(frame, prior: MaterialPrior):
    """Frozen material parameters; output is bottom-referenced vertical height.

    A coherent wrong path can evade these gates. Diagnostics are model checks,
    NOT a guarantee of correctness or calibrated statistical coverage.
    """
    v, q = samples(frame, prior.observation_sigma_m)
    out = dict(accepted=False, height_m=None, reasons=[], independent_tiles=len(v),
        approximate_95_halfwidth_m=None, research_only=True, hardware_qualified=False)
    if len(v) < 16:
        out['reasons'] = ['insufficient_optical_bottom_candidates']
        return out
    c = coefficient(prior.index, q, prior.sensor_model)
    per_tile = (v-prior.bias_m)/c
    h = float(np.median(per_tile))
    residual = v-(prior.bias_m+h*c)
    inlier = np.abs(residual) <= 3*prior.observation_sigma_m
    out['model_inlier_fraction'] = float(inlier.mean())
    if inlier.sum() >= 16:
        h = float(np.median(per_tile[inlier]))
    out['height_m'] = h
    # Includes a conservative *frame-common* error term, not sigma/sqrt(pixels).
    epsilon = 1e-4
    dc = (coefficient(prior.index+epsilon, q, prior.sensor_model)
          - coefficient(prior.index-epsilon, q, prior.sensor_model))/(2*epsilon)
    sensitivity = float(np.median(np.abs(dc/c)))
    noise_height = prior.observation_sigma_m / float(np.median(c))
    param_height = abs(h)*sensitivity*prior.index_jackknife_sd
    offset_height = prior.bias_jackknife_sd_m / float(np.median(c))
    halfwidth = 1.96*np.sqrt(noise_height**2 + param_height**2 + offset_height**2)
    out['approximate_95_halfwidth_m'] = float(halfwidth)
    reasons = []
    if not .005 < h < .85*prior.bottom_distance_m:
        reasons.append('outside_supported_height_geometry')
    if inlier.mean() < .60:
        reasons.append('mixed_path_or_scattering_inconsistency')
    if h < .5*prior.calibrated_height_min_m or h > 1.5*prior.calibrated_height_max_m:
        reasons.append('outside_calibration_height_envelope')
    if halfwidth > max(.005, .02*max(h, 0)):
        reasons.append('optical_sensitivity_exceeds_height_tolerance')
    out['reasons'], out['accepted'] = reasons, not reasons
    return out
