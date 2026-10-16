"""Optional plane-height fit for disparity returns censored by camera range.

The caller must provide the camera's observed maximum axial depth after all
SDK post-processing. This is a measurement model, not a liquid mask.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr


def fit_censored_disparity(depth_m, roi, calibration, baseline_m,
                           max_return_depth_m):
    """Fit height and disparity spread; return None if evidence is unusable."""
    raw = np.asarray(depth_m)
    spatial_roi = np.asarray(roi, dtype=bool)
    if raw.ndim == 2 and raw.shape == spatial_roi.shape:
        support = spatial_roi & np.isfinite(raw) & (raw > 0)
        yy, xx = np.nonzero(support)
        frames_used = 1
    elif raw.ndim == 3 and raw.shape[1:] == spatial_roi.shape:
        support = spatial_roi[None, ...] & np.isfinite(raw) & (raw > 0)
        _, yy, xx = np.nonzero(support)
        frames_used = len(raw)
    else:
        raise ValueError('aligned_depth_frames_and_container_roi_required')
    if len(xx) < 24:
        return None
    # A mismatched SDK range invalidates the truncation likelihood. The
    # caller can still report its ungated container-return fallback.
    if np.max(raw[support]) > float(max_return_depth_m) + .002:
        return None
    k = np.asarray(calibration.intrinsics)
    pose = np.asarray(calibration.camera_to_world_cv)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    fb = fx * float(baseline_m)
    if fb <= 0 or not np.isfinite(fb) or max_return_depth_m <= 0:
        raise ValueError('positive_baseline_and_sensor_max_depth_required')
    incidence = -(pose[2, 0] * (xx - k[0, 2]) / fx +
                  pose[2, 1] * (yy - k[1, 2]) / fy + pose[2, 2])
    good = incidence > .05
    if np.count_nonzero(good) < 24:
        return None
    observed = fb / raw[support][good]
    incidence = incidence[good]
    lower_disparity = fb / float(max_return_depth_m)
    camera_to_bottom = float(pose[2, 3] - calibration.bottom_z_m)
    if camera_to_bottom <= .02:
        return None

    # Range truncation is on observed depth. A broad 5% outlier component
    # limits isolated erroneous returns without rejecting the whole frame.
    def objective(parameters):
        height_m, sigma_px = parameters
        predicted = fb * incidence / (camera_to_bottom - height_m)
        normal_logpdf = (-.5 * ((observed - predicted) / sigma_px) ** 2
                         - np.log(sigma_px) - .9189385332046727
                         - log_ndtr((predicted - lower_disparity) / sigma_px))
        return -float(np.logaddexp(np.log(.95) + normal_logpdf,
                                   np.log(.05 / 5.)).sum())

    initial_height = camera_to_bottom - fb / float(np.median(observed))
    high = camera_to_bottom - .02
    low = -float(max_return_depth_m)
    initial_height = float(np.clip(initial_height, low + .001, high - .001))
    result = minimize(objective, (initial_height, .25), method='L-BFGS-B',
                      bounds=((low, high), (.03, 1.5)),
                      options={'maxiter': 80, 'ftol': 1e-7})
    if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
        return None
    return {'height_m': float(result.x[0]),
            'disparity_sigma_px': float(result.x[1]),
            'support_pixels': len(observed),
            'frames_used': frames_used,
            'optimizer_converged': bool(result.success),
            'max_return_depth_m': float(max_return_depth_m)}
