"""Gravity-constrained surface evidence from calibrated current returns."""
from dataclasses import replace

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .empty_geometry_v41 import height_calibration
from .liquid_height_metrics import world_points
from .ordinary_surface_v35 import interior_ray_roi
from .surface_refinement import StereoNoiseModel, censored_clearance, weighted_median


def current_support(bundle, depth, fraction=.85):
    cal = height_calibration(bundle)
    raw = np.asarray(depth)
    rim = float(bundle['rim_z_m'])
    opening = interior_ray_roi(replace(cal, bottom_z_m=rim-1e-5), rim,
                               raw.shape, fraction)
    xyz = world_points(raw, cal)
    xy = xyz[..., :2] / np.asarray(cal.radii_m)
    inside = (abs(xy)**cal.footprint_exponent).sum(-1) <= 1.02**cal.footprint_exponent
    margin = max(.03, .02*rim)
    selected = (opening & np.isfinite(raw) & (raw > 0) & inside &
                (xyz[..., 2] >= cal.bottom_z_m-margin) &
                (xyz[..., 2] <= rim+margin))
    return cal, xyz, selected


def estimate_surface_evidence(bundle, depth, *, fraction=.85, layer_mode=False,
                              selection_mask=None, max_return_depth_m=10., min_points=24,
                              min_tiles=4):
    if selection_mask is None:
        cal, xyz, selected = current_support(bundle, depth, fraction)
    else:
        cal = height_calibration(bundle)
        xyz = world_points(np.asarray(depth), cal)
        # A sensor-domain likelihood must see the actual noisy return tails.
        # Clipping samples at the vessel bottom creates an unmodelled censor.
        raw = np.asarray(depth)
        selected = np.asarray(selection_mask, bool) & np.isfinite(raw) & (raw > 0)
    y, x = np.nonzero(selected)
    out = {'height_m': None, 'selected_pixels': len(x), 'layer_pixels': 0,
           'selected_mask': selected, 'layer_mask': np.zeros_like(selected)}
    if len(x) < min_points:
        return out
    if len(x) > 6000:
        ids = np.linspace(0, len(x)-1, 6000).astype(int)
        x, y = x[ids], y[ids]
    raw = np.asarray(depth)
    z = raw[y, x]
    heights = xyz[y, x, 2]-cal.bottom_z_m
    k, pose = cal.intrinsics, cal.camera_to_world_cv
    rays = np.column_stack(((x-k[0, 2])/k[0, 0], (y-k[1, 2])/k[1, 1], np.ones(len(x))))
    cosine = -(rays @ pose[:3, :3].T)[:, 2]
    tile = x*8//raw.shape[1] + 8*(y*8//raw.shape[0])
    _, inverse, counts = np.unique(tile, return_inverse=True, return_counts=True)
    weights = 1/np.sqrt(counts[inverse])
    sensor = bundle['sensor_family']
    model = None if sensor == 'tof' else StereoNoiseModel.simulation_proxy(sensor)
    if model is not None:
        model = replace(model, max_depth_m=max_return_depth_m)
    if model:
        sigma = np.sqrt((model.depth_noise_floor_m + model.depth_noise_quadratic*z*z)**2 +
                        (model.disparity_sigma_px*z*z/(k[0, 0]*model.baseline_m))**2)
    else:
        sigma = .0015 + .0007*z
    keep = np.ones(len(z), bool)
    if layer_mode:
        step = max(.003, min(.05, float(np.median(sigma))*.5))
        lo, hi = float(heights.min()), float(heights.max())
        edges = np.linspace(lo-step, hi+step, min(3000, max(4, int((hi-lo)/step)+3)))
        mass = np.histogram(heights, edges, weights=weights/np.maximum(sigma, .003))[0]
        density = gaussian_filter1d(mass, 1.0)
        best = None
        for index in np.argsort(density)[-12:][::-1]:
            center = .5*(edges[index]+edges[index+1])
            inlier = abs(heights-center) <= np.maximum(.006, 2.5*sigma)
            tiles = len(np.unique(tile[inlier]))
            if inlier.sum() < min_points or tiles < min_tiles:
                continue
            score = float(np.sum(weights[inlier]/np.maximum(sigma[inlier], .003)))
            if best is None or score > best[0]:
                best = (score, inlier)
        if best is None:
            out['reason'] = 'no_spatially_supported_layer'
            return out
        keep = best[1]
    weights = weights[keep]
    weights /= weights.sum()
    value = weighted_median(heights[keep], weights)
    if model:
        clearance = censored_clearance(z[keep], cosine[keep], weights, k[0, 0], model)
        if clearance is None:
            out['reason'] = 'sensor_fit_unobservable'
            return out
        value = float(pose[2, 3]-cal.bottom_z_m-clearance)
    out['layer_mask'][y[keep], x[keep]] = True
    out.update(height_m=float(value), layer_pixels=int(keep.sum()),
               layer_tiles=len(np.unique(tile[keep])),
               median_noise_m=float(np.median(sigma[keep])))
    return out


def dry_changed_evidence(bundle, depth, dry_depth, fixed_roi):
    """Use independently observed dry bottom to veto unchanged bottom echoes.

    A changed point needs a positive displacement larger than four estimated
    noise standard deviations. A known-empty frame supplies a reference, not
    a known current liquid height. No holes are filled and no height is clipped.
    """
    cal = height_calibration(bundle)
    dry = np.asarray(dry_depth)
    raw = np.asarray(depth)
    dry_height = world_points(dry, cal)[..., 2]-cal.bottom_z_m
    wet_height = world_points(raw, cal)[..., 2]-cal.bottom_z_m
    valid_dry = np.asarray(fixed_roi, bool) & np.isfinite(dry) & (dry > 0)
    adjacent = valid_dry[:, 1:] & valid_dry[:, :-1]
    differences = np.diff(dry_height, axis=1)[adjacent]
    if len(differences) < 24:
        sigma_dry = float('inf')
    else:
        sigma_dry = max(.001, float(np.median(abs(
            differences-np.median(differences)))/(.67448975*np.sqrt(2))))
    sensor = bundle['sensor_family']
    z = np.where(np.isfinite(raw) & (raw > 0), raw, 0.)
    if sensor == 'tof':
        sigma_wet = .0015+.0007*z
    else:
        model = StereoNoiseModel.simulation_proxy(sensor)
        fb = float(cal.intrinsics[0, 0])*model.baseline_m
        sigma_wet = np.sqrt((model.depth_noise_floor_m+model.depth_noise_quadratic*z*z)**2 +
                            (model.disparity_sigma_px*z*z/fb)**2)
    bottom_observed = valid_dry & (abs(dry_height) <= max(.03, 4*sigma_dry))
    changed = (bottom_observed & np.isfinite(raw) & (raw > 0) &
               ((wet_height-dry_height) > np.maximum(.03, 4*np.hypot(sigma_wet, sigma_dry))))
    result = estimate_surface_evidence(
        bundle, raw, selection_mask=changed, layer_mode=True,
        min_points=12 if sensor == 'tof' else 24, min_tiles=1)
    result.update(dry_reference_pixels=int(bottom_observed.sum()),
                  dry_noise_sigma_m=sigma_dry if np.isfinite(sigma_dry) else None,
                  changed_pixels=int(changed.sum()))
    return result
