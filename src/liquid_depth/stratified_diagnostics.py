"""Offline diagnostics for stratified transparent liquids; no sensor/inference model."""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment


def fresnel_unpolarized(n_from, n_to, angle_rad=0.0):
    """Lossless dielectric power reflectance at a single wavelength/interface."""
    if not all(math.isfinite(x) and x > 0 for x in (n_from, n_to)):
        raise ValueError("Refractive indices must be finite and positive")
    if not math.isfinite(angle_rad) or not 0 <= angle_rad < math.pi / 2:
        raise ValueError("Incidence angle must be in [0, pi/2)")
    if n_from == n_to:
        return 0.0
    sin_t = n_from / n_to * math.sin(angle_rad)
    if sin_t >= 1:
        return 1.0
    ci, ct = math.cos(angle_rad), math.sqrt(1 - sin_t * sin_t)
    rs = (n_from * ci - n_to * ct) / (n_from * ci + n_to * ct)
    rp = (n_to * ci - n_from * ct) / (n_to * ci + n_from * ct)
    return (rs * rs + rp * rp) / 2


def slab_geometry(thicknesses_m, refractive_indices, angle_rad=0.0, n_air=1.0):
    """Exact Snell lateral shift through planar layers, plus paraxial apparent depth.

    This is a controlled optical calculation, NOT a rendered RGB image or RGB-D
    camera output. It omits absorption, scattering, glass walls and sensor transfer.
    Thicknesses and indices are simulation parameters, never inference inputs.
    """
    t = np.asarray(thicknesses_m, dtype=float)
    n = np.asarray(refractive_indices, dtype=float)
    if t.ndim != 1 or not len(t) or n.shape != t.shape:
        raise ValueError("Expected nonempty matching one-dimensional arrays")
    if not np.isfinite(t).all() or (t <= 0).any() or not np.isfinite(n).all() or (n <= 0).any():
        raise ValueError("Thicknesses and indices must be finite and positive")
    if not math.isfinite(n_air) or n_air <= 0 or not math.isfinite(angle_rad) or not 0 <= angle_rad < math.pi / 2:
        raise ValueError("Invalid exterior medium or angle")
    sine = n_air * math.sin(angle_rad) / n
    if (sine >= 1).any():
        raise ValueError("No transmitted path for this planar stack")
    theta = np.arcsin(sine)
    return dict(
        total_thickness_m=float(t.sum()),
        lateral_displacement_m=float((t * np.tan(theta)).sum()),
        straight_air_displacement_m=float(t.sum() * math.tan(angle_rad)),
        paraxial_apparent_depth_m=float((t * n_air / n).sum()),
        geometric_interface_depths_m=np.cumsum(t)[:-1].tolist(),
        transmitted_angles_rad=theta.tolist(),
    )


def match_interfaces(predicted_m, truth_m, absolute_tolerance_m=.005, relative_tolerance=.02):
    """One-to-one matching of ACTIVE predictions on one ray (evaluation only).

    First maximize matches inside tolerance, then minimize normalized error.
    Depth matching alone does not certify interface semantics.
    """
    pred, truth = np.asarray(predicted_m, float), np.asarray(truth_m, float)
    if pred.ndim != 1 or truth.ndim != 1:
        raise ValueError("Depth sets must be one-dimensional")
    if any(not np.isfinite(x).all() or (x <= 0).any() for x in (pred, truth)):
        raise ValueError("Depth sets must contain finite positive values")
    if not math.isfinite(absolute_tolerance_m) or absolute_tolerance_m <= 0 or not math.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("Invalid tolerance")
    n, m = len(truth), len(pred)
    pairs = []
    if n:
        tolerance = np.maximum(absolute_tolerance_m, relative_tolerance * truth)
        error = np.abs(truth[:, None] - pred[None, :])
        allowed = error <= tolerance[:, None]
        # n independent dummy slots permit abstention. Any extra match dominates
        # the sum of all possible feasible normalized-error tie-break penalties.
        cost = np.ones((n, m + n), dtype=float)
        cost[:, :m] = np.where(allowed, error / tolerance[:, None] / (n + 1), n + 1)
        rows, cols = linear_sum_assignment(cost)
        pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if j < m and allowed[i, j]]
    matched = len(pairs)
    legacy_hits = int((np.abs(truth[:, None] - pred[None, :]).min(1) <= np.maximum(absolute_tolerance_m, relative_tolerance * truth)).sum()) if n and m else 0
    return dict(
        truth_count=n, predicted_count=m, matched_count=matched,
        recall=matched / n if n else None,
        precision=matched / m if m else None,
        all_truth_matched=matched == n,
        complete_set_correct=matched == n == m,
        legacy_recall=legacy_hits / n if n else None,
        pairs=pairs,
    )
