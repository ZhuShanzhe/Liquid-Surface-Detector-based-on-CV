"""Dry-anchored, multi-candidate return-layer vote for partial depth loss.

The empty-container geometry defines the zero-height layer. A second peak
inside the physical vessel can represent wet-surface returns. This is an
evidence-based candidate selector; it does not snap or clip a measurement.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def vote_dry_anchored_layer(
    height_returns_m: np.ndarray,
    *,
    rim_height_m: float,
    primary_height_m: float,
    container_quantile_height_m: float,
) -> dict:
    values = np.asarray(height_returns_m, dtype=float)
    values = values[np.isfinite(values)]
    result = {
        "height_m": float(primary_height_m),
        "selected": False,
        "candidate": "primary",
        "bottom_peak_m": None,
        "wet_peak_m": None,
        "wet_to_bottom_vote_ratio": None,
        "support_returns": len(values),
    }
    if len(values) < 24 or not np.isfinite(primary_height_m) or rim_height_m <= 0:
        return result

    # A fixed 50 mm grid is a layer-discovery scale, not output quantization.
    # Its origin is fixed so repeated runs and different vessel heights agree.
    edges = np.arange(-2.0, max(rim_height_m + .55, .6) + .05, .05)
    counts, edges = np.histogram(values, bins=edges)
    smooth = gaussian_filter1d(counts.astype(float), 1.5)
    if smooth.max() <= 0:
        return result
    peaks, _ = find_peaks(
        smooth, distance=5, prominence=max(float(smooth.max()) * .025, 1.0))
    center = lambda index: float((edges[index] + edges[index + 1]) / 2)
    bottom = [index for index in peaks if abs(center(index)) < .15]
    if not bottom:
        return result
    bottom_index = max(bottom, key=lambda index: smooth[index])
    bottom_height = center(bottom_index)
    local_bottom = values[np.abs(values - bottom_height) <= .075]
    if len(local_bottom) >= 24:
        # The histogram finds the layer; local robust center retains metric precision.
        bottom_height = float(np.median(local_bottom))
    result["bottom_peak_m"] = bottom_height

    positive = [index for index in peaks
                if .2 < center(index) < rim_height_m + .05]
    if positive:
        wet_index = max(positive, key=lambda index: smooth[index])
        wet_height = center(wet_index)
        ratio = float(smooth[wet_index] / smooth[bottom_index])
        result["wet_peak_m"] = wet_height
        result["wet_to_bottom_vote_ratio"] = ratio
        if smooth[bottom_index] >= 1.3 * smooth[wet_index]:
            if ratio >= .30:
                # When the robust ROI quantile independently agrees with the
                # candidate layer, retain its sub-bin metric resolution.
                chosen = (float(container_quantile_height_m)
                          if abs(wet_height - container_quantile_height_m) <= .2
                          else wet_height)
                result.update(height_m=chosen, selected=True,
                              candidate="wet_surface_peak")
            elif abs(primary_height_m - bottom_height) > .15:
                result.update(height_m=bottom_height, selected=True,
                              candidate="dry_anchored_near_zero_peak")
    elif abs(primary_height_m - bottom_height) > .15:
        result.update(height_m=bottom_height, selected=True,
                      candidate="dry_anchored_near_zero_peak")
    return result
