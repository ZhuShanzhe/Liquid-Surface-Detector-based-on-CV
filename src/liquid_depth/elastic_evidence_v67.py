"""Optional ratio-scaled evidence window for far structured-light partial loss.

The center is a robust depth-only preliminary prediction. Truth masks, labels,
RGB and future frames are never used at inference.
"""
from __future__ import annotations

import numpy as np


def elastic_evidence_window(
    height_returns_m: np.ndarray, *, camera_to_bottom_m: float,
    primary_height_m: float, dry_layer_already_selected: bool = False,
) -> dict:
    values = np.asarray(height_returns_m, dtype=float)
    values = values[np.isfinite(values)]
    result = {"height_m": float(primary_height_m), "applied": False,
              "reason": "insufficient_or_out_of_scope_evidence",
              "support_returns": len(values), "retained_returns": 0,
              "retained_fraction": None, "coarse_prior_m": None,
              "radius_m": None, "height_ratio": None,
              "quantile_gap_m": None}
    if (dry_layer_already_selected or len(values) < 24 or
            not np.isfinite(camera_to_bottom_m) or camera_to_bottom_m < 8 or
            not np.isfinite(primary_height_m)):
        return result
    coarse, upper = np.percentile(values, [50, 58])
    ratio = float(coarse / camera_to_bottom_m)
    radius = float(np.clip(.04 + .45 * abs(ratio), .05, .18))
    band = values[np.abs(values - coarse) <= radius]
    result.update(coarse_prior_m=float(coarse), radius_m=radius,
                  height_ratio=ratio, quantile_gap_m=float(upper - coarse),
                  retained_returns=len(band),
                  retained_fraction=float(len(band) / len(values)))
    if not (.05 < coarse and 0 < ratio < .35 and
            upper - coarse > 3 * radius and
            len(band) >= max(24, int(.01 * len(values)))):
        result["reason"] = "no_large_asymmetric_tail_or_insufficient_local_support"
        return result
    result.update(height_m=float(np.median(band)), applied=True,
                  reason="supported_local_evidence")
    return result
