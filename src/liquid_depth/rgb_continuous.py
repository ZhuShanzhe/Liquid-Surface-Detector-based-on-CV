"""Opt-in continuous silhouette fitting; no depth input or invented SR evidence."""

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize_scalar

from .rgb_witness import RGBContourWitness


class RGBContinuousWitness(RGBContourWitness):
    def _project_contour(self, level, matrix, pose):
        angle = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        world = np.column_stack(
            (self.rx * np.cos(angle), self.ry * np.sin(angle), np.full(len(angle), self.bottom + level))
        )
        camera = (world - pose[:3, 3]) @ pose[:3, :3]
        if (camera[:, 2] <= 0).any():
            return None
        projected = camera @ matrix.T
        return projected[:, :2] / projected[:, 2:3]

    def _fit(self, mask, matrix, pose):
        if np.count_nonzero(mask) < 24:
            return self.reference, 0.0, self.search_span, True
        signed = cv2.distanceTransform(mask, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)

        def residual(level):
            uv = self._project_contour(level, matrix, pose)
            if uv is None:
                return np.full(256, 1e6)
            return map_coordinates(
                signed, [uv[:, 1], uv[:, 0]], order=1, mode="constant", cval=-max(mask.shape)
            )

        def objective(level):
            r = abs(residual(level))
            return float(np.mean(np.where(r < 2, 0.5 * r * r, 2 * r - 2)))

        grid = self.reference + np.linspace(-self.search_span, self.search_span, 41)
        costs = np.array([objective(h) for h in grid])
        best = int(np.argmin(costs))
        if best in (0, len(grid) - 1):
            return float(grid[best]), 0.0, self.search_span, True
        fitted = minimize_scalar(
            objective, bounds=(grid[best - 1], grid[best + 1]), method="bounded", options={"xatol": 1e-6}
        )
        h = float(fitted.x)
        eps = max(1e-5, self.search_span * 1e-3)
        derivative = (residual(h + eps) - residual(h - eps)) / (2 * eps)
        sensitivity = float(np.sqrt(np.mean(derivative**2)))
        # Correlated boundary pixels are NOT independent samples: no sqrt(N) gain.
        spread = max(0.25, float(np.median(abs(residual(h)))))
        sigma = max(1e-4, spread / max(sensitivity, 1e-6))
        projected = self._project_mask(h, matrix, pose, mask.shape)
        score = np.count_nonzero(projected & mask) / max(1, np.count_nonzero(projected | mask))
        return h, float(score), sigma, (not fitted.success or sensitivity < 1e-3)

    def estimate(self, *args, **kwargs):
        result = super().estimate(*args, **kwargs)
        result["fitter"] = "continuous_signed_distance_v1"
        return result

    def to_dict(self):
        return {**super().to_dict(), "schema_version": 2, "fitter": "continuous_signed_distance_v1"}

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema_version") != 2 or payload.get("fitter") != "continuous_signed_distance_v1":
            raise ValueError("Continuous RGB reference must be calibrated for this fitter")
        return super().from_dict({**payload, "schema_version": 1})
