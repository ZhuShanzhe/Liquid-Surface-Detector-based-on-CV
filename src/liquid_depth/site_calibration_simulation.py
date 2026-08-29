from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .calibration import fit_output_calibration


@dataclass(frozen=True)
class CameraErrorProfile:
    """Conservative decomposition of vendor depth-error specifications."""

    name: str
    reference: str
    min_range_m: float
    max_range_m: float
    fixed_scale_bound: float
    fixed_offset_bound_m: float
    random_sigma_m: float
    random_sigma_fraction: float
    curvature_bound: float
    quantization_m: float = 0.001

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CAMERA_ERROR_PROFILES = {
    "stereo_2pct": CameraErrorProfile(
        name="stereo_2pct",
        reference="Orbbec Gemini 2: depth accuracy <=2% at 2 m on a high-reflectivity plane",
        min_range_m=0.10,
        max_range_m=10.0,
        fixed_scale_bound=0.015,
        fixed_offset_bound_m=0.003,
        random_sigma_m=0.001,
        random_sigma_fraction=0.010,
        curvature_bound=0.005,
    ),
    "long_baseline_stereo": CameraErrorProfile(
        name="long_baseline_stereo",
        reference="Orbbec Gemini 335L: accuracy <=1% at 2 m and <=2% at 4 m",
        min_range_m=0.10,
        max_range_m=10.0,
        fixed_scale_bound=0.008,
        fixed_offset_bound_m=0.002,
        random_sigma_m=0.001,
        random_sigma_fraction=0.004,
        curvature_bound=0.010,
    ),
    "tof_typical": CameraErrorProfile(
        name="tof_typical",
        reference="Orbbec Femto Bolt: system error <11 mm + 0.1% distance; random sigma <=17 mm",
        min_range_m=0.50,
        max_range_m=5.46,
        fixed_scale_bound=0.001,
        fixed_offset_bound_m=0.011,
        random_sigma_m=0.017,
        random_sigma_fraction=0.0,
        curvature_bound=0.001,
    ),
}


@dataclass(frozen=True)
class SiteError:
    scale: float
    offset_m: float
    curvature: float


def _sample_site_error(
    profile: CameraErrorProfile,
    rng: np.random.Generator,
) -> SiteError:
    return SiteError(
        scale=float(rng.uniform(-profile.fixed_scale_bound, profile.fixed_scale_bound)),
        offset_m=float(
            rng.uniform(-profile.fixed_offset_bound_m, profile.fixed_offset_bound_m)
        ),
        curvature=float(rng.uniform(-profile.curvature_bound, profile.curvature_bound)),
    )


def _camera_measurement(
    truth_m: float,
    profile: CameraErrorProfile,
    site: SiteError,
    rng: np.random.Generator,
) -> float:
    span = max(profile.max_range_m - profile.min_range_m, 1e-6)
    position = (truth_m - profile.min_range_m) / span
    systematic = truth_m * (
        1.0 + site.scale + site.curvature * (position - 0.5) ** 2
    )
    sigma = profile.random_sigma_m + profile.random_sigma_fraction * truth_m
    measured = systematic + site.offset_m + float(rng.normal(0.0, sigma))
    if profile.quantization_m > 0:
        measured = round(measured / profile.quantization_m) * profile.quantization_m
    return max(measured, 0.0)


def build_empirical_pools(records: list[dict]) -> dict[str, dict]:
    """Split model residuals into tuning and assessment halves by stable order."""

    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record["scenario"]), []).append(record)
    pools: dict[str, dict] = {}
    for scenario, values in grouped.items():
        ordered = sorted(values, key=lambda item: str(item.get("sample_id", "")))
        tuning = ordered[::2]
        assessment = ordered[1::2] or ordered[::2]
        tuning_residuals = [
            float(item["error_m"])
            for item in tuning
            if item.get("accepted") and item.get("error_m") is not None
        ]
        assessment_residuals = [
            float(item["error_m"])
            for item in assessment
            if item.get("accepted") and item.get("error_m") is not None
        ]
        if not assessment_residuals:
            assessment_residuals = tuning_residuals
        pools[scenario] = {
            "tuning_samples": len(tuning),
            "assessment_samples": len(assessment),
            "assessment_acceptance": sum(bool(item.get("accepted")) for item in assessment)
            / max(len(assessment), 1),
            "tuning_absolute_bias_m": float(np.median(tuning_residuals))
            if tuning_residuals
            else 0.0,
            "assessment_absolute_residuals_m": assessment_residuals,
        }
    return pools


def calibration_levels(
    profile: CameraErrorProfile,
    count: int,
    design: str,
) -> np.ndarray:
    if count < 3:
        raise ValueError("At least three calibration levels are required")
    low, high = profile.min_range_m, profile.max_range_m
    if design == "linear":
        return np.linspace(low, high, count)
    if design == "log":
        return np.geomspace(low, high, count)
    if design != "hybrid":
        raise ValueError(f"Unknown calibration design: {design}")
    logarithmic = np.geomspace(low, high, count - 1)
    arithmetic_midpoint = np.asarray([(low + high) / 2.0])
    return np.sort(np.concatenate((logarithmic, arithmetic_midpoint)))


def _measure_level(
    truth_m: float,
    frames: int,
    scenario_pool: dict,
    profile: CameraErrorProfile,
    site: SiteError,
    rng: np.random.Generator,
    minimum_accepted_frames: int,
) -> float | None:
    acceptance = float(scenario_pool["assessment_acceptance"])
    residuals = scenario_pool["assessment_absolute_residuals_m"]
    if not residuals:
        return None
    measurements = []
    for _ in range(frames):
        if rng.random() > acceptance:
            continue
        measured = _camera_measurement(truth_m, profile, site, rng)
        absolute_residual = float(rng.choice(residuals))
        measurements.append(measured + absolute_residual)
    if len(measurements) < minimum_accepted_frames:
        return None
    return float(np.median(measurements))


def _summary(errors: list[float], truths: list[float]) -> dict[str, float | int]:
    if not errors:
        return {
            "accepted_levels": 0,
            "mae_m": float("nan"),
            "rmse_m": float("nan"),
            "max_abs_error_m": float("nan"),
            "abs_rel": float("nan"),
            "within_tolerance_rate": 0.0,
        }
    absolute = np.abs(np.asarray(errors, dtype=np.float64))
    reference = np.asarray(truths, dtype=np.float64)
    tolerance = np.maximum(0.003, 0.01 * reference)
    return {
        "accepted_levels": len(errors),
        "mae_m": float(absolute.mean()),
        "rmse_m": float(np.sqrt(np.mean(absolute**2))),
        "max_abs_error_m": float(absolute.max()),
        "abs_rel": float(np.mean(absolute / np.maximum(reference, 1e-6))),
        "within_tolerance_rate": float(np.mean(absolute <= tolerance)),
    }


def simulate_site_calibration(
    records: list[dict],
    profile: CameraErrorProfile,
    *,
    trials: int,
    seed: int,
    calibration_level_count: int,
    calibration_frames: int,
    validation_frames: int,
    minimum_accepted_frames: int,
    design: str,
    scenario_bias_correction: bool = False,
) -> dict:
    """Run fixed-site calibration on ordinary frames and validate every scenario."""

    rng = np.random.default_rng(seed)
    pools = build_empirical_pools(records)
    if "ordinary" not in pools:
        raise ValueError("The residual records must include ordinary scenes")
    levels = calibration_levels(profile, calibration_level_count, design)
    scenario_errors: dict[str, list[float]] = {
        scenario: [] for scenario in sorted(pools)
    }
    scenario_truths: dict[str, list[float]] = {
        scenario: [] for scenario in sorted(pools)
    }
    successful_sites = 0
    attempted_validation_levels = trials * 3
    ordinary_bias = float(pools["ordinary"]["tuning_absolute_bias_m"])
    for _ in range(trials):
        site = _sample_site_error(profile, rng)
        predicted_calibration, known_calibration = [], []
        for level in levels:
            predicted = _measure_level(
                float(level),
                calibration_frames,
                pools["ordinary"],
                profile,
                site,
                rng,
                minimum_accepted_frames,
            )
            if predicted is not None:
                predicted_calibration.append(predicted)
                known_calibration.append(float(level))
        if len(predicted_calibration) < 3:
            continue
        calibration = fit_output_calibration(
            predicted_calibration,
            known_calibration,
        )
        successful_sites += 1
        validation_levels = np.exp(
            rng.uniform(
                np.log(profile.min_range_m),
                np.log(profile.max_range_m),
                size=3,
            )
        )
        for scenario, pool in pools.items():
            absolute_correction = (
                float(pool["tuning_absolute_bias_m"]) - ordinary_bias
                if scenario_bias_correction
                else 0.0
            )
            for truth_m in validation_levels:
                raw = _measure_level(
                    float(truth_m),
                    validation_frames,
                    pool,
                    profile,
                    site,
                    rng,
                    minimum_accepted_frames,
                )
                if raw is None:
                    continue
                raw -= absolute_correction
                estimate = calibration["scale"] * raw + calibration["offset_m"]
                scenario_errors[scenario].append(estimate - float(truth_m))
                scenario_truths[scenario].append(float(truth_m))
    by_scenario = {}
    for scenario in sorted(pools):
        values = _summary(scenario_errors[scenario], scenario_truths[scenario])
        values["level_coverage"] = values["accepted_levels"] / max(
            successful_sites * 3, 1
        )
        values["empirical_frame_acceptance"] = pools[scenario][
            "assessment_acceptance"
        ]
        values["tuning_absolute_bias_m"] = pools[scenario]["tuning_absolute_bias_m"]
        by_scenario[scenario] = values
    all_errors = [
        value for values in scenario_errors.values() for value in values
    ]
    all_truths = [
        value for values in scenario_truths.values() for value in values
    ]
    global_summary = _summary(all_errors, all_truths)
    global_summary["level_coverage"] = global_summary["accepted_levels"] / max(
        successful_sites * attempted_validation_levels / max(trials, 1) * len(pools),
        1,
    )
    return {
        "camera_profile": profile.to_dict(),
        "strategy": {
            "calibration_levels": calibration_level_count,
            "calibration_frames_per_level": calibration_frames,
            "validation_frames_per_level": validation_frames,
            "minimum_accepted_frames": minimum_accepted_frames,
            "level_design": design,
            "scenario_bias_correction": scenario_bias_correction,
        },
        "trials": trials,
        "successful_calibration_sites": successful_sites,
        "site_success_rate": successful_sites / max(trials, 1),
        "global": global_summary,
        "by_scenario": by_scenario,
    }

