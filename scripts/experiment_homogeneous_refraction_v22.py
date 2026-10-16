#!/usr/bin/env python3
"""Analytical sensor/correspondence experiment, NOT rendered-camera validation.

Truth is used by the forward simulator and scoring only. The inverse sees
frozen calibration and unlabelled optical observations. No event masks.
"""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import time
import numpy as np
from liquid_depth.homogeneous_refraction_v22 import coefficient, fit_material_prior, infer_height


def forward(rng, *, index, height, bottom, model, condition, bias, sigma):
    # A grid of independent tile representatives, not densely independent pixels.
    x, y = np.meshgrid(np.linspace(-.48, .48, 8), np.linspace(-.32, .32, 8))
    q2 = (x*x+y*y).ravel()
    n = min(1.58, index+.065) if condition == 'changed_liquid' else index
    value = height*coefficient(n, q2, model)+bias
    value += rng.normal(0, sigma, len(value)) + rng.normal(0, sigma*.15)
    # Only 2% direct-surface events for ToF. Their truth is NOT supplied to inverse.
    if model == 'nondispersive_coaxial_tof_proxy':
        top = rng.random(len(value)) < .02
        value[top] = -height + bias + rng.normal(0, sigma, top.sum())
    if condition == 'translucent_scatter':
        # Positive excess path is a proxy for scattering, not derived radiative transport.
        value += rng.exponential(.001+.0005*bottom, len(value))
    if condition == 'multipath_phase':
        if model == 'nondispersive_coaxial_tof_proxy':
            # Phase-sum stress test at 10MHz, two paths, unknown amplitude ratio.
            # Optical depth is axial and the physical phase range is slant.
            scale = np.sqrt(1+q2)
            k = 4*np.pi*1e7/299792458.
            direct = (bottom-height)*scale
            through = (bottom+height*coefficient(index, q2, model))*scale
            amplitude = rng.uniform(.25, .75, len(value))
            phasor = np.exp(1j*k*through)+amplitude*np.exp(1j*k*direct)
            phase = np.mod(np.angle(phasor), 2*np.pi)
            value = phase/k/scale-bottom+bias+rng.normal(0, sigma, len(value))
        else:
            # Wrong RGB reference matches instead of a ToF phase formula.
            bad = rng.random(len(value)) < .60
            value[bad] += .12*height
    if condition == 'lost_bottom':
        value[:] = np.nan
    dropout = rng.random(len(value)) < .15
    value[dropout] = np.nan
    return dict(value_m=value, q2=q2)


def summary(records):
    accepted = [r for r in records if r['accepted']]
    errors = [abs(r['prediction_m']-r['height_m']) for r in accepted]
    rel = [e/r['height_m'] for e, r in zip(errors, accepted)]
    passed = [e <= max(.005, .02*r['height_m']) for e, r in zip(errors, accepted)]
    return dict(frames=len(records), accepted=len(accepted),
        coverage_pct=100*len(accepted)/len(records),
        mae_mm=float(np.mean(errors)*1000) if errors else None,
        absrel_pct=float(np.mean(rel)*100) if rel else None,
        p95_mm=float(np.percentile(errors, 95)*1000) if errors else None,
        pass_pct=100*sum(passed)/len(accepted) if accepted else None,
        effective_pass_pct=100*sum(passed)/len(records),
        inaccurate_acceptance_pct=100*(len(accepted)-sum(passed))/len(records),
        evaluable_output_pct=100. if accepted else None,
        reasons=dict(Counter(reason for r in records for reason in r['reasons'])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--noise-profile', choices=('ideal_tile', 'conservative_sensor'), default='ideal_tile')
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    records, calibrations, guards = [], [], []
    models = ('nondispersive_coaxial_tof_proxy', 'rgb_reference_snell')
    for model_index, model in enumerate(models):
        for bottom_index, bottom in enumerate((.3, .8, 2., 6., 9.8)):
            for index_id, index in enumerate((1.28, 1.333, 1.42, 1.5)):
                for repeat in range(3):
                    seed = 2026101200+model_index*10000+bottom_index*1000+index_id*100+repeat
                    rng = np.random.default_rng(seed)
                    sigma = .00025+.00008*bottom
                    if args.noise_profile == 'conservative_sensor':
                        sigma = (.0015+.0007*bottom if model_index == 0 else .0005+.0006*bottom)
                    bias = rng.uniform(-.002, .002)
                    known = np.array([.15, .35, .60])*bottom
                    calibration_frames = [forward(rng, index=index, height=h, bottom=bottom,
                        model=model, condition='clear', bias=bias, sigma=sigma) for h in known]
                    # Known liquid-height labels have independent measurement error.
                    labelled = known + rng.normal(0, .0002+.001*known)
                    try:
                        prior = fit_material_prior(calibration_frames, labelled, sensor_model=model,
                            bottom_distance_m=bottom, observation_sigma_m=sigma)
                        calibration = dict(seed=seed, model=model, true_index=index, bottom_m=bottom,
                            known_height_labels_m=labelled.tolist(), prior=prior.to_dict(), accepted=True)
                    except ValueError as exc:
                        prior = None
                        calibration = dict(seed=seed, model=model, true_index=index, bottom_m=bottom,
                            accepted=False, reason=str(exc))
                    calibrations.append(calibration)
                    # A scalar affine prior uses the same calibration, but ignores ray optics.
                    means = [float(np.nanmedian(f['value_m'])) for f in calibration_frames]
                    slope, intercept = np.polyfit(labelled, means, 1)
                    # Identical known liquid levels must fail the index/bias observability gate.
                    if repeat == 0:
                        try:
                            fit_material_prior([calibration_frames[1]]*3, [known[1]]*3,
                                sensor_model=model, bottom_distance_m=bottom, observation_sigma_m=sigma)
                            guards.append(False)
                        except ValueError:
                            guards.append(True)
                    for condition in ('clear', 'translucent_scatter', 'changed_liquid',
                                      'multipath_phase', 'lost_bottom'):
                        for ratio in (.08, .12, .20, .28, .42, .52, .65, .72):
                            height = ratio*bottom
                            frame = forward(rng, index=index, height=height, bottom=bottom,
                                model=model, condition=condition, bias=bias, sigma=sigma)
                            started = time.perf_counter()
                            if prior is None:
                                result = dict(accepted=False, height_m=None, reasons=['prior_rejected'])
                            else:
                                result = infer_height(frame, prior)
                            ms = 1000*(time.perf_counter()-started)
                            base = dict(seed=seed, model=model, condition=condition, bottom_m=bottom,
                                noise_profile=args.noise_profile, observation_sigma_m=sigma,
                                height_m=height, surface_return_probability=.02,
                                timing_scope='CPU tile observations only; excludes camera/matching/render',
                                inference_ms=ms)
                            records.append(dict(base, method='few_shot_physical',
                                prediction_m=result['height_m'], accepted=result['accepted'], reasons=result['reasons']))
                            value, q = frame['value_m'], frame['q2']
                            valid = np.isfinite(value) & (value >= -3*sigma) & (q >= .01)
                            if valid.sum() >= 16:
                                affine = float((np.median(value[valid])-intercept)/slope)
                                naive = float(np.median(value[valid])/np.median(coefficient(1.333, q[valid], model)))
                            else:
                                affine = naive = None
                            for name, prediction in (('same_prior_affine', affine), ('fixed_water_index', naive)):
                                records.append(dict(base, method=name, prediction_m=prediction,
                                    accepted=prediction is not None, reasons=[] if prediction is not None else ['no_observation']))
    grouped = defaultdict(list)
    for record in records:
        grouped[(record['model'], record['condition'], record['method'])].append(record)
    grouped_summary = [dict(model=m, condition=c, method=t, **summary(rs))
                       for (m, c, t), rs in grouped.items()]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(research_only=True, hardware_qualified=False,
        production_enabled=False, experiment='analytical tile-observation inverse optics',
        noise_profile=args.noise_profile,
        not_rendered_camera_or_real_sensor_validation=True,
        calibrated_prior_frames=3, independent_test_heights=8,
        condition_thresholds_not_tuned_on_test=True,
        surface_event_masks_available_to_inverse=False,
        known_test_height_available_to_inverse=False,
        repeated_same_level_guard_passed=all(guards),
        calibrations=calibrations, summary=grouped_summary, records=records), indent=2))
    print('CALIBRATION', sum(c['accepted'] for c in calibrations), '/', len(calibrations))
    print(json.dumps(grouped_summary, indent=2))


if __name__ == '__main__':
    main()
