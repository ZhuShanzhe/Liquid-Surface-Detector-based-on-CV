#!/usr/bin/env python3
"""Priority stereo/structured-light analytical Snell calibration experiment."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
from liquid_depth.stereo_refraction_v22 import apparent_contraction, fit_stereo_prior, infer_stereo_height
from experiment_homogeneous_refraction_v22 import summary


def forward(rng, h, n, bottom, baseline, bias, pixel_sigma, tile_pixels, condition, ideal):
    q = np.linspace(-.45, .45, 64)
    changed_n = min(1.58, n+.065) if condition == 'changed_liquid' else n
    true = apparent_contraction(h, changed_n, q, bottom, baseline)
    z = bottom-true
    if ideal:
        sigma = .00025+.00008*bottom
        measured = true + rng.normal(0, sigma, len(q)) + rng.normal(0, .15*sigma)
    else:
        fx = 500.
        disparity = fx*baseline/z
        # Median aggregation of independent pixel matching noise; this assumption
        # is optimistic if vendor filtering makes pixels correlated.
        disparity += rng.normal(0, pixel_sigma*np.sqrt(np.pi/(2*tile_pixels)), len(q))
        disparity += rng.normal(0, .002)  # residual frame-common disparity drift
        measured = bottom-fx*baseline/np.maximum(disparity, .05)
    measured += bias
    # Only 2% sampled depth belongs to direct surface, no GT return mask supplied.
    top = rng.random(len(q)) < .02
    measured[top] = h + bias + rng.normal(0, .0005, top.sum())
    if condition == 'translucent_scatter':
        # Structured-matching corruption proxy, not a radiative-transfer solution.
        bad = rng.random(len(q)) < .25
        measured[bad] += rng.normal(.01*bottom, .02*bottom, bad.sum())
    if condition == 'lost_bottom':
        measured[:] = np.nan
    measured[rng.random(len(q)) < .15] = np.nan
    return dict(qx=q, value_m=measured)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--profile', choices=('ideal', 'pixel_64', 'pixel_1024'), required=True)
    args = ap.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    records, priors = [], []
    for fi, (family, baseline, pixel_sigma) in enumerate((('active_stereo', .055, .18), ('structured_light', .035, .26))):
        for bi, bottom in enumerate((.3, .8, 2., 6., 9.8)):
            for ni, n in enumerate((1.28, 1.333, 1.42, 1.5)):
                for repeat in range(2):
                    seed = 2026101300+fi*10000+bi*1000+ni*100+repeat
                    rng = np.random.default_rng(seed)
                    tile_pixels = 1024 if args.profile == 'pixel_1024' else 64
                    ideal = args.profile == 'ideal'
                    sigma = (.00025+.00008*bottom if ideal else
                             bottom**2/(500*baseline)*np.sqrt(pixel_sigma**2*np.pi/(2*tile_pixels)+.002**2))
                    bias = rng.uniform(-.0015, .0015)
                    known = np.array([.15, .35, .60])*bottom
                    fs = [forward(rng, h, n, bottom, baseline, bias, pixel_sigma, tile_pixels, 'clear', ideal)
                          for h in known]
                    labels = known+rng.normal(0, .0002+.001*known)
                    try:
                        prior = fit_stereo_prior(fs, labels, sensor_family=family, bottom_m=bottom,
                            baseline_m=baseline, sigma_m=max(.0001, sigma))
                        priors.append(dict(seed=seed, family=family, bottom_m=bottom, true_index=n,
                            accepted=True, prior=prior.to_dict(), known_height_labels_m=labels.tolist()))
                    except ValueError as exc:
                        prior = None
                        priors.append(dict(seed=seed, family=family, bottom_m=bottom, true_index=n,
                            accepted=False, reason=str(exc)))
                    medians = [np.nanmedian(f['value_m']) for f in fs]
                    slope, intercept = np.polyfit(labels, medians, 1)
                    for condition in ('clear', 'translucent_scatter', 'changed_liquid', 'lost_bottom'):
                        for ratio in (.08, .12, .20, .28, .42, .52, .65, .72):
                            h = bottom*ratio
                            observed = forward(rng, h, n, bottom, baseline, bias, pixel_sigma,
                                               tile_pixels, condition, ideal)
                            start = time.perf_counter()
                            result = infer_stereo_height(observed, prior) if prior is not None else dict(
                                accepted=False, height_m=None, reasons=['prior_rejected'])
                            base = dict(seed=seed, family=family, condition=condition, noise_profile=args.profile,
                                bottom_m=bottom, height_m=h, assumed_fx_px=500., baseline_m=baseline,
                                assumed_raw_disparity_noise_px=pixel_sigma, tile_pixels=tile_pixels,
                                surface_return_probability=.02, observation_sigma_upper_bound_m=sigma,
                                inference_ms=1000*(time.perf_counter()-start),
                                timing_scope='CPU analytic inversion only, not camera/ROI/matching')
                            records.append(dict(base, method='few_shot_stereo_physical',
                                prediction_m=result['height_m'], accepted=result['accepted'], reasons=result['reasons']))
                            values = observed['value_m']
                            valid = np.isfinite(values)
                            median = float(np.median(values[valid])) if valid.sum() >= 16 else None
                            for name, pred in (('same_prior_affine', (median-intercept)/slope if median is not None else None),
                                               ('raw_points_as_surface', median)):
                                records.append(dict(base, method=name, prediction_m=pred,
                                    accepted=pred is not None, reasons=[] if pred is not None else ['no_observation']))
    grouped = defaultdict(list)
    ranged = defaultdict(list)
    for r in records:
        grouped[(r['family'], r['condition'], r['method'])].append(r)
        ranged[(r['family'], r['condition'], r['method'], r['bottom_m'])].append(r)
    summaries = [dict(family=f, condition=c, method=m, **summary(rs)) for (f, c, m), rs in grouped.items()]
    by_range = [dict(family=f, condition=c, method=m, camera_to_bottom_m=b, **summary(rs))
                for (f, c, m, b), rs in ranged.items()]
    # Also show un-gated candidates; coverage must not hide the actual error.
    candidates = []
    for (f, c, m), rs in grouped.items():
        if m != 'few_shot_stereo_physical':
            continue
        available = [dict(r, accepted=r['prediction_m'] is not None) for r in rs]
        candidates.append(dict(family=f, condition=c, **summary(available)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(research_only=True, hardware_qualified=False, production_enabled=False,
        analytical_epipolar_plane_only=True, no_rendered_or_hardware_matching_validation=True,
        current_truth_height_or_event_mask_available_to_inverse=False,
        profile=args.profile, priors=priors, summary=summaries, by_range=by_range,
        ungated_candidates=candidates, records=records), indent=2))
    print('PRIORS', sum(p['accepted'] for p in priors), '/', len(priors))
    for row in summaries:
        if row['method'] == 'few_shot_stereo_physical':
            print(row)


if __name__ == '__main__':
    main()
