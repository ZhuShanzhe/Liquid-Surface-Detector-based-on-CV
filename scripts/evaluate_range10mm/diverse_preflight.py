"""Input-only observability audit; valid-prior initialization failure is retained."""
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get('LIQUID_EVAL_ROOT', Path(__file__).resolve().parent))
REPO = Path('/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV')
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts')]
from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.sim_input_qualification import qualify_simulated_input
from tof_phase_simulator import generate, rays
from diverse_geometry import make_bundle, phase_inverse_inputs
from liquid_depth.tof_scene_meter import ToFSceneMeter
from liquid_depth.tof_phase_refraction import decode

states = [json.loads(p.read_text()) for p in sorted(ROOT.glob('capture-??.json'))]
ids = sorted({x['installation_id'] for s in states for x in s['qualified']})
geometry_errors = []
for gid in ids:
    for variant in ('ideal', 'operator_noise'):
        try:
            make_bundle(ROOT, gid, variant)
        except ValueError as exc:
            geometry_errors.append(dict(installation_id=gid, variant=variant, error=str(exc)))
(ROOT/'preflight-geometry.json').write_text(json.dumps(
    dict(installations=len(ids), variants=2, errors=geometry_errors), indent=2))
if geometry_errors:
    raise RuntimeError('Fix invalid empty calibration before inference: '+repr(geometry_errors[:3]))

phase_bad = []
annotation_errors = []
initialization_failures = []
prior_audits = {}
for state in states:
    if state['category'] not in ('transparent', 'translucent'):
        continue
    accepted = []
    for item in state['qualified']:
        folder = Path(item['frame'])
        req = json.loads((folder/'request.json').read_text())
        gid = item['installation_id']; material = req['material']
        rng = np.random.default_rng(2026102183731+gid*7727+(material.endswith('translucent')))
        physical = dict(
            seed=2026102183731+gid*7727+(material.endswith('translucent')),
            index=float(rng.uniform(1.31,1.45)), surface_ratio=float(rng.uniform(.03,.12)),
            extinction=float(rng.uniform(.02,.08)) if material.endswith('translucent') else 0.,
            scatter_strength=float(rng.uniform(.08,.25)) if material.endswith('translucent') else 0.,
        )
        empty = ROOT/'empty'/f'{gid:04d}'
        cal = HeightCalibration.from_simulation(json.loads((empty/'metadata.json').read_text()))
        pp = dict(
            bottom_m=req['bottom_m'], container_max_height_m=req['capacity_m'],
            fx_px=float(cal.intrinsics[0,0]), center_q=[0.,0.],
            frequencies_hz=[20e6,30e6,50e6], fixed_camera=True,
            sensor_profile='amcw_itof_phase3_formal_v1',
        )
        c, roi = generate('tof', pp, physical, req['height_m'], 1001+item['attempt'])
        sem = np.load(folder/'offline_geometry.npz')['semantic']
        roi &= np.load(empty/'offline_geometry.npz')['semantic'][111:159,208:272] == 2
        c[:,:,(sem[111:159,208:272] != 2)] = 1200
        q = qualify_simulated_input(
            np.load(folder/'depth_m.npy'), sem, material=material,
            correlations=c, phase_roi=roi,
        )
        known = np.array([0,.1,.35,.65,.95])*req['capacity_m']
        priors = [generate('tof',pp,physical,float(h),j)[0] for j,h in enumerate(known)]
        supported = roi & np.all(np.stack([abs(decode(p))>8 for p in priors]), axis=(0,1))
        fit_ids = np.flatnonzero(supported)[::max(1,int(supported.sum()//128))][:160]
        supported_current = int(np.all(abs(decode(c))>8, axis=0).ravel()[fit_ids].sum())
        q.update(
            phase_prior_modulation_support_pixels=int(supported.sum()),
            phase_selected_current_support_pixels=supported_current,
            phase_qualification_pending=False, phase_preflight_checked=True,
        )
        if supported.sum() < 64 or supported_current < 64:
            q = dict(q, eligible=False,
                     reasons=q['reasons']+['insufficient_phase_selected_modulation_support'])

        # An inverse-model fitting failure on legal, observable priors is not
        # evidence that the input is meaningless. Preserve it for end-to-end
        # initialization success / coverage metrics, instead of deleting it.
        prior_key = (gid, material)
        if q['eligible'] and prior_key not in prior_audits:
            phase = dict(profile=pp, ray_coordinates=rays(pp), roi=roi,
                         prior_correlations=priors, known_heights_m=known)
            audits = {}
            for variant in ('ideal', 'operator_noise'):
                _, _, audit = make_bundle(ROOT, gid, variant)
                try:
                    inverse, annotation = phase_inverse_inputs(phase, audit, gid, material, variant)
                except ValueError as exc:
                    failure = dict(installation_id=gid, material=material, variant=variant,
                                   error=str(exc), reason='invalid_prior_annotation')
                    annotation_errors.append(failure)
                    audits[variant] = dict(status='annotation_needs_repair', **failure)
                    continue
                try:
                    ToFSceneMeter.from_phase_priors(sensor='tof', scene=material, **inverse)
                    audits[variant] = dict(status='initialized', annotation=annotation)
                except Exception as exc:
                    failure = dict(
                        installation_id=gid, material=material, variant=variant,
                        error=str(exc), exception_type=type(exc).__name__,
                        reason='initialization_failed_on_legal_observable_priors',
                        input_excluded=False, wet_measurement_error_used=False,
                    )
                    initialization_failures.append(failure)
                    audits[variant] = dict(status='initialization_failed', annotation=annotation, **failure)
            prior_audits[prior_key] = audits
        if q['eligible']:
            item['qualification'].update(q)
            item['preflight_phase_initialization'] = prior_audits.get(prior_key, {})
            accepted.append(item)
        else:
            phase_bad.append(dict(frame=str(folder), category=state['category'], qualification=q))
            state['skipped'].append(dict(frame=str(folder), request=req,
                                         qualification=q, measurement_calls=0))
    state['qualified'] = accepted
    (ROOT/f"capture-{state['category_id']:02d}.json").write_text(json.dumps(state))

report = dict(
    rejected=phase_bad, annotation_errors=annotation_errors,
    initialization_failures=initialization_failures,
    initialization_failure_policy='retained as eligible; report initialization success and end-to-end coverage',
    qualification_policy='objective geometry, visible ROI and measured modulation support only; never wet error or fit residual',
)
(ROOT/'preflight-phase.json').write_text(json.dumps(report, indent=2))
if annotation_errors:
    raise RuntimeError('Repair invalid phase annotations before inference: '+repr(annotation_errors[:3]))
if phase_bad:
    raise RuntimeError('Phase-only objective input exclusions require replenishment before inference')
(ROOT/'preflight.complete').write_text('true')
print('PREFLIGHT_OK',len(ids),'RETAINED_INITIALIZATION_FAILURES',len(initialization_failures),flush=True)
