"""Independent empty-only desktop calibration, matching the fixed-10mm study.

This is a SIMULATION producer adapter, never a measurement shortcut. It owns
empty-scene annotations and visibility; the returned bundle contains only the
same measured geometry contract accepted by the formal meter. No current wet
height or wet semantic mask is accepted by either calibration function.
"""
import numpy as np

from .empty_scene_routes_v44 import build_scene_geometry
from .rim_geometry_v43 import build_rim_geometry

METRIC_ERROR_LIMIT_M = .010
METRIC_ERROR_SIGMA_M = METRIC_ERROR_LIMIT_M / 3.
SHAPES = {'cylinder': 'circle', 'ellipse': 'ellipse', 'rectangle': 'rectangle'}


def annotated_value(value, rng, noisy):
    if not noisy:
        return float(value)
    while True:
        error = float(rng.normal(0, METRIC_ERROR_SIGMA_M))
        if abs(error) <= METRIC_ERROR_LIMIT_M:
            return float(value) + error


def metric_record(reference, annotated, noisy):
    return dict(reference_m=float(reference), annotated_m=float(annotated),
                error_mm=1000.*float(annotated-reference), noise_applied=bool(noisy),
                error_limit_mm=10., sigma_mm=10./3.)


def empty_calibration(dry, cal, empty_visible, *, shape, radii_m, capacity_m,
                      bottom_m, noise_mode='fixed_10mm', seed=2026102197701):
    """Project simulated EMPTY annotations and reuse established geometry fit.

    Visibility of the inset bottom is tested before it can become an operator
    click. Hidden bottom outlines are never used. Reannotation retries are based
    solely on prior input validity, not on any downstream measurement error.
    """
    if noise_mode not in ('ideal', 'fixed_10mm'):
        raise ValueError('unsupported_calibration_noise_mode')
    kind = SHAPES[shape]
    rx, ry = map(float, radii_m)
    theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
    xy = (np.array([[-rx,-ry], [rx,-ry], [rx,ry], [-rx,ry]]) if kind == 'rectangle'
          else np.column_stack((rx*np.cos(theta), ry*np.sin(theta))))

    def project(points):
        camera = (np.asarray(points)-cal.camera_to_world_cv[:3,3]) @ cal.camera_to_world_cv[:3,:3]
        p = camera @ cal.intrinsics.T
        return p[:,:2]/p[:,2:]

    bottom = project(np.column_stack((xy, np.zeros(len(xy)))))
    rim = project(np.column_stack((xy, np.full(len(xy), capacity_m))))
    inset = project(np.column_stack((xy*.96, np.zeros(len(xy)))))
    visible = np.asarray(empty_visible, bool)
    if visible.shape != dry.shape:
        raise ValueError('aligned_empty_visibility_required')

    def seen(p):
        if not np.all(np.isfinite(p)):
            return False
        x, y = np.rint(p).astype(int)
        return (1 <= x < dry.shape[1]-1 and 1 <= y < dry.shape[0]-1
                and bool(visible[y-1:y+2,x-1:x+2].any()))

    bottom_visible = all(seen(p) for p in inset)
    rng = np.random.default_rng(seed)
    noisy = noise_mode == 'fixed_10mm'
    failures = []
    for attempt in range(6):
        annotations = {}

        def length(name, value):
            measured = annotated_value(value, rng, noisy)
            annotations[name] = metric_record(value, measured, noisy)
            return measured

        def pixels(x):
            return (x+rng.normal(0,.3,x.shape)).tolist() if noisy else x.tolist()

        up = cal.camera_to_world_cv[2,:3].copy()
        if noisy:
            up += rng.normal(0,.001,3)
        up /= np.linalg.norm(up)
        anchor = length('bottom_distance', bottom_m)
        wall = length('wall_height', capacity_m)
        check = project([[-rx*.5,0,0], [rx*.5,0,0]])
        spec = dict(shape=kind, bottom_outline_px=pixels(bottom), wall_height_m=wall,
            metric_anchor=dict(kind='perpendicular_distance', distance_m=anchor),
            up_camera=up.tolist(), sensor_family='tof', baseline_m=0.,
            check_spans=[dict(pixels=pixels(check), length_m=length('check_span',rx))],
            confirmed_empty=True, horizontal_flat_bottom=True, vertical_walls=True,
            fixed_camera=True, rectified_aligned=True)
        rim_spec = {**spec, 'rim_outline_px':pixels(rim),
            'measured_rim_dimensions_m':[length('rim_width',2*rx), length('rim_length',2*ry)]}
        if kind == 'rectangle' and (np.any(rim < 1.5)
                or np.any(rim[:,0] >= dry.shape[1]-1.5)
                or np.any(rim[:,1] >= dry.shape[0]-1.5)):
            edges = []
            for edge in range(4):
                t = np.linspace(0,1,65)[:,None]
                line = xy[edge]*(1-t)+xy[(edge+1)%4]*t
                pp = project(np.column_stack((line,np.full(len(line),capacity_m))))
                inside = ((pp[:,0]>=1.5)&(pp[:,0]<dry.shape[1]-1.5)
                          &(pp[:,1]>=1.5)&(pp[:,1]<dry.shape[0]-1.5))
                chosen = pp[inside][::4]
                if len(chosen)>=2:
                    edges.append(dict(edge=edge,pixels=pixels(chosen)))
            rim_spec['visible_rim_edges_px'] = edges
        try:
            bundle = (build_scene_geometry(dry,cal.intrinsics,spec,rim_spec=rim_spec)
                      if bottom_visible else build_rim_geometry(dry,cal.intrinsics,rim_spec))
            audit = dict(mode=noise_mode, attempt=attempt+1, rejected_reasons=failures,
                metric_anchor_m=anchor, measured_capacity_m=wall,
                bottom_distance_error_mm=(anchor-bottom_m)*1000.,
                metric_annotations=annotations, source=bundle['source'], quality=bundle['quality'],
                independent_of_wet_truth=True, bottom_outline_visible=bottom_visible,
                hidden_bottom_outline_used=False, metric_error_limit_mm=10.,
                distribution='zero-mean Gaussian sigma=0.010/3 m; rejection sampled within +/-0.010 m',
                click_sigma_px=.3 if noisy else 0., gravity_component_sigma=.001 if noisy else 0.)
            return bundle, audit
        except ValueError as exc:
            failures.append(str(exc))
    raise ValueError('invalid_empty_calibration_after_reannotation:'+repr(failures))


def phase_annotations(phase, audit, *, noise_mode, seed=2026102188841):
    """Measured 1-empty + 4-known liquid levels; forward parameters stay private."""
    original = np.asarray(phase['known_heights_m'],float)
    if original.shape != (5,) or original[0] != 0 or not np.all(np.diff(original)>0):
        raise ValueError('invalid_five_prior_annotation_schedule')
    if noise_mode not in ('ideal','fixed_10mm'):
        raise ValueError('unsupported_calibration_noise_mode')
    rng = np.random.default_rng(seed)
    noisy = noise_mode == 'fixed_10mm'
    for attempt in range(1,101):
        known = np.r_[0., [annotated_value(h,rng,noisy) for h in original[1:]]]
        if np.all(np.diff(known)>0) and known[-1]<=audit['measured_capacity_m']:
            inverse = {**phase, 'known_heights_m':known,
                'profile':{**phase['profile'], 'bottom_m':audit['metric_anchor_m'],
                           'container_max_height_m':audit['measured_capacity_m']}}
            details = {f'known_liquid_height_{j}':metric_record(h,k,noisy)
                       for j,(h,k) in enumerate(zip(original[1:],known[1:]),1)}
            return inverse, dict(annotation_attempt=attempt, metric_annotations=details,
                confirmed_empty_height_m=0., input_validity_only=True, current_test_truth_used=False)
    raise ValueError('phase_prior_labels_need_reannotation')


def phase_modulation_qualification(phase, current=None):
    """Check the SAME raw-observation fit pixels as the frozen phase meter.

    Never uses a fitted parameter, fit residual, wet height or measurement error.
    The old global 30-pixel check alone can accept 63 remaining fit pixels even
    though the actual inversion requires 64. This check is before inference.
    """
    from .tof_phase_refraction import decode
    decoded=[decode(c) for c in phase['prior_correlations']]
    supported=np.asarray(phase['roi'],bool)&np.all(np.stack([abs(z)>8 for z in decoded]),axis=(0,1))
    fit_ids=np.flatnonzero(supported)[::max(1,int(supported.sum()//128))][:160]
    count=None if current is None else int(np.all(abs(decode(current))>8,axis=0).ravel()[fit_ids].sum())
    reasons=[]
    if supported.sum()<64 or (count is not None and count<64):
        reasons.append('insufficient_phase_selected_modulation_support')
    return dict(eligible=not reasons,reasons=reasons,
        phase_prior_modulation_support_pixels=int(supported.sum()),
        phase_selected_current_support_pixels=count,phase_selected_fit_pixels=len(fit_ids),
        phase_preflight_checked=True,phase_qualification_pending=False,
        uses_estimate_or_estimation_error=False)
