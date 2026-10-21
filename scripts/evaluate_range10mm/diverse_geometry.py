"""Empty-only calibration with fixed +/-10 mm independent metric annotations."""
import json
from pathlib import Path
import numpy as np
from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.empty_scene_routes_v44 import build_scene_geometry
from liquid_depth.rim_geometry_v43 import build_rim_geometry

METRIC_ERROR_LIMIT_M = .010
METRIC_ERROR_SIGMA_M = METRIC_ERROR_LIMIT_M / 3.
CALIBRATION_ERROR_POLICY = (
    'every nonzero metric annotation: zero-mean Gaussian sigma=0.010/3 m, '
    'rejection-sampled within [-0.010,+0.010] m; not range-proportional; '
    'confirmed empty liquid height remains zero'
)


def calibration_value(value, rng):
    """Draw annotation error without using any wet-frame measurement result."""
    while True:
        error = float(rng.normal(0, METRIC_ERROR_SIGMA_M))
        if abs(error) <= METRIC_ERROR_LIMIT_M:
            return float(value) + error


def metric_record(reference, annotated, noisy):
    return dict(
        reference_m=float(reference), annotated_m=float(annotated),
        error_m=float(annotated-reference), error_mm=1000.*float(annotated-reference),
        error_limit_mm=1000.*METRIC_ERROR_LIMIT_M,
        sigma_mm=1000.*METRIC_ERROR_SIGMA_M, noise_applied=bool(noisy),
    )


def phase_inverse_inputs(phase, audit, gid, material, variant):
    """Only the four known-height labels and measured geometry reach inversion."""
    original = np.asarray(phase['known_heights_m'], dtype=float)
    if len(original) != 5 or original[0] != 0 or not np.all(np.diff(original) > 0):
        raise ValueError('invalid_five_prior_annotation_schedule')
    if variant == 'ideal':
        details = {f'known_liquid_height_{j}': metric_record(v, v, False)
                   for j, v in enumerate(original[1:], 1)}
        return phase, dict(
            annotation_attempt=1, errors_m=[0., 0., 0., 0.],
            metric_annotations=details, error_limit_mm=10., confirmed_empty_height_m=0.,
            input_validity_only=True, calibration_error_policy=CALIBRATION_ERROR_POLICY,
        )
    if variant != 'operator_noise':
        raise ValueError('unknown_calibration_variant:'+str(variant))
    rng = np.random.default_rng(2026102188841+gid*9011+(material.endswith('translucent')))
    cap = audit['measured_capacity_m']
    for attempt in range(1, 101):
        known = np.r_[0., [calibration_value(v, rng) for v in original[1:]]]
        # These are annotation validity checks, never a wet-frame error check.
        if np.all(np.diff(known) > 0) and known[-1] <= cap:
            details = {f'known_liquid_height_{j}': metric_record(v, k, True)
                       for j, (v, k) in enumerate(zip(original[1:], known[1:]), 1)}
            inverse = {
                **phase,
                'profile': {**phase['profile'], 'bottom_m': audit['metric_anchor_m'],
                            'container_max_height_m': cap},
                'known_heights_m': known,
            }
            return inverse, dict(
                annotation_attempt=attempt, errors_m=(known[1:]-original[1:]).tolist(),
                metric_annotations=details, error_limit_mm=10., confirmed_empty_height_m=0.,
                input_validity_only=True, calibration_error_policy=CALIBRATION_ERROR_POLICY,
            )
    raise ValueError('phase_prior_labels_need_reannotation')


def make_bundle(root, gid, variant):
    if variant not in ('ideal', 'operator_noise'):
        raise ValueError('unknown_calibration_variant:'+str(variant))
    empty = Path(root)/'empty'/f'{gid:04d}'
    req = json.loads((empty/'request.json').read_text())
    cal = HeightCalibration.from_simulation(json.loads((empty/'metadata.json').read_text()))
    dry = np.load(empty/'dry_depth_m.npy')
    rx = req['radius_m']; ry = rx*req['aspect']; cap = req['capacity_m']
    kind = {'cylindrical':'circle', 'elliptical':'ellipse',
            'rectangular_tank':'rectangle'}[req['shape']]
    theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
    xy = (np.array([[-rx,-ry], [rx,-ry], [rx,ry], [-rx,ry]]) if kind == 'rectangle'
          else np.column_stack((rx*np.cos(theta), ry*np.sin(theta))))

    def project(points):
        cam = (np.asarray(points)-cal.camera_to_world_cv[:3,3])@cal.camera_to_world_cv[:3,:3]
        p = cam@cal.intrinsics.T
        return p[:,:2]/p[:,2:]

    bottom = project(np.column_stack((xy, np.zeros(len(xy)))))
    rim = project(np.column_stack((xy, np.full(len(xy), cap))))
    # A hidden bottom boundary is not an operator click. Retain the existing
    # visible-rim + measured-dimension route when the bottom outline is hidden.
    visible = np.load(empty/'offline_geometry.npz')['semantic'] == 2
    inset = project(np.column_stack((xy*.96, np.zeros(len(xy)))))

    def seen(p):
        if not np.all(np.isfinite(p)):
            return False
        x, y = np.rint(p).astype(int)
        return (1 <= x < dry.shape[1]-1 and 1 <= y < dry.shape[0]-1
                and bool(visible[y-1:y+2,x-1:x+2].any()))

    bottom_visible = all(seen(p) for p in inset)
    rng = np.random.default_rng(2026102197701+gid*9973)
    noisy = variant == 'operator_noise'; errors = []
    for attempt in range(6):
        metric_annotations = {}

        def length(name, d):
            annotated = calibration_value(d, rng) if noisy else float(d)
            metric_annotations[name] = metric_record(d, annotated, noisy)
            return annotated

        def pixels(x):
            return (x+rng.normal(0,.3,x.shape)).tolist() if noisy else x.tolist()

        up = cal.camera_to_world_cv[2,:3].copy()
        if noisy:
            up += rng.normal(0,.001,3)
        up /= np.linalg.norm(up)
        anchor = length('bottom_distance', req['bottom_m'])
        wall = length('wall_height', cap)
        check = project([[-rx*.5,0,0], [rx*.5,0,0]])
        spec = dict(
            shape=kind, bottom_outline_px=pixels(bottom), wall_height_m=wall,
            metric_anchor=dict(kind='perpendicular_distance', distance_m=anchor),
            up_camera=up.tolist(), sensor_family='tof', baseline_m=0.,
            check_spans=[dict(pixels=pixels(check), length_m=length('check_span', rx))],
            confirmed_empty=True, horizontal_flat_bottom=True, vertical_walls=True,
            fixed_camera=True, rectified_aligned=True,
        )
        rimspec = {
            **spec, 'rim_outline_px':pixels(rim),
            'measured_rim_dimensions_m':[length('rim_width', 2*rx), length('rim_length', 2*ry)],
        }
        if kind == 'rectangle' and (np.any(rim < 1.5)
                or np.any(rim[:,0] >= dry.shape[1]-1.5)
                or np.any(rim[:,1] >= dry.shape[0]-1.5)):
            edges = []
            for edge in range(4):
                t = np.linspace(0,1,65)[:,None]
                line = xy[edge]*(1-t)+xy[(edge+1)%4]*t
                pp = project(np.column_stack((line, np.full(len(line), cap))))
                inside = ((pp[:,0]>=1.5)&(pp[:,0]<dry.shape[1]-1.5)
                          &(pp[:,1]>=1.5)&(pp[:,1]<dry.shape[0]-1.5))
                chosen = pp[inside][::4]
                if len(chosen) >= 2:
                    edges.append(dict(edge=edge, pixels=pixels(chosen)))
            rimspec['visible_rim_edges_px'] = edges
        try:
            bundle = (build_scene_geometry(dry, cal.intrinsics, spec, rim_spec=rimspec)
                      if bottom_visible else build_rim_geometry(dry, cal.intrinsics, rimspec))
            audit = dict(
                variant=variant, installation_id=gid, attempt=attempt+1,
                rejected_reasons=errors, metric_anchor_m=anchor,
                anchor_error_mm=1000*(anchor-req['bottom_m']),
                source=bundle['source'], quality=bundle['quality'],
                independent_of_wet_truth=True, bottom_outline_visible=bottom_visible,
                hidden_bottom_outline_used=False, calibration_error_policy=CALIBRATION_ERROR_POLICY,
                measured_capacity_m=wall, anchor_error_limit_mm=10.,
                metric_error_limit_mm=10., metric_annotations=metric_annotations,
                click_sigma_px=.3, gravity_component_sigma=.001,
            )
            p = Path(root)/'calibration'/variant
            p.mkdir(parents=True, exist_ok=True)
            (p/f'{gid:04d}.json').write_text(json.dumps(
                dict(bundle=bundle, audit=audit, spec=spec, rim_spec=rimspec)))
            return bundle, dry, audit
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError('invalid_empty_calibration_after_reannotation:'+repr(errors))
