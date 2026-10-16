"""Fallback empty-vessel geometry from a visible rim and independent metrology.

This is not bottom-depth recovery. Use only when bottom returns are insufficient:
the operator must identify the empty-vessel rim, measure its internal dimensions
and wall height, and provide independently calibrated gravity and perpendicular
camera-to-bottom distance. No hidden simulator pose or test liquid depth enters.
"""
import cv2
import numpy as np

from .empty_geometry_v41 import _intersect, _pixels, _project
from .partial_rim_geometry import complete_rectangular_rim


def build_rim_geometry(empty_depth_m, intrinsics, spec):
    for flag in ('confirmed_empty', 'horizontal_flat_bottom', 'vertical_walls',
                 'fixed_camera', 'rectified_aligned'):
        if spec.get(flag) is not True:
            raise ValueError('installation_confirmation_required:' + flag)
    raw = np.asarray(empty_depth_m, float)
    k = np.asarray(intrinsics, float)
    if raw.ndim != 2 or k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('valid_aligned_empty_frame_and_intrinsics_required')
    kind = spec.get('shape')
    if kind not in ('circle', 'ellipse', 'rectangle'):
        raise ValueError('unsupported_container_shape')
    partial_quality = {}
    if spec.get('visible_rim_edges_px'):
        if kind != 'rectangle':
            raise ValueError('partial_rim_only_supports_rectangle')
        gravity_for_partial = np.asarray(spec.get('up_camera'), float)
        if (gravity_for_partial.shape != (3,) or not np.isfinite(gravity_for_partial).all()
                or np.linalg.norm(gravity_for_partial) < .5):
            raise ValueError('independently_calibrated_gravity_required')
        gravity_for_partial /= np.linalg.norm(gravity_for_partial)
        anchor_for_partial = spec.get('metric_anchor', {})
        if anchor_for_partial.get('kind') != 'perpendicular_distance':
            raise ValueError('independent_perpendicular_bottom_distance_required')
        rim_distance_for_partial = float(anchor_for_partial['distance_m']) - float(spec['wall_height_m'])
        if not np.isfinite(rim_distance_for_partial) or rim_distance_for_partial <= 0:
            raise ValueError('invalid_bottom_distance_or_wall_height')
        outline, partial_quality = complete_rectangular_rim(
            raw.shape, k, spec, gravity_for_partial, rim_distance_for_partial)
    else:
        outline = _pixels(spec.get('rim_outline_px', []), raw.shape,
                          4 if kind == 'rectangle' else 8)
    if kind == 'rectangle' and len(outline) != 4:
        raise ValueError('rectangle_requires_four_rim_corners')
    hull = cv2.convexHull(outline.astype(np.float32)).reshape(-1, 2)
    area = float(cv2.contourArea(hull))
    if area < 256:
        raise ValueError('rim_outline_too_small')
    if kind == 'rectangle' and len(hull) != 4:
        raise ValueError('rim_rectangle_not_convex')
    gravity = np.asarray(spec.get('up_camera'), float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity) < .5:
        raise ValueError('independently_calibrated_gravity_required')
    up = gravity / np.linalg.norm(gravity)
    if up[2] >= 0 or -up[2] < np.cos(np.deg2rad(60)):
        raise ValueError('outside_top_camera_geometry')
    anchor = spec.get('metric_anchor', {})
    if anchor.get('kind') != 'perpendicular_distance':
        raise ValueError('independent_perpendicular_bottom_distance_required')
    distance = float(anchor.get('distance_m', np.nan))
    wall = float(spec.get('wall_height_m', np.nan))
    if not np.isfinite(distance) or not np.isfinite(wall) or not 0 < wall < distance <= 11:
        raise ValueError('invalid_bottom_distance_or_wall_height')
    rim_distance = distance - wall
    sensor = spec.get('sensor_family', 'active_stereo')
    baseline = float(spec.get('baseline_m', 0))
    if sensor not in ('active_stereo', 'structured_light', 'tof'):
        raise ValueError('unsupported_sensor_family')
    if (sensor == 'tof' and baseline != 0) or (sensor != 'tof' and baseline <= 0):
        raise ValueError('actual_stereo_baseline_required_or_zero_for_tof')

    points = _intersect(outline, k, up, rim_distance)
    xaxis = np.array([1., 0., 0.])
    xaxis -= up * np.dot(xaxis, up)
    xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(up, xaxis)
    closest_rim = -rim_distance * up
    xy = np.column_stack(((points - closest_rim) @ xaxis,
                          (points - closest_rim) @ yaxis)).astype('float32')
    if kind == 'circle':
        a = np.column_stack((2 * xy, np.ones(len(xy))))
        b = (xy ** 2).sum(axis=1)
        cx, cy, intercept = np.linalg.lstsq(a, b, rcond=None)[0]
        rx = ry = float(np.sqrt(max(intercept + cx * cx + cy * cy, 1e-12)))
        angle = 0.
    else:
        fit = cv2.minAreaRect(xy) if kind == 'rectangle' else cv2.fitEllipse(xy)
        (cx, cy), (width, length), angle = fit
        rx, ry = float(width / 2), float(length / 2)
    if min(rx, ry) <= .001:
        raise ValueError('degenerate_rim_dimensions')
    theta = np.deg2rad(angle)
    ex = np.cos(theta) * xaxis + np.sin(theta) * yaxis
    ey = np.cross(up, ex)
    origin_rim = closest_rim + cx * xaxis + cy * yaxis
    origin_bottom = origin_rim - wall * up
    basis = np.column_stack((ex, ey, up))
    pose = np.eye(4)
    pose[:3, :3] = basis.T
    pose[:3, 3] = -basis.T @ origin_bottom
    normalized = np.column_stack(((points - origin_rim) @ ex / rx,
                                   (points - origin_rim) @ ey / ry))
    radial = np.max(abs(normalized), axis=1) if kind == 'rectangle' else np.linalg.norm(normalized, axis=1)
    outline_error = float(np.percentile(abs(radial - 1), 95))
    if outline_error > .2:
        raise ValueError('rim_outline_inconsistent_with_selected_shape')
    if kind != 'rectangle':
        az = np.sort(np.mod(np.arctan2(normalized[:, 1], normalized[:, 0]), 2 * np.pi))
        if max(np.diff(np.r_[az, az[0] + 2 * np.pi])) > np.deg2rad(100):
            raise ValueError('incomplete_rim_boundary')
    measured = np.asarray(spec.get('measured_rim_dimensions_m', []), float)
    if measured.shape != (2,) or not np.isfinite(measured).all() or min(measured) <= 0:
        raise ValueError('two_independently_measured_rim_dimensions_required')
    fit_size = np.sort([2 * rx, 2 * ry])
    measured = np.sort(measured)
    rel = np.abs(fit_size - measured) / measured
    if np.max(rel) > .08:
        raise ValueError('rim_dimension_crosscheck_failed')
    # Metric dimensions, measured independently of the rim clicks, provide the
    # scale when the rim itself is only a few dozen pixels wide. The click fit
    # still determines location/orientation and must agree within 8%.
    if rx <= ry:
        rx, ry = float(measured[0] / 2), float(measured[1] / 2)
    else:
        rx, ry = float(measured[1] / 2), float(measured[0] / 2)
    exponent = 16. if kind == 'rectangle' else 2.
    if kind == 'rectangle':
        contour = np.array([[-rx, -ry], [rx, -ry], [rx, ry], [-rx, ry]])
    else:
        az = np.linspace(0, 2 * np.pi, 65)[:-1]
        contour = np.column_stack((rx * np.cos(az), ry * np.sin(az)))
    def camera_contour(height):
        return origin_bottom + contour[:, 0, None] * ex + contour[:, 1, None] * ey + height * up
    return {
        'schema_version': 1, 'source': 'empty_rim_plus_independent_metric_and_gravity',
        'shape': kind, 'sensor_family': sensor,
        'image_size': [raw.shape[1], raw.shape[0]], 'baseline_m': baseline,
        'rim_z_m': wall,
        'calibration': {'intrinsics': k.tolist(), 'camera_to_world_cv': pose.tolist(),
                        'bottom_z_m': 0., 'radii_m': [rx, ry],
                        'footprint_exponent': exponent},
        'bottom_plane_camera': {'up': up.tolist(),
                                'perpendicular_distance_m': distance},
        'internal_dimensions_m': {'width': 2 * rx, 'length': 2 * ry, 'height': wall},
        'projected_bottom_outline_px': _project(camera_contour(0), k),
        'projected_rim_outline_px': _project(camera_contour(wall), k),
        'operator_inputs': spec,
        'quality': {'normal_source': 'independent_gravity',
                    'bottom_depth_observed': False,
                    'rim_outline_area_px2': area,
                    'rim_outline_relative_residual_p95': outline_error,
                    'rim_dimension_crosscheck_max_rel': float(max(rel)),
                    'camera_axis_tilt_deg': float(np.degrees(np.arccos(np.clip(-up[2], -1, 1)))),
                    **partial_quality},
        'warnings': ['bottom_unobserved_requires_independent_field_validation'] +
                    (['rim_click_metric_disagreement_over_3_percent'] if max(rel) > .03 else []) +
                    (['partial_rim_extrapolation_requires_field_review'] if partial_quality else []),
        'production_qualified': False,
        'supported_routes': [],
        'continuous_motion_supported': False,
        'rectangle_footprint': 'conservative_superellipse_exponent16' if kind == 'rectangle' else None,
    }
