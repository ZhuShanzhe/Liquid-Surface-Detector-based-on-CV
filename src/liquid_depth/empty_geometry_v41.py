"""Semi-automatic empty-vessel geometry. No test heights or hidden CAD inputs.

Supported: visible flat horizontal bottom, vertical circular/elliptical/rectangular
vessel, fixed rectified camera. An independent metric span or perpendicular
camera-to-bottom distance is mandatory: biased raw depth is NOT its own ruler.
"""
import cv2
import numpy as np
from .liquid_height_metrics import HeightCalibration


def _pixels(points,shape,minimum=2):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<minimum or not np.isfinite(p).all():
        raise ValueError('insufficient_finite_image_points')
    if np.any(p<0) or np.any(p[:,0]>=shape[1]) or np.any(p[:,1]>=shape[0]):
        raise ValueError('selected_points_outside_image')
    return p


def _rays(p,k):
    return np.column_stack(((p[:,0]-k[0,2])/k[0,0],(p[:,1]-k[1,2])/k[1,1],np.ones(len(p))))


def _plane(raw,k,mask):
    valid=mask&np.isfinite(raw)&(raw>0)
    y,x=np.nonzero(valid)
    if len(x)<128 or len(x)<.15*mask.sum():raise ValueError('bottom_depth_support_insufficient')
    ids=np.linspace(0,len(x)-1,min(12000,len(x))).astype(int)
    x,y=x[ids],y[ids];a=_rays(np.column_stack((x,y)),k);v=1/raw[y,x]
    rng=np.random.default_rng(4101);best=np.linalg.lstsq(a,v,rcond=None)[0]
    score=float(np.median(abs(a@best-v)))
    for _ in range(64):
        ii=rng.choice(len(v),3,replace=False)
        if np.linalg.cond(a[ii])>1e5:continue
        p=np.linalg.solve(a[ii],v[ii]);cost=float(np.median(abs(a@p-v)))
        if cost<score:score,best=cost,p
    for _ in range(6):
        residual=v-a@best;med=np.median(residual)
        sigma=max(1e-6,float(np.median(abs(residual-med))*1.4826))
        w=np.minimum(1.,2.5*sigma/np.maximum(abs(residual),1e-12))
        best=np.linalg.solve(a.T@(a*w[:,None]),a.T@(w*v))
    if best[2]<=0 or not np.isfinite(best).all():raise ValueError('bottom_plane_not_in_front_of_camera')
    n=-best/np.linalg.norm(best);distance=1/np.linalg.norm(best)
    if -n[2]<np.cos(np.deg2rad(60)):raise ValueError('outside_top_camera_geometry')
    cells=(x-x.min())*8//max(int(np.ptp(x))+1,1)+8*((y-y.min())*8//max(int(np.ptp(y))+1,1))
    if len(np.unique(cells))<12:raise ValueError('bottom_depth_spatial_coverage_insufficient')
    return n,float(distance),dict(bottom_valid_points=int(valid.sum()),plane_fit_points=len(v),
        bottom_valid_fraction=float(valid.sum()/max(mask.sum(),1)),spatial_cells8=len(np.unique(cells)),
        inverse_depth_mad_per_m=float(sigma),raw_plane_distance_m=float(distance))


def _intersect(points,k,up,distance):
    rays=_rays(points,k);den=-(rays@up)
    if np.any(den<=1e-5):raise ValueError('selected_ray_parallel_to_bottom')
    return rays*(distance/den)[:,None]


def _span(anchor,k,up,shape):
    p=_pixels(anchor.get('pixels',[]),shape,2)
    if len(p)!=2:raise ValueError('metric_span_requires_two_endpoints')
    length=float(anchor['length_m'])
    if not np.isfinite(length) or length<=0:raise ValueError('positive_measured_span_required')
    unit=_intersect(p,k,up,1.)
    factor=float(np.linalg.norm(unit[1]-unit[0]))
    if np.linalg.norm(p[1]-p[0])<20 or factor<=1e-6:raise ValueError('metric_span_too_short_in_image')
    return p,length,factor


def _project(points,k):
    a=np.asarray(points)@k.T
    if np.any(a[:,2]<=0):raise ValueError('model_projection_behind_camera')
    return (a[:,:2]/a[:,2:]).tolist()


def build_geometry(empty_depth_m,intrinsics,spec):
    for flag in ('confirmed_empty','horizontal_flat_bottom','fixed_camera','rectified_aligned'):
        if spec.get(flag) is not True:raise ValueError('installation_confirmation_required:'+flag)
    raw=np.asarray(empty_depth_m,float);k=np.asarray(intrinsics,float)
    if raw.ndim!=2 or k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:
        raise ValueError('metric_depth_and_valid_intrinsics_required')
    kind=spec.get('shape')
    if kind not in ('circle','ellipse','rectangle'):raise ValueError('unsupported_container_shape')
    outline=_pixels(spec.get('bottom_outline_px',[]),raw.shape,4 if kind=='rectangle' else 6)
    if kind=='rectangle' and len(outline)!=4:raise ValueError('rectangle_requires_four_visible_corners')
    hull=cv2.convexHull(outline.astype(np.float32)).reshape(-1,2)
    if len(hull)!=len(outline):raise ValueError('bottom_outline_must_be_convex_and_complete')
    if cv2.contourArea(hull)<256:raise ValueError('bottom_outline_too_small')
    for anchor in [spec.get('metric_anchor',{})]+spec.get('check_spans',[]):
        for p in anchor.get('pixels',[]):
            if cv2.pointPolygonTest(hull,tuple(map(float,p)),True)<-3:
                raise ValueError('metric_span_endpoint_outside_selected_bottom')
    center=np.mean(hull,axis=0);inner=center+(hull-center)*.75
    mask=np.zeros(raw.shape,'uint8');cv2.fillConvexPoly(mask,np.round(inner).astype('int32'),1)
    up,raw_distance,quality=_plane(raw,k,mask.astype(bool))
    warnings=['geometry_requires_operator_review_and_independent_field_validation']
    if spec.get('up_camera') is not None:
        n=np.asarray(spec['up_camera'],float)
        if n.shape!=(3,) or not np.isfinite(n).all() or np.linalg.norm(n)<1e-6:raise ValueError('invalid_independent_gravity_direction')
        n/=np.linalg.norm(n)
        if n[2]>=0:raise ValueError('gravity_up_must_face_camera')
        disagreement=float(np.degrees(np.arccos(np.clip(n@up,-1,1))))
        quality['depth_plane_gravity_disagreement_deg']=disagreement
        if disagreement>8:raise ValueError('bottom_plane_disagrees_with_independent_gravity')
        up=n;quality['normal_source']='independent_gravity'
    else:
        quality['normal_source']='estimated_bottom_normal_assuming_horizontal_bottom'
        warnings.append('no_independent_gravity_check')
    anchor=spec.get('metric_anchor',{})
    if anchor.get('kind')=='span':
        anchor_pixels,length,factor=_span(anchor,k,up,raw.shape);distance=length/factor
    elif anchor.get('kind')=='perpendicular_distance':
        distance=float(anchor['distance_m'])
        if not np.isfinite(distance) or distance<=0:raise ValueError('positive_perpendicular_distance_required')
        anchor_pixels=None
    else:raise ValueError('independent_metric_anchor_required')
    wall=float(spec.get('wall_height_m',0))
    sensor_family=spec.get('sensor_family','active_stereo')
    if sensor_family not in ('active_stereo','structured_light','tof'):
        raise ValueError('unsupported_sensor_family')
    baseline=float(spec.get('baseline_m',0))
    if not np.isfinite(wall) or not 0<wall<distance:raise ValueError('internal_height_must_be_positive_and_below_camera')
    if not np.isfinite(baseline) or (sensor_family=='tof' and baseline!=0) or (sensor_family!='tof' and baseline<=0):
        raise ValueError('actual_stereo_baseline_required_or_zero_for_tof')
    pts=_intersect(outline,k,up,distance)
    xaxis=np.array([1.,0.,0.]);xaxis-=up*(xaxis@up);xaxis/=np.linalg.norm(xaxis)
    yaxis=np.cross(up,xaxis);closest=-distance*up
    xy=np.column_stack(((pts-closest)@xaxis,(pts-closest)@yaxis)).astype('float32')
    if kind=='circle':
        a=np.column_stack((2*xy,np.ones(len(xy))));b=(xy**2).sum(1)
        params=np.linalg.lstsq(a,b,rcond=None)[0];cx,cy=params[:2]
        radius=np.sqrt(max(params[2]+cx*cx+cy*cy,1e-12));rx=ry=float(radius);angle=0.
    else:
        fit=cv2.minAreaRect(xy) if kind=='rectangle' else cv2.fitEllipse(xy)
        (cx,cy),(w,h),angle=fit;rx,ry=float(w/2),float(h/2)
    if min(rx,ry)<=.001:raise ValueError('degenerate_container_dimensions')
    theta=np.deg2rad(angle);ex=np.cos(theta)*xaxis+np.sin(theta)*yaxis;ey=np.cross(up,ex)
    origin=closest+cx*xaxis+cy*yaxis
    camera_from_world=np.column_stack((ex,ey,up));rotation=camera_from_world.T
    pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=-rotation@origin
    normalized=np.column_stack(((pts-origin)@ex/rx,(pts-origin)@ey/ry))
    radial=np.max(abs(normalized),axis=1) if kind=='rectangle' else np.linalg.norm(normalized,axis=1)
    outline_error=float(np.percentile(abs(radial-1),95));quality['outline_relative_residual_p95']=outline_error
    if outline_error>.12:raise ValueError('outline_inconsistent_with_selected_shape')
    if kind!='rectangle':
        az=np.sort(np.mod(np.arctan2(normalized[:,1],normalized[:,0]),2*np.pi))
        gap=float(np.max(np.diff(np.r_[az,az[0]+2*np.pi])))
        if gap>np.deg2rad(105):raise ValueError('incomplete_bottom_boundary_not_observable')
    checks=[]
    for check in spec.get('check_spans',[]):
        q,length,factor=_span(check,k,up,raw.shape)
        if anchor_pixels is not None and (np.allclose(q,anchor_pixels) or np.allclose(q,anchor_pixels[::-1])):
            raise ValueError('validation_span_must_be_independent_of_fit_anchor')
        error=distance*factor-length
        checks.append(dict(measured_m=length,predicted_m=distance*factor,error_m=error,relative_error=abs(error)/length))
        if abs(error)>max(.003,.02*length):raise ValueError('independent_dimension_check_failed')
    quality['independent_dimension_checks']=checks
    if not checks:warnings.append('metric_anchor_not_independently_checked')
    if distance>10:warnings.append('outside_10m_application_validation_envelope_numeric_value_retained')
    exponent=16. if kind=='rectangle' else 2.
    cal=HeightCalibration(k,pose,0.,(rx,ry),exponent)
    if kind=='rectangle':
        contour=np.array([[-rx,-ry],[rx,-ry],[rx,ry],[-rx,ry]])
    else:
        aa=np.linspace(0,2*np.pi,65)[:-1];contour=np.column_stack((rx*np.cos(aa),ry*np.sin(aa)))
    def camera_contour(z):return origin+contour[:,0,None]*ex+contour[:,1,None]*ey+z*up
    quality.update(raw_vs_metric_distance_delta_m=distance-raw_distance,
        camera_axis_tilt_deg=float(np.degrees(np.arccos(np.clip(-up[2],-1,1)))))
    return dict(schema_version=1,source='empty_rgbd_plus_operator_metric_anchor',shape=kind,
        sensor_family=sensor_family,
        image_size=[raw.shape[1],raw.shape[0]],baseline_m=baseline,rim_z_m=wall,
        calibration=dict(intrinsics=k.tolist(),camera_to_world_cv=pose.tolist(),bottom_z_m=0.,radii_m=[rx,ry],footprint_exponent=exponent),
        bottom_plane_camera=dict(up=up.tolist(),perpendicular_distance_m=distance),
        internal_dimensions_m=dict(width=2*rx,length=2*ry,height=wall),
        projected_bottom_outline_px=_project(camera_contour(0),k),projected_rim_outline_px=_project(camera_contour(wall),k),
        operator_inputs=spec,quality=quality,warnings=warnings,production_qualified=False,
        supported_routes=['ordinary_diffuse_v40'],continuous_motion_supported=False,
        rectangle_footprint='conservative_superellipse_exponent16' if kind=='rectangle' else None)


def height_calibration(bundle):
    c=bundle['calibration']
    return HeightCalibration(np.asarray(c['intrinsics']),np.asarray(c['camera_to_world_cv']),float(c['bottom_z_m']),tuple(c['radii_m']),float(c['footprint_exponent']))


def refraction_geometry(bundle,right_intrinsics,*,left_intrinsics):
    """Conservative adapter for existing near-vertical rectified stereo solver.

Does NOT extend its equations to arbitrary pose, ToF, sidewall refraction or
replace the separate empty-plus-four known liquid-level calibration.
"""
    k=np.asarray(bundle['calibration']['intrinsics']);right=np.asarray(right_intrinsics,float)
    left=np.asarray(left_intrinsics,float)
    if left.shape!=(3,3) or not np.allclose(left,k,rtol=0,atol=1e-5):
        raise ValueError('geometry_must_be_built_in_actual_rectified_left_camera_coordinates')
    if right.shape!=(3,3) or not np.isfinite(right).all():raise ValueError('actual_rectified_right_intrinsics_required')
    if bundle['quality']['camera_axis_tilt_deg']>.5:raise ValueError('existing_refraction_solver_requires_near_vertical_camera')
    if not np.isclose(k[0,0],k[1,1],rtol=.001) or not np.allclose(right[:2,:2],k[:2,:2],rtol=.001,atol=.001):
        raise ValueError('existing_refraction_solver_requires_equal_rectified_focal_lengths')
    return dict(fx=float(k[0,0]),cx=float(k[0,2]),cy=float(k[1,2]),reference_cx=float(right[0,2]),reference_cy=float(right[1,2]),
        baseline_m=bundle['baseline_m'],bottom_m=bundle['bottom_plane_camera']['perpendicular_distance_m'],
        container_max_height_m=bundle['rim_z_m'],width=bundle['image_size'][0],height=bundle['image_size'][1],
        geometry_source='empty_geometry_v41',research_only=True,requires_one_empty_plus_four_known_liquid_levels=True)


def make_refraction_meter(bundle,right_intrinsics,masks,prior_images,known_heights,condition,method='baseline_v32',sr_backend=None,*,left_intrinsics):
    """Explicit experimental integration; raw rectified L/R priors still required.

    ROI masks must be selected independently in each rectified camera image,
    not blindly copied from the registered RGB-D geometry image.
    """
    cal=refraction_geometry(bundle,right_intrinsics,left_intrinsics=left_intrinsics)
    heights=np.asarray(known_heights,float)
    if heights.shape!=(5,) or not np.isfinite(heights).all() or heights[0]!=0 or np.any(heights[1:]<=0):
        raise ValueError('requires_one_empty_followed_by_four_positive_known_liquid_depths')
    if len(np.unique(heights))!=5:raise ValueError('known_calibration_depths_must_be_distinct')
    for ch in ('left','right'):
        mask=np.asarray(masks[ch])
        if mask.shape!=(cal['height'],cal['width']) or not np.any(mask):
            raise ValueError('independent_rectified_camera_roi_mask_required:'+ch)
    from .range_runtime_v33 import FivePriorImageMeter
    return FivePriorImageMeter(cal,masks,prior_images,heights,condition,method,sr_backend)
