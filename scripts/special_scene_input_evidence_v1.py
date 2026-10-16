"""Separate online raw-support check and offline synthetic return-origin audit."""
import numpy as np

POLICY = dict(version='input_evidence_v1', minimum_interior_returns=64,
              minimum_occupied_4x4_tiles=6, minimum_footprint_span_fraction=.25,
              interior_power_boundary=.85,
              note='engineering input-quality rule, not an industrial standard or guarantee of correct surface interpretation')


def assess(depth_m,bundle):
    raw=np.asarray(depth_m)
    yy,xx=np.nonzero(np.isfinite(raw)&(raw>0))
    cal=bundle['calibration']
    k=np.asarray(cal['intrinsics'],float);pose=np.asarray(cal['camera_to_world_cv'],float)
    rays=np.column_stack((xx,yy,np.ones(len(xx))))@np.linalg.inv(k).T
    xyz=(rays*raw[yy,xx,None])@pose[:3,:3].T+pose[:3,3]
    rx,ry=cal['radii_m'];power=float(cal['footprint_exponent'])
    # An uncertainty buffer around the physical volume does not set a liquid-depth range.
    buffer=max(.05,.04*float(bundle['bottom_plane_camera']['perpendicular_distance_m']))
    inside=(abs(xyz[:,0]/rx)**power+abs(xyz[:,1]/ry)**power<=POLICY['interior_power_boundary'])
    inside&=(xyz[:,2]>=-buffer)&(xyz[:,2]<=bundle['rim_z_m']+buffer)
    points=xyz[inside]
    reasons=[]
    if len(points)<POLICY['minimum_interior_returns']:
        reasons.append('fewer_than_64_interior_returns')
    if len(points):
        normalized=(points[:,:2]/[rx,ry]+1)/2
        tiles=np.clip((normalized*4).astype(int),0,3)
        occupied=len(np.unique(tiles[:,0]+4*tiles[:,1]))
        span=np.ptp(normalized,axis=0)
    else:
        occupied=0;span=np.zeros(2)
    if occupied<POLICY['minimum_occupied_4x4_tiles']:
        reasons.append('fewer_than_6_spatial_tiles')
    if np.any(span<POLICY['minimum_footprint_span_fraction']):
        reasons.append('insufficient_two_axis_spatial_span')
    return dict(eligible=not reasons,reasons=reasons,interior_returns=len(points),
                occupied_4x4_tiles=occupied,footprint_span_fraction=span.tolist(),
                uses_true_liquid_depth=False,uses_predicted_liquid_depth=False,
                uses_offline_semantic_or_origin_labels=False,policy=POLICY['version'])


def assess_capture_origin(depth_m,semantic,error_type,*,contains_liquid):
    """Offline capture audit, NOT a deployable online gate or an inference input.

    Uses geometric return-source classes, never numerical depth accuracy. A noisy
    direct surface return remains evidence even when its distance is very wrong.
    """
    region=np.asarray(semantic)==(2 if contains_liquid else 1)
    direct=region&(np.asarray(depth_m)>0)
    if contains_liquid:
        direct&=np.asarray(error_type)==0
    yy,xx=np.nonzero(region)
    vy,vx=np.nonzero(direct)
    reasons=[]
    if len(vx)<64:reasons.append('capture_fewer_than_64_direct_surface_returns')
    if len(vx) and len(xx):
        w=max(int(xx.max()-xx.min()+1),1);h=max(int(yy.max()-yy.min()+1),1)
        normalized=np.column_stack(((vx-xx.min())/w,(vy-yy.min())/h))
        tiles=np.clip((normalized*4).astype(int),0,3)
        occupied=len(np.unique(tiles[:,0]+4*tiles[:,1]))
        span=np.ptp(normalized,axis=0)
    else:occupied=0;span=np.zeros(2)
    if occupied<6:reasons.append('capture_fewer_than_6_surface_tiles')
    if np.any(span<.25):reasons.append('capture_insufficient_surface_span')
    return dict(eligible=not reasons,reasons=reasons,direct_surface_returns=len(vx),
        occupied_4x4_tiles=occupied,span_fraction=span.tolist(),
        uses_true_or_predicted_liquid_depth_error=False,uses_offline_return_source_labels=True,
        inference_may_not_receive_these_labels=True,
        note='capture-information sufficiency audit only, not real-world online observability certification')
