"""2/4-frame ordinary flat-surface likelihood, fixed calibrated camera only."""
from dataclasses import replace
import numpy as np
from .ordinary_surface_v35 import interior_ray_roi
from .ordinary_surface_v36 import estimate_v36,discrete_clearance
from .surface_refinement import support,StereoNoiseModel,weighted_median


def observations(raw,cal,rim,sensor):
    roi=interior_ray_roi(cal,rim,raw.shape,.65)
    if roi.sum()<256:roi=interior_ray_roi(replace(cal,bottom_z_m=rim-1e-5),rim,raw.shape,.65)
    data,reason=support(raw,roi,cal.intrinsics,cal.camera_to_world_cv,cal.radii_m,max_points=8192)
    if data is None:return None,reason
    if sensor!='tof' and np.median(data[2])<3:
        roi=interior_ray_roi(cal,rim,raw.shape,.9)
        if roi.sum()<256:roi=interior_ray_roi(replace(cal,bottom_z_m=rim-1e-5),rim,raw.shape,.9)
        more,_=support(raw,roi,cal.intrinsics,cal.camera_to_world_cv,cal.radii_m,max_points=8192)
        if more is not None:data=more
    points,pixels,z,_,_=data;k=cal.intrinsics;t=cal.camera_to_world_cv
    rays=np.column_stack(((pixels[:,0]-k[0,2])/k[0,0],(pixels[:,1]-k[1,2])/k[1,1],np.ones(len(z))))
    c=-(rays@t[:3,:3].T)[:,2]
    return (z,c,np.full(len(z),1/len(z))),None


def estimate_window(raw_frames,cal,rim_z_m,sensor,*,timestamps=None,motion_detected=False,
                    fixed_pose=True,noise_model=None,disparity_step=1/16,depth_step=.001):
    """All windows end at the latest frame. Missing history is not fabricated.

    Returns unqualified diagnostics. No inference from known liquid heights;
    no zeroing, capacity clipping, or confidence-based output rejection.
    Motion flags must come from the caller, not an evaluator label.
    """
    if not 1<=len(raw_frames)<=4:raise ValueError('Use 1 to 4 chronological frames')
    if any(np.asarray(x).shape!=np.asarray(raw_frames[-1]).shape for x in raw_frames):raise ValueError('Aligned shape required')
    if timestamps is not None:
        t=np.asarray(timestamps,float)
        if len(t)!=len(raw_frames) or not np.isfinite(t).all() or (np.diff(t)<=0).any():raise ValueError('Increasing timestamps required')
        stale=t[-1]-t[0]>.5
    else:stale=False
    if len(raw_frames)==1 or motion_detected or not fixed_pose or stale or sensor=='tof':
        p=estimate_v36(None,raw_frames[-1],cal,rim_z_m,sensor,uniform=True,noise_model=noise_model,
            disparity_step=disparity_step,depth_step=depth_step)
        p['frames_used']=1
        if len(raw_frames)>1:p['reasons']+=['current_frame_only_motion_pose_age_or_unchanged_tof']
        return p
    data=[];reasons=[]
    for raw in raw_frames:
        d,reason=observations(raw,cal,rim_z_m,sensor)
        if d is not None:data.append(d)
        else:reasons.append(reason)
    # A past-only estimate must not masquerade as an observable current output.
    if d is None:
        return dict(statistics=None,qualified=False,reasons=['no_current_support'],frames_used=0)
    if not data:return dict(statistics=None,qualified=False,reasons=reasons,frames_used=0)
    z=np.concatenate([d[0] for d in data]);c=np.concatenate([d[1] for d in data]);w=np.concatenate([d[2]/len(data) for d in data])
    pose=cal.camera_to_world_cv;model=noise_model or StereoNoiseModel.simulation_proxy(sensor)
    clearance=discrete_clearance(z,c,w,cal.intrinsics[0,0],model,bottom_clearance=pose[2,3]-cal.bottom_z_m,
        echo=True,disparity_step=disparity_step,depth_step=depth_step)
    if clearance is None:return dict(statistics=None,qualified=False,reasons=['joint_likelihood_unobservable'],frames_used=len(data))
    h=float(pose[2,3]-cal.bottom_z_m-clearance)
    return dict(statistics={k:h for k in ('mean_m','min_m','max_m','p05_m','p95_m')},qualified=False,
        reasons=reasons+['fixed_pose_static_surface_required','correlation_uncertainty_not_calibrated'],
        frames_used=len(data),selected_points=len(z))


def temporal_block_correlation(raw_frames,cal,rim_z_m,block=8):
    """Observed residual-field correlation, no noise mode or surface truth."""
    shape=raw_frames[-1].shape;roi=interior_ray_roi(cal,rim_z_m,shape,.65)
    if roi.sum()<256:roi=interior_ray_roi(replace(cal,bottom_z_m=rim_z_m-1e-5),rim_z_m,shape,.65)
    yy,xx=np.indices(shape);k=cal.intrinsics;t=cal.camera_to_world_cv
    rays=np.stack(((xx-k[0,2])/k[0,0],(yy-k[1,2])/k[1,1],np.ones(shape)),-1)@t[:3,:3].T
    c=-rays[...,2];tiles=(yy//block)*((shape[1]+block-1)//block)+xx//block;length=int(tiles.max()+1)
    means=[]
    for raw in raw_frames:
        valid=roi&np.isfinite(raw)&(raw>0)&(c>.1)
        if valid.sum()<24:return None
        clearance=np.median(raw[valid]*c[valid]);residual=1/raw[valid]-c[valid]/clearance
        center=np.median(residual);scale=max(1e-8,1.4826*np.median(abs(residual-center)))
        # Winsorization applies only to correlation diagnosis, never to liquid heights.
        residual=np.clip(residual-center,-3*scale,3*scale)
        ids=tiles[valid];counts=np.bincount(ids,minlength=length)
        sums=np.bincount(ids,weights=residual,minlength=length)
        means.append(np.where(counts>=8,sums/np.maximum(counts,1),np.nan))
    values=[]
    for a,b in zip(means[:-1],means[1:]):
        use=np.isfinite(a)&np.isfinite(b)
        if use.sum()>=12 and np.std(a[use])>1e-10 and np.std(b[use])>1e-10:
            values.append(float(np.corrcoef(a[use],b[use])[0,1]))
    return float(np.median(values)) if values else None


def estimate_guarded4(raw_frames,cal,rim_z_m,sensor,**kwargs):
    """Conservative optional fusion; fallbacks preserve current numeric output."""
    if not 1<=len(raw_frames)<=4:raise ValueError('Use 1 to 4 chronological frames')
    sensor_kwargs={k:kwargs[k] for k in ('noise_model','disparity_step','depth_step') if k in kwargs}
    current=estimate_window(raw_frames[-1:],cal,rim_z_m,sensor,**sensor_kwargs)
    if current['statistics'] is None:return current
    clearance=cal.camera_to_world_cv[2,3]-cal.bottom_z_m-current['statistics']['mean_m']
    if sensor=='tof' or clearance<3 or len(raw_frames)<4:
        current['reasons']+=['current_only_near_or_unchanged_tof_or_warmup'];return current
    rho=temporal_block_correlation(raw_frames,cal,rim_z_m)
    if rho is None or rho>.5:
        current.update(temporal_block_correlation=rho)
        current['reasons']+=['current_only_persistent_or_unmeasurable_correlation'];return current
    result=estimate_window(raw_frames,cal,rim_z_m,sensor,**kwargs)
    result['temporal_block_correlation']=rho
    if result['statistics'] is None:
        current['reasons']+=['current_only_joint_unobservable'];return current
    return result
