"""Opt-in ordinary flat opaque-liquid estimator. No ground-truth inputs."""
from dataclasses import replace
import cv2
import numpy as np
from liquid_depth.surface_refinement import RefinedSurfaceEstimator, StereoNoiseModel


def interior_ray_roi(cal, rim_z_m, shape, fraction=.85):
    """Rays inside both end sections stay inside a vertical convex vessel.

    Selection uses installed geometry, not measured noisy XYZ or liquid height.
    Tapered vessels and a moving/uncalibrated camera are outside this contract.
    """
    if not .1 < fraction <= 1 or rim_z_m <= cal.bottom_z_m:
        raise ValueError('Invalid vessel calibration')
    y,x=np.indices(shape);k=cal.intrinsics;t=cal.camera_to_world_cv
    rays=np.stack(((x-k[0,2])/k[0,0],(y-k[1,2])/k[1,1],np.ones(shape)),-1)@t[:3,:3].T
    roi=rays[...,2]<-1e-6
    for plane in (cal.bottom_z_m,rim_z_m):
        scale=(plane-t[2,3])/np.where(abs(rays[...,2])>1e-9,rays[...,2],np.nan)
        xy=t[:2,3]+scale[...,None]*rays[...,:2]
        roi &= (scale>0)&((abs(xy/(np.asarray(cal.radii_m)*fraction))**cal.footprint_exponent).sum(-1)<=1)
    return roi


def estimate_ordinary(rgb, raw, cal, rim_z_m, sensor, fraction=.85):
    adaptive=fraction==0
    roi=interior_ray_roi(cal,rim_z_m,raw.shape,.65 if adaptive else fraction)
    fallback=False
    if adaptive and roi.sum()<256:
        # A deep narrow vessel can have no ray traversing both bottom and opening.
        # Use the installed opening only; flag its weaker wall exclusion explicitly.
        virtual_bottom=replace(cal,bottom_z_m=rim_z_m-1e-5)
        roi=interior_ray_roi(virtual_bottom,rim_z_m,raw.shape,.65)
        fallback=True
    noise=None if sensor=='tof' else StereoNoiseModel.simulation_proxy(sensor)
    result=RefinedSurfaceEstimator(mode='sensor',stereo_noise=noise).estimate(
        rgb,raw,{'mask':roi},cal.intrinsics,cal.camera_to_world_cv,
        cal.bottom_z_m,cal.area_grid(),cal.radii_m)
    s=result['statistics']
    if s:s={k:s[k.replace('_m','_depth_m')] for k in ('mean_m','min_m','max_m','p05_m','p95_m')}
    flags=result['quality_flags']+(['opening_only_geometry_fallback'] if fallback else [])
    return dict(statistics=s,reasons=flags,qualified=False,
        selected_pixels=int(roi.sum()),route='ordinary_calibrated_vertical_container_flat_surface')
