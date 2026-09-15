"""Spatially independent burst reduction for fixed overhead refraction.
No material labels or known refractive indices are accepted.
"""
from dataclasses import replace
import warnings
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from .refraction_core_v20 import estimate_layers as core,RefractionConfig,calibrate_empty,ray_factors
from .refraction_evidence_v20 import add_chromatic_evidence,qualify_profile
from .refraction_calibration_v19 import apply_reference_quality

def reduce_burst(raw,empty_raw,camera):
    scale=float(camera.get('depth_scale_to_m',1.))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid SDK depth scale')
    z=np.asarray(raw,float)*scale
    if z.ndim==2:z=z[None]
    if z.ndim!=3 or z.shape[1:]!=np.shape(empty_raw):raise ValueError('Invalid registered burst shape')
    if len(z)<16:return None,dict(status='insufficient_stable_frames')
    z=np.where(np.isfinite(z)&(z>0),z,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        count=np.sum(np.isfinite(z),0)
        mean=np.nanmean(z,0);sd=np.nanstd(z,0)
        a=np.nanmean(z[:len(z)//2],0);b=np.nanmean(z[len(z)//2:],0)
    cal=calibrate_empty(np.asarray(empty_raw,float)*float(camera.get('depth_scale_to_m',1.)))
    stable=(count>=max(12,int(len(z)*.5)))&(sd<max(.003,4*cal['noise_m']))
    probe=stable&cal['roi']&np.isfinite(a)&np.isfinite(b)
    if probe.sum()<500:return None,dict(status='insufficient_stable_depth')
    shift=float(np.median(b[probe]-a[probe]))
    if abs(shift)>.001:return None,dict(status='moving_surface',half_sequence_shift_m=shift)
    # Estimate the residual spatial noise from the observed planar top return.
    values=mean[probe];step=.00025
    bins=np.arange(values.min()-.002,values.max()+.002,step)
    hist,_=np.histogram(values,bins);sm=gaussian_filter1d(hist.astype(float),1.)
    peaks,_=find_peaks(sm,prominence=max(2,len(values)*.0004),distance=4)
    centers=(bins[:-1]+bins[1:])/2
    choices=[k for k in peaks if centers[k]<cal['bottom_depth_m']-.012 and np.sum(abs(values-centers[k])<.002)>max(80,.015*len(values))]
    if not choices:return None,dict(status='no_supported_free_surface')
    top=centers[min(choices,key=lambda k:centers[k])]
    selected=values[abs(values-top)<.002]
    sigma=max(.0003,float(1.4826*np.median(abs(selected-np.median(selected)))))
    reduced=np.where(stable,mean,0)
    return reduced,dict(status='burst_reduced',frames=len(z),stable_pixels=int(probe.sum()),
        spatial_noise_m=sigma,half_sequence_shift_m=shift,independent_unit='spatial_pixel')

def estimate_layers(rgb,raw,empty_rgb,empty_raw,camera,config=RefractionConfig(),use_rgb=True,geometry_only=False):
    if camera.get('sensor_model')!='nondispersive_coaxial_tof_proxy':
        return dict(status='unsupported_sensor_model',accepted=False,thickness_m=None)
    if camera.get('depth_alignment','color')!='color':raise ValueError('Register depth to color before inference')
    if min(float(camera['fx']),float(camera['fy']))<=0:raise ValueError('Invalid intrinsics')
    for image in [rgb,empty_rgb]:
        if np.shape(image)!=(*np.shape(empty_raw),3) or not np.isfinite(image).all() or np.min(image)<0 or np.max(image)>1:raise ValueError('Expected finite RGB in [0,1] with registered shape')
    reduced,diag=reduce_burst(raw,empty_raw,camera)
    if reduced is None:return dict(diag,accepted=False,thickness_m=None,liquid_identity_verified=False)
    # The reduction is in meters; empty depth remains in original SDK units.
    scale=float(camera.get('depth_scale_to_m',1.))
    r=core(rgb,reduced/scale,empty_rgb,empty_raw,camera,
        config=replace(config,measured_noise_m=diag['spatial_noise_m']),use_rgb=use_rgb)
    r['temporal']=diag;r['frames']=diag['frames'];r['liquid_identity_verified']=False
    if not geometry_only:
        r=qualify_profile(add_chromatic_evidence(r,rgb,empty_rgb,empty_raw,camera))
        r=apply_reference_quality(r,empty_rgb,empty_raw,camera)
    return r
