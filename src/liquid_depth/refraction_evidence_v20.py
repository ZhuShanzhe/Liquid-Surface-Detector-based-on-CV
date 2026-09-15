"""Continuous subpixel chromatic reference matching.
The score threshold is a development-selected heuristic, not a probability.
"""
import cv2,numpy as np
from scipy.ndimage import map_coordinates,spline_filter
from .refraction_core_v20 import calibrate_empty,bottom_mapping
from .refraction_evidence_v19 import qualify_profile as qualify_v19

def linear_rgb(x):
    x=np.asarray(x,np.float64)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def add_chromatic_evidence(result,rgb,empty,empty_raw,camera):
    r=dict(result)
    if not r.get('profile'):return r
    roi=calibrate_empty(np.asarray(empty_raw,float)*float(camera.get('depth_scale_to_m',1.)))['roi']
    def chromatic(x):
        y=linear_rgb(x)
        return y/(y.sum(2,keepdims=True)+.001)
    ref=chromatic(empty);obs=chromatic(rgb);mean=ref[roi].mean(0)
    _,_,axes=np.linalg.svd(ref[roi]-mean,full_matrices=False)
    ref=cv2.GaussianBlur((ref-mean)@axes[0],(0,0),1.)
    obs=cv2.GaussianBlur((obs-mean)@axes[0],(0,0),1.)
    texture=float(np.std(ref[roi]));coef=spline_filter(ref,order=3)
    yy,xx=np.indices(roi.shape)
    ids=np.flatnonzero(roi&(np.max(rgb,axis=2)<.97))
    if len(ids):ids=ids[np.linspace(0,len(ids)-1,min(12000,len(ids))).astype(int)]
    x=((xx+.5-camera['cx'])/camera['fx']).ravel()[ids]
    y=((yy+.5-camera['cy'])/camera['fy']).ravel()[ids]
    target=obs.ravel()[ids];target=target-target.mean() if len(target) else target
    profile=[]
    for row in r['profile']:
        p=dict(row);photo=0.
        if texture>.025 and len(ids)>100:
            u,v=bottom_mapping(x,y,r['top_depth_m'],p['thickness_m'],r['total_depth_m'],p['n_upper'],p['n_lower'],r['bottom_depth_m'])
            mx=u*camera['fx']+camera['cx']-.5;my=v*camera['fy']+camera['cy']-.5
            values=map_coordinates(coef,np.stack([my,mx]),order=3,prefilter=False,mode='nearest')
            values=values-values.mean()
            photo=float(1-np.dot(values,target)/(np.linalg.norm(values)*np.linalg.norm(target)+1e-9))
        p['chromatic_error']=photo;profile.append(p)
    r.update(profile=profile,chromatic_texture=texture,reference_sampling='continuous_cubic_spline',
        evidence_rgb_pixels=len(ids),linearization='sRGB_piecewise')
    return r

def qualify_profile(result):
    return qualify_v19(result,weight=1000000.,contrast_floor=.02)
