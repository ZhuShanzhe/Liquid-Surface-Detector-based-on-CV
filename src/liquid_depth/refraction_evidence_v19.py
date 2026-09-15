"""Reference chromaticity and a homogeneous-medium competing explanation."""
import cv2,numpy as np
from .refraction_inverse_v19 import calibrate_empty,bottom_mapping

def chromaticity_pair(rgb,empty,roi):
    def chromatic(x):
        linear=np.maximum(np.asarray(x,np.float32),0)**2.2
        return linear/(np.sum(linear,axis=2,keepdims=True)+.01)
    ref=chromatic(empty);obs=chromatic(rgb)
    samples=ref[roi];mean=samples.mean(0)
    _,_,axes=np.linalg.svd(samples[::max(1,len(samples)//4096)]-mean,full_matrices=False)
    axis=axes[0]
    a=(ref-mean)@axis;b=(obs-mean)@axis
    a=cv2.GaussianBlur(a,(0,0),.6);b=cv2.GaussianBlur(b,(0,0),.6)
    scale=float(np.std(a[roi]))
    return a/(scale+1e-6),b/(scale+1e-6),scale

def add_chromatic_evidence(result,rgb,empty,empty_raw,camera):
    r=dict(result)
    if not r.get('profile'):return r
    cal=calibrate_empty(np.asarray(empty_raw,float)*float(camera.get('depth_scale_to_m',1.)));roi=cal['roi']
    ref,obs,texture=chromaticity_pair(rgb,empty,roi)
    yy,xx=np.indices(roi.shape);qx=(xx+.5-camera['cx'])/camera['fx'];qy=(yy+.5-camera['cy'])/camera['fy']
    ids=np.flatnonzero(roi&(np.max(rgb,axis=2)<.97));ids=ids[np.linspace(0,len(ids)-1,min(4096,len(ids))).astype(int)]
    x=qx.ravel()[ids];y=qy.ravel()[ids];target=obs.ravel()[ids]
    target=target-target.mean()
    profile=[]
    for row in r['profile']:
        p=dict(row);photo=0.
        if texture>.025 and len(ids)>100:
            u,v=bottom_mapping(x,y,r['top_depth_m'],p['thickness_m'],r['total_depth_m'],p['n_upper'],p['n_lower'],r['bottom_depth_m'])
            mx=(u*camera['fx']+camera['cx']-.5).astype(np.float32)[None];my=(v*camera['fy']+camera['cy']-.5).astype(np.float32)[None]
            values=cv2.remap(ref,mx,my,cv2.INTER_LINEAR).ravel();values=values-values.mean()
            photo=float(1-np.sum(values*target)/(np.sqrt(np.sum(values**2)*np.sum(target**2))+1e-9))
        p['chromatic_error']=photo;profile.append(p)
    r['profile']=profile;r['chromatic_texture']=texture
    return r

def qualify_profile(result,weight=30000.,contrast_floor=.02,channel='chromatic_error'):
    r=dict(result)
    if not r.get('profile'):return r
    p=r['profile'];cost=[v['raw_loss']+weight*v.get(channel,v['photo_error']) for v in p]
    plausible=[v for v,c in zip(p,cost) if c<=min(cost)+6.]
    lo=min(v['thickness_m'] for v in plausible);hi=max(v['thickness_m'] for v in plausible)
    point=min(plausible,key=lambda v:abs(v['thickness_m']-(lo+hi)/2))
    contrast=min(abs(v['n_upper']-v['n_lower']) for v in plausible)
    geometry=(hi-lo)<=.010
    minimum=min(p,key=lambda v:v['raw_loss']+weight*v.get(channel,v['photo_error']))
    photo=minimum.get(channel,minimum['photo_error'])
    accepted=geometry and contrast>contrast_floor and photo<.60
    r.update(geometric_accepted=geometry,accepted=accepted,liquid_identity_verified=False,
             status='refraction_evidence_supported' if accepted else ('homogeneous_medium_not_excluded' if geometry else 'ambiguous_refractive_index'),
             candidate_thickness_m=point['thickness_m'],thickness_m=point['thickness_m'] if accepted else None,
             internal_depth_m=r['top_depth_m']+point['thickness_m'] if accepted else None,
             n_upper_estimate=point['n_upper'],n_lower_estimate=point['n_lower'],
             profile_range_m=[lo,hi],profile_width_m=hi-lo,contrast_lower_bound=contrast,evidence_weight=weight,evidence_channel=channel)
    return r
