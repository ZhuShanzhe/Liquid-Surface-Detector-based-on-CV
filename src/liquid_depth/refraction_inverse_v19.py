"""Fixed overhead two-fluid inverse optics with an acquired empty reference.

This prototype supports only a declared nondispersive coaxial ToF proxy.
No layer count, material index, thickness, liquid mask or camera height is input.
"""
from dataclasses import dataclass,asdict
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

@dataclass(frozen=True)
class RefractionConfig:
    min_index:float=1.05
    max_index:float=1.80
    index_steps:int=76
    min_internal_points:int=24
    min_internal_fraction:float=.002
    min_tiles:int=4
    max_profile_width_m:float=.010
    profile_delta:float=6.
    rgb_weight:float=100.
    max_photo_error:float=.60

def ray_factors(n,q2):
    c0=1/np.sqrt(1+q2)
    s2=q2/(1+q2)
    return n*c0/np.sqrt(1-s2/(n*n))

def bottom_mapping(qx,qy,d,t,h,n1,n2,bottom_depth):
    q2=qx*qx+qy*qy
    a1=np.sqrt(n1*n1+(n1*n1-1)*q2)
    a2=np.sqrt(n2*n2+(n2*n2-1)*q2)
    scale=(d+t/a1+(h-t)/a2)/bottom_depth
    return qx*scale,qy*scale

def calibrate_empty(empty_raw):
    z=np.asarray(empty_raw,float);h,w=z.shape
    seed=z[h*2//5:h*3//5,w*2//5:w*3//5]
    seed=seed[np.isfinite(seed)&(seed>0)]
    if len(seed)<80:raise ValueError('Insufficient empty bottom measurements')
    center=float(np.median(seed))
    selected=seed[np.abs(seed-center)<.006]
    bottom=float(np.mean(selected))
    sigma=max(.0006,float(1.4826*np.median(abs(selected-np.median(selected)))))
    mask=(np.isfinite(z)&(z>0)&(abs(z-bottom)<.004)).astype(np.uint8)
    # Recover the connected central opaque plane from the measured empty depth.
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    num,labels,stats,centroids=cv2.connectedComponentsWithStats(mask)
    label=labels[h//2,w//2]
    if label==0:
        candidates=[k for k in range(1,num) if stats[k,cv2.CC_STAT_AREA]>80]
        if not candidates:raise ValueError('No connected bottom plane')
        label=min(candidates,key=lambda k:np.linalg.norm(centroids[k]-[w/2,h/2]))
    roi=cv2.erode((labels==label).astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool)
    values=z[roi&(z>0)&(abs(z-bottom)<.004)]
    if len(values)<200:raise ValueError('Insufficient calibrated interior support')
    return dict(bottom_depth_m=float(values.mean()),noise_m=sigma,roi=roi)

def normalized_image(rgb):
    x=cv2.cvtColor(np.asarray(rgb,dtype=np.float32),cv2.COLOR_RGB2GRAY)
    low=cv2.GaussianBlur(x,(0,0),3.)
    residual=x-low
    scale=np.sqrt(cv2.GaussianBlur(residual*residual,(0,0),5.)+.001)
    return residual/scale

def estimate_layers(rgb,raw,empty_rgb,empty_raw,camera,config=RefractionConfig(),use_rgb=True):
    if camera.get('sensor_model')!='nondispersive_coaxial_tof_proxy':
        return dict(status='unsupported_sensor_model',accepted=False,thickness_m=None)
    scale=float(camera.get('depth_scale_to_m',1.))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid SDK depth scale')
    if camera.get('depth_alignment','color')!='color':raise ValueError('Register depth to color before inference')
    frames=np.asarray(raw,float)*scale
    if frames.ndim==2:frames=frames[None]
    if frames.ndim!=3 or frames.shape[1:]!=np.shape(empty_raw) or rgb.shape!=empty_rgb.shape or rgb.shape[:2]!=frames.shape[1:]:
        raise ValueError('Reference and observation must have identical registered dimensions')
    if min(float(camera['fx']),float(camera['fy']))<=0:raise ValueError('Invalid focal length')
    cal=calibrate_empty(np.asarray(empty_raw,float)*scale);single_roi=cal['roi'];B=cal['bottom_depth_m'];sigma=cal['noise_m']
    count,h,w=frames.shape;z=frames.reshape(count*h,w);roi=np.tile(single_roi,(count,1))
    vv,uu=np.indices((h,w));qx=np.tile((uu+.5-camera['cx'])/camera['fx'],(count,1));qy=np.tile((vv+.5-camera['cy'])/camera['fy'],(count,1));q2=qx*qx+qy*qy
    good=roi&np.isfinite(z)&(z>0)&(z>B-.35)&(z<B+.25)
    values=z[good]
    base=dict(accepted=False,thickness_m=None,bottom_depth_m=B,calibration_noise_m=sigma,calibration_pixels=int(single_roi.sum()),frames=count)
    if len(values)<500:return dict(base,status='insufficient_depth')
    step=.0005;edges=np.arange(values.min()-.002,values.max()+.002,step)
    hist,_=np.histogram(values,edges);smooth=gaussian_filter1d(hist.astype(float),max(1.,sigma/step))
    peaks,_=find_peaks(smooth,prominence=max(3,len(values)*.0004),distance=max(4,int(4*sigma/step)))
    centers=(edges[:-1]+edges[1:])/2
    top_choices=[k for k in peaks if centers[k]<B-.012 and np.sum(abs(values-centers[k])<3*sigma)>max(80,.015*len(values))]
    if not top_choices:return dict(base,status='no_supported_free_surface')
    first=min(top_choices,key=lambda k:centers[k]);near=good&(abs(z-centers[first])<3*sigma)
    d=float(z[near].mean());H=B-d
    base.update(top_depth_m=d,total_depth_m=H,top_points=int(near.sum()))
    frame_tops=[float(frames[k][near.reshape(count,h,w)[k]].mean()) for k in range(count) if near.reshape(count,h,w)[k].sum()>30]
    if len(frame_tops)>1 and np.ptp(frame_tops)>.004:return dict(base,status='moving_free_surface')
    if not .010<H<.35:return dict(base,status='invalid_total_depth')
    far=good&(z>d+.75*H)
    if far.sum()<200:return dict(base,status='no_bottom_return')
    # Dominant far mode may include outliers; start with its densest 8 mm neighborhood.
    farvals=z[far];hist,bins=np.histogram(farvals,np.arange(farvals.min()-.001,farvals.max()+.001,.001))
    mode=float((bins[np.argmax(hist)]+bins[np.argmax(hist)+1])/2)
    bottom_mask=far&(abs(z-mode)<max(.012,6*sigma))
    def effective_cost(n):
        r=(z[bottom_mask]-d-H*ray_factors(n,q2[bottom_mask]))/sigma
        return float(np.mean(np.minimum(r*r,25)))
    eff=minimize_scalar(effective_cost,bounds=(config.min_index,config.max_index),method='bounded').x
    bottom_mask=good&(abs(z-d-H*ray_factors(eff,q2))<4*sigma)
    if bottom_mask.sum()<200:return dict(base,status='unsupported_bottom_model')
    # Remove angular change before looking for an interior echo.
    optical=(z-d)/ray_factors(eff,q2)*eff
    lower=max(6*sigma,.007);upper=H*eff-max(6*sigma,.007)
    inside=good&(optical>lower)&(optical<upper)
    if inside.sum()<config.min_internal_points:return dict(base,status='internal_not_observed',effective_index=float(eff))
    e=np.arange(lower,upper+step,step);hist,_=np.histogram(optical[inside],e)
    sm=gaussian_filter1d(hist.astype(float),max(1.,sigma/step))
    ps,_=find_peaks(sm,prominence=max(2.,len(values)*.00015),distance=max(4,int(4*sigma/step)))
    centers2=(e[:-1]+e[1:])/2
    candidates=[]
    for k in ps:
        mask=inside&(abs(optical-centers2[k])<3*sigma)
        count=int(mask.sum())
        if count<max(config.min_internal_points,config.min_internal_fraction*len(values)):continue
        yy,xx=np.nonzero(mask);yy=yy%h;yb,xb=np.nonzero(single_roi)
        tiles=set(zip(np.minimum(2,3*(xx-xb.min())//max(1,xb.max()-xb.min()+1)),np.minimum(2,3*(yy-yb.min())//max(1,yb.max()-yb.min()+1))))
        if len(tiles)<config.min_tiles:continue
        candidates.append((count,mask,len(tiles)))
    if not candidates:return dict(base,status='internal_not_spatially_supported',effective_index=float(eff))
    _,inner_mask,tiles=max(candidates,key=lambda x:x[0])
    qi=q2[inner_mask];zi=z[inner_mask]
    qb=q2[bottom_mask];zb=z[bottom_mask]
    # Retain deterministic spatial samples; component identities come only from measured ranges.
    pick=np.linspace(0,len(zb)-1,min(6000,len(zb))).astype(int);qb=qb[pick];zb=zb[pick]
    ns=np.linspace(config.min_index,config.max_index,config.index_steps)
    profile=[];obs=normalized_image(rgb);ref=normalized_image(empty_rgb)
    photomask=single_roi&(np.max(rgb,axis=2)<.97)
    ids=np.flatnonzero(photomask);ids=ids[np.linspace(0,len(ids)-1,min(4096,len(ids))).astype(int)]
    x=qx.ravel()[ids];y=qy.ravel()[ids];target=obs.ravel()[ids]
    texture=float(np.std(ref[single_roi]))
    for n1 in ns:
        fi=ray_factors(n1,qi);t=float(np.sum(fi*(zi-d))/np.sum(fi*fi))
        if not .001<t<H-.003:continue
        def bottom_cost(n2):
            resid=zb-d-t*ray_factors(n1,qb)-(H-t)*ray_factors(n2,qb)
            return float(np.mean(np.minimum((resid/sigma)**2,16)))
        n2=float(minimize_scalar(bottom_cost,bounds=(config.min_index,config.max_index),method='bounded').x)
        ri=(zi-d-t*ray_factors(n1,qi))/sigma
        rb=(zb-d-t*ray_factors(n1,qb)-(H-t)*ray_factors(n2,qb))/sigma
        # Both groups contribute according to measured sample count.
        raw_loss=float(np.minimum(ri*ri,16).sum()+np.minimum(rb*rb,16).sum())
        photo=0.
        if use_rgb and texture>.15 and len(ids)>100:
            a,b=bottom_mapping(x,y,d,t,H,n1,n2,B)
            mx=(a*camera['fx']+camera['cx']-.5).astype(np.float32)[None]
            my=(b*camera['fy']+camera['cy']-.5).astype(np.float32)[None]
            warped=cv2.remap(ref,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel()
            denom=np.sqrt(np.sum(warped**2)*np.sum(target**2))+1e-9
            photo=float(1-np.sum(warped*target)/denom)
        loss=raw_loss+config.rgb_weight*photo
        profile.append(dict(n_upper=float(n1),n_lower=n2,thickness_m=t,loss=loss,raw_loss=raw_loss,photo_error=photo))
    if not profile:return dict(base,status='no_physical_solution')
    best=min(profile,key=lambda r:r['loss'])
    plausible=[r for r in profile if r['loss']<=best['loss']+config.profile_delta]
    tlo=min(r['thickness_m'] for r in plausible);thi=max(r['thickness_m'] for r in plausible)
    width=thi-tlo
    # Use the minimax center of the plausible thickness range, not an unstable boundary optimum.
    point=min(plausible,key=lambda r:abs(r['thickness_m']-(tlo+thi)/2))
    accepted=width<=config.max_profile_width_m
    status='measured' if accepted else 'ambiguous_refractive_index'
    if use_rgb and texture>.15 and best['photo_error']>config.max_photo_error:
        accepted=False;status='reference_inconsistent'
    base.update(status=status,accepted=accepted,thickness_m=point['thickness_m'] if accepted else None,
                candidate_thickness_m=point['thickness_m'],internal_depth_m=d+point['thickness_m'] if accepted else None,
                n_upper_estimate=point['n_upper'],n_lower_estimate=point['n_lower'],effective_index=float(eff),
                profile_range_m=[tlo,thi],profile_width_m=width,internal_points=int(inner_mask.sum()),internal_tiles=tiles,
                bottom_points=int(bottom_mask.sum()),texture_score=texture,photo_error=best['photo_error'],
                raw_effective_thickness_m=float(np.median((zi-d)/ray_factors(eff,qi))),profile=profile)
    return base
