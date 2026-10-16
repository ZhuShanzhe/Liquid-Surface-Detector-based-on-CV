"""Finite opaque-sided vessel IR proxy with metric bottom texture.

No glass-wall transmission or spectral sensor certification. Operator ROI is a
fixed empty-bottom footprint, not a per-test-frame truth mask. All algorithms see
the same images/masks; only the evaluator reads hidden liquid height and index.
"""
import cv2
import numpy as np
from .refraction_images_v23 import bottom_map,inverse_bottom,texture


def geometry(bottom,capacity,width_m,fx=450.,baseline=.055):
    if not 0<capacity<bottom or width_m<=0:raise ValueError('invalid_vessel_geometry')
    # One common sensor crop, fixed by the installed rim geometry, not liquid H.
    rim=bottom-capacity;depth=.8*width_m
    xlo=int(np.floor(fx*(-width_m/2-baseline)/rim))-6
    xhi=int(np.ceil(fx*(width_m/2)/rim))+6
    ylo=int(np.floor(-fx*depth/2/rim))-6;yhi=int(np.ceil(fx*depth/2/rim))+6
    h,w=yhi-ylo,xhi-xlo
    yy,xx=np.indices((h,w));q=np.stack(((xx+xlo)/fx,(yy+ylo)/fx),-1)
    masks={}
    for ch,cx in [('left',0.),('right',baseline)]:
        p=q*bottom+np.array([cx,0.])
        masks[ch]=((np.abs(p[...,0])<width_m/2)&(np.abs(p[...,1])<depth/2)).astype(np.uint8)*255
    cal=dict(width=w,height=h,fx=fx,cx=-xlo,cy=-ylo,bottom_m=bottom,baseline_m=baseline,
        container_max_height_m=capacity,container_width_m=width_m,container_length_m=depth,
        empty_width_px=fx*width_m/bottom,empty_height_px=fx*depth/bottom,
        operator_roi_source='fixed_empty_bottom_geometry',fixed_camera=True,
        opaque_sidewall=True,glass_wall_raytracing=False,spectral_sensor_calibrated=False)
    return cal,masks


def render(seed,height,index,bottom,capacity,width_m,condition,grain_m=.025,supersample=3):
    if condition not in ('transparent','translucent_mild'):raise ValueError('outside_target_scope')
    if not 0<=height<capacity:raise ValueError('height_outside_capacity')
    cal,masks=geometry(bottom,capacity,width_m);fx=cal['fx'];ss=supersample
    hh,ww=cal['height']*ss,cal['width']*ss
    yy,xx=np.indices((hh,ww));q=np.stack((((xx+.5)/ss-.5-cal['cx'])/fx,((yy+.5)/ss-.5-cal['cy'])/fx),-1)
    # Texture grain is in metres, independent of camera distance. Integrate the
    # rendered radiance across native pixels instead of enlarging tiny images.
    step=.0025;size=1024
    rngtex=np.random.default_rng(seed);tex=rngtex.random((size,size)).astype(np.float32)
    tex=cv2.GaussianBlur(tex,(0,0),max(.6,grain_m/step))
    tex=(tex-tex.mean())/max(float(tex.std()),1e-6)*.15+.5;tex=np.clip(tex,.05,.95)
    proj=texture(seed+117,size)
    rng=np.random.default_rng(seed+int(round(height*1e7))*101+7919)
    images={}
    for channel,centre in [('left',(0.,0.)),('right',(.055,0.))]:
        x=bottom_map(q,height,index,bottom,centre)
        qp=inverse_bottom(x,height,index,bottom,(.055,0.))
        albedo=cv2.remap(tex,(x[...,0]/step+size/2).astype(np.float32),(x[...,1]/step+size/2).astype(np.float32),
            cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
        pattern=cv2.remap(proj,(qp[...,0]*fx+size/2).astype(np.float32),(qp[...,1]*fx+size/2).astype(np.float32),
            cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
        # Ballistic attenuation uses metric path length, not H/B. Still a mild
        # scatter/veil proxy rather than a volumetric radiative-transfer renderer.
        path=2*height*np.sqrt(1+np.sum(q*q,axis=-1)/index**2)
        attenuation=np.exp(-.65*path) if condition=='translucent_mild' else np.ones_like(path)
        bottom_signal=.025+albedo*(.4+.5*pattern)*attenuation+(1-attenuation)*.13
        if condition=='translucent_mild':bottom_signal=cv2.GaussianBlur(bottom_signal,(0,0),(.20+.65*height)*ss)
        opening=np.asarray(centre)+q*(bottom-capacity)
        at_open=(np.abs(opening[...,0])<width_m/2)&(np.abs(opening[...,1])<.4*width_m)
        at_bottom=(np.abs(x[...,0])<width_m/2)&(np.abs(x[...,1])<.4*width_m)
        signal=np.where(at_open,np.where(at_bottom,bottom_signal,.075),.14)
        signal=cv2.GaussianBlur(signal.astype(np.float32),(0,0),.40*ss)
        signal=cv2.resize(signal,(cal['width'],cal['height']),interpolation=cv2.INTER_AREA)
        sigma=.003 if condition=='transparent' else .006
        signal+=rng.normal(0,sigma,signal.shape).astype(np.float32)
        signal*=1+(-.025 if channel=='left' else .035)
        images[channel]=np.uint8(np.clip(signal,0,1)*255)
    truth=dict(height_m=height,index=index,grain_m=grain_m,seed=seed)
    return images,cal,masks,truth
