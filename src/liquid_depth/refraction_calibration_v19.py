"""Quality checks for an actually captured empty-container calibration."""
import numpy as np
from .refraction_inverse_v19 import calibrate_empty

def assess_reference(rgb,empty_raw,camera,max_saturated_fraction=.01):
    image=np.asarray(rgb,float)
    if image.ndim!=3 or image.shape[2]!=3 or not np.isfinite(image).all() or image.min()<0 or image.max()>1:
        raise ValueError('Expected finite RGB intensities in [0,1]')
    cal=calibrate_empty(np.asarray(empty_raw,float)*float(camera.get('depth_scale_to_m',1.)))
    if image.shape[:2]!=cal['roi'].shape:raise ValueError('Reference dimensions do not match')
    samples=image[cal['roi']]
    fraction=float((samples.max(1)>=.995).mean())
    return dict(usable=fraction<=max_saturated_fraction,saturated_fraction=fraction,
                saturation_limit=max_saturated_fraction,calibration_pixels=int(cal['roi'].sum()))

def apply_reference_quality(result,rgb,empty_raw,camera):
    r=dict(result);quality=assess_reference(rgb,empty_raw,camera);r['reference_quality']=quality
    if not quality['usable']:
        r.update(accepted=False,thickness_m=None,internal_depth_m=None,status='reference_requires_exposure_adjustment')
    return r
