"""V20.1: conservative multi-scale evidence with the frozen V19 fallback."""
import numpy as np
from .refraction_inverse_v20 import estimate_layers as spatial_estimate
from .refraction_inverse_v19 import estimate_layers as legacy_estimate
from .refraction_evidence_v19 import add_chromatic_evidence as legacy_photo,qualify_profile as legacy_qualify
from .refraction_multiscale_v20 import add_chromatic_evidence,qualify_profile
from .refraction_calibration_v19 import apply_reference_quality

def choose_result(spatial,legacy,scales):
    """Selection uses observable diagnostics only; scene labels are never read."""
    base=dict(spatial)
    if not spatial.get('temporal'):
        return base
    valid=len(scales)==3 and all(r.get('accepted',False) for r in scales)
    lo=min((r['profile_range_m'][0] for r in scales),default=0.)
    hi=max((r['profile_range_m'][1] for r in scales),default=float('inf'))
    thicknesses=[r['candidate_thickness_m'] for r in scales]
    # Use the union, never the intersection, of scale-dependent plausible sets.
    width=hi-lo
    valid=valid and width<=.010 and np.ptp(thicknesses)<=.003
    diagnostic=dict(scales=[dict(blur=b,accepted=q.get('accepted',False),
          thickness_m=q.get('candidate_thickness_m'),range_m=q.get('profile_range_m'),
          status=q['status']) for b,q in zip([.6,1.,1.4],scales)],
          union_range_m=[lo,hi] if scales else None,consistent=bool(valid),
          legacy_accepted=bool(legacy.get('accepted',False)))
    if valid:
        center=float(np.median(thicknesses))
        chosen=min(scales,key=lambda q:abs(q['candidate_thickness_m']-center))
        if legacy.get('accepted') and abs(center-legacy['thickness_m'])>.003:
            base.update(accepted=False,thickness_m=None,internal_depth_m=None,status='method_disagreement',
                selection='rejected',consensus=diagnostic,liquid_identity_verified=False)
            return base
        result=dict(chosen,status='multi_scale_refraction_supported',selection='v20_multiscale',
            thickness_m=center,internal_depth_m=spatial['top_depth_m']+center,
            profile_range_m=[lo,hi],profile_width_m=width,consensus=diagnostic,
            liquid_identity_verified=False)
        return result
    if legacy.get('accepted'):
        return dict(legacy,status='legacy_evidence_supported',selection='v19_fallback',
            temporal=spatial['temporal'],consensus=diagnostic,liquid_identity_verified=False)
    base.update(accepted=False,thickness_m=None,internal_depth_m=None,
        status='no_consistent_refraction_evidence' if scales else spatial['status'],
        selection='rejected',consensus=diagnostic,liquid_identity_verified=False)
    return base

def estimate_layers(rgb,raw,empty_rgb,empty_raw,camera,use_rgb=True):
    spatial=spatial_estimate(rgb,raw,empty_rgb,empty_raw,camera,use_rgb=use_rgb)
    if not spatial.get('temporal'):return spatial
    legacy=legacy_estimate(rgb,raw,empty_rgb,empty_raw,camera,use_rgb=True)
    legacy=apply_reference_quality(legacy_qualify(legacy_photo(legacy,rgb,empty_rgb,empty_raw,camera)),
                                  empty_rgb,empty_raw,camera)
    scales=[]
    if spatial.get('profile'):
        for blur in [.6,1.,1.4]:
            q=qualify_profile(add_chromatic_evidence(spatial,rgb,empty_rgb,empty_raw,camera,blur=blur))
            scales.append(apply_reference_quality(q,empty_rgb,empty_raw,camera))
    return choose_result(spatial,legacy,scales)
