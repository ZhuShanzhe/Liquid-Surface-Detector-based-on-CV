"""Opt-in opaque ToF compound repair; no simulator labels or target heights.

The frozen facade remains the baseline. This module independently audits native
input validity, wet/dry surface evidence and highest-point support. Visual
features are allowed ONLY for floating-object highest height, never liquid.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import uniform_filter
from liquid_depth.empty_geometry_v41 import height_calibration
from liquid_depth.liquid_height_metrics import world_points

VERSION='opaque_tof_compound_evidence_v2_20261020'

@dataclass(frozen=True)
class CompoundRepairPolicy:
    # Sensor-domain validity, NOT a lower limit on liquid height.
    minimum_sensor_depth_m: float=.08
    interior_radius_fraction: float=.86
    minimum_changed_returns: int=64
    minimum_changed_tiles: int=6
    minimum_changed_fraction: float=.20
    replacement_score_ratio: float=1.50
    minimum_replacement_delta_m: float=.030

def sanitize_capture(depth_m,minimum_depth_m=.08,validity=None):
    d=np.asarray(depth_m)
    if d.ndim!=2:raise ValueError('metric_depth_2d_required')
    if not np.isfinite(minimum_depth_m) or minimum_depth_m<0:raise ValueError('invalid_sensor_minimum')
    keep=np.isfinite(d)&(d>0)&(d>=minimum_depth_m)
    if validity is not None:
        v=np.asarray(validity)
        if v.shape!=d.shape or v.dtype!=np.bool_:raise ValueError('aligned_boolean_capture_validity_required')
        keep&=v
    return np.where(keep,d,0).astype(d.dtype),dict(rejected_nonfinite=int((~np.isfinite(d)).sum()),
        rejected_below_sensor_minimum=int((np.isfinite(d)&(d>0)&(d<minimum_depth_m)).sum()),
        rejected_native_invalid=int(((d>0)&~np.asarray(validity)).sum()) if validity is not None else 0,
        native_validity_supplied=validity is not None,no_maximum_depth_cutoff=True)

def supported_local_peak(height,mask,noise):
    """A peak requires its own measured neighbourhood, not one arbitrary max."""
    mask=np.asarray(mask,bool)&np.isfinite(height)
    n=uniform_filter(mask.astype(float),3,mode='constant')*9
    total=uniform_filter(np.where(mask,height,0.),3,mode='constant')*9
    second=uniform_filter(np.where(mask,height*height,0.),3,mode='constant')*9
    mean=total/np.maximum(n,1)
    std=np.sqrt(np.maximum(0,second/np.maximum(n,1)-mean*mean))
    limit=np.maximum(.02,3*np.asarray(noise))
    witness=mask&(n>=4)&(abs(height-mean)<=limit)&(std<=np.maximum(.03,4*np.asarray(noise)))
    if not witness.any():return None,0
    return float(mean[witness].max()),int(witness.sum())

class CompoundToFRepair:
    def __init__(self,base,dry_depth_m,policy=None):
        if base.bundle['sensor_family']!='tof':raise ValueError('tof_only_compound_repair')
        self.base=base;self.policy=policy or CompoundRepairPolicy();self.dry=np.asarray(dry_depth_m)
        self.cal=height_calibration(base.bundle)
        self.dry_xyz=world_points(self.dry,self.cal)
        self.dry_height=self.dry_xyz[...,2]-self.cal.bottom_z_m
        self.dry_sigma=.0015+.0007*np.maximum(self.dry,0)
        self.aperture=np.asarray(base.ordinary_container_roi,bool)
        self.dry_floor=(self.dry>0)&(abs(self.dry_height)<=np.maximum(.05,4*self.dry_sigma))
        yy,xx=np.indices(self.dry.shape);y,x=np.nonzero(self.aperture)
        if not len(x):raise ValueError('empty_calibrated_aperture')
        self.tiles=(np.clip((xx-x.min())/max(np.ptp(x),1)*4,0,3).astype(int)
                    +4*np.clip((yy-y.min())/max(np.ptp(y),1)*4,0,3).astype(int))

    def estimate_surface(self,depth_m,legacy_liquid):
        d=np.asarray(depth_m);xyz=world_points(d,self.cal)
        h=xyz[...,2]-self.cal.bottom_z_m;sigma=.0015+.0007*np.maximum(d,0)
        normalized=np.sum(abs(xyz[...,:2]/np.asarray(self.cal.radii_m))**self.cal.footprint_exponent,axis=-1)
        inner=normalized<=self.policy.interior_radius_fraction**self.cal.footprint_exponent
        valid=self.aperture&self.dry_floor&inner&np.isfinite(d)&(d>=self.policy.minimum_sensor_depth_m)
        # This is evidence selection inside the calibrated vessel, not an
        # output clamp. Floating-object tops are handled separately below.
        margin=np.maximum(.05,4*sigma)
        valid&=(h>=-margin)&(h<=float(self.base.bundle['rim_z_m'])+margin)
        delta=h-self.dry_height
        combined=np.sqrt(sigma*sigma+self.dry_sigma*self.dry_sigma)
        changed=valid&(abs(delta)>np.maximum(.025,3*combined))
        # Physical footprint tiles, not the wider rim's image bounding box.
        # Otherwise a low surface in a deep vessel spuriously occupies only
        # four image cells although its returns cover the container bottom.
        xy=xyz[...,:2]/np.asarray(self.cal.radii_m)
        ti=np.clip(((xy+1)*2).astype(int),0,3)
        tile=ti[...,0]+4*ti[...,1]
        z=h[changed];tiles=len(np.unique(tile[changed]))
        frac=float(changed.sum()/max(valid.sum(),1));proposal=None;mad=None;strong=False
        if len(z)>=self.policy.minimum_changed_returns and tiles>=self.policy.minimum_changed_tiles and frac>=self.policy.minimum_changed_fraction:
            center=float(np.median(z));mad=max(float(np.median(abs(z-center))),float(np.median(sigma[changed])))
            coherent=changed&(abs(h-center)<=max(.025,4*mad))
            z=h[coherent];tiles=len(np.unique(tile[coherent]))
            if len(z)>=self.policy.minimum_changed_returns and tiles>=self.policy.minimum_changed_tiles:
                proposal=float(np.median(z));strong=True
        selected=legacy_liquid;replaced=False;old_score=None;new_score=None
        if strong:
            width=max(.020,2.5*mad,3*float(np.median(sigma[changed])))
            # Persistent empty-frame returns can be bottom/wall echoes. They
            # retain equal treatment at all candidate heights but less identity
            # evidence than a coherent wet-only layer. No truth is consulted.
            identity=np.where(changed,1.,.15)
            def score(v):return float(np.sum(identity[valid]*np.exp(-.5*((h[valid]-v)/width)**2)))
            new_score=score(proposal);old_score=score(legacy_liquid) if legacy_liquid is not None else 0.
            if legacy_liquid is None or (abs(proposal-legacy_liquid)>max(self.policy.minimum_replacement_delta_m,4*float(np.median(sigma[valid])))
                    and new_score>self.policy.replacement_score_ratio*max(old_score,1e-9)):
                selected=proposal;replaced=True
            band=max(.025,4*mad)
            surface=valid&(abs(h-proposal)<=band)
        elif selected is not None:
            band=max(.025,4*float(np.median(sigma[valid])) if valid.any() else .025)
            surface=valid&(abs(h-selected)<=band)
        else:surface=np.zeros(d.shape,bool)
        peak,support=supported_local_peak(h,surface,sigma)
        diagnostics=dict(interior_valid_returns=int(valid.sum()),changed_returns=int(changed.sum()),changed_fraction=frac,
            changed_tiles=tiles,changed_proposal_m=proposal,changed_mad_m=mad,changed_evidence_strong=strong,
            legacy_liquid_m=legacy_liquid,liquid_replaced=replaced,old_identity_score=old_score,new_identity_score=new_score,
            surface_peak_m=peak,surface_peak_support=support,no_truth_or_visual_liquid_input=True)
        return selected,peak,diagnostics,dict(xyz=xyz,height=h,sigma=sigma,valid=valid,inner=inner)

    def repair(self,result,depth_m,*,rgb_bgr=None,float_backend=None,visual_peak=False):
        liquid,peak,diag,ctx=self.estimate_surface(depth_m,result.get('liquid_depth_m'))
        accepted=[];float_diag=dict(requested=bool(visual_peak),executed=False)
        if visual_peak and float_backend is not None and rgb_bgr is not None:
            from liquid_depth.rgbd_evidence_model import observed_features
            from liquid_depth.floating_rank_v8 import proposals
            from liquid_depth.floating_apex_v5 import select_v5
            raw=np.asarray(depth_m)
            # Fixed calibrated aperture + actual 3D interior, not arbitrary
            # near-camera back-projections from any point in the whole image.
            roi=self.aperture&ctx['inner']&self.dry_floor&(raw>=self.policy.minimum_sensor_depth_m)&np.isfinite(raw)
            pred=float_backend.predictor.predict(observed_features(rgb_bgr,raw,self.dry,roi,self.base.bundle))
            original,_=proposals(self.base,self.dry,rgb_bgr,raw,pred,liquid,roi,
                float_backend.previous,float_backend.apex,include_relaxed=False)
            supported=[]
            for c in original:
                yy,xx=np.nonzero(c['mask']);h=ctx['height'][c['mask']]
                if len(h)<3 or np.ptp(xx)<1 or np.ptp(yy)<1:continue
                # Local robust fitting checks observed metric support, and
                # cannot create extra independent samples by interpolation.
                a=np.column_stack((np.ones(len(xx)),xx-np.median(xx),yy-np.median(yy)))
                beta=np.linalg.lstsq(a,h,rcond=None)[0];resid=h-a@beta
                noise=float(np.median(ctx['sigma'][c['mask']]))
                if np.median(abs(resid))>max(.015,3*noise):continue
                witness=np.sum(abs(h-c['height_raw_m'])<=max(.015,3*noise))
                if witness<3:continue
                supported.append(c)
            chosen,accepted=select_v5(supported,float_backend.apex['policy_by_sensor']['tof'])
            float_diag=dict(requested=True,executed=True,original_candidates=len(original),supported_candidates=len(supported),
                selected=None if chosen is None else {k:v for k,v in chosen.items() if k not in {'mask','features','indices'}})
            if chosen is not None:
                peak=chosen['height_m'] if peak is None else max(peak,chosen['height_m'])
        fallback=False
        if peak is None:
            peak=result.get('highest_height_m');fallback=True
        diagnostics={**result.get('diagnostics',{}),'compound_repair':{**diag,'float_peak':float_diag,'peak_legacy_fallback':fallback}}
        return {**result,'liquid_depth_m':liquid,'highest_height_m':peak,'numeric_output':liquid is not None,
            'route':result.get('route','')+'+'+VERSION,'diagnostics':diagnostics,
            'compound_repair_version':VERSION,'liquid_visual_guidance_used':False,
            'liquid_geometry_override_used':bool(diag['liquid_replaced']),
            'visual_module_executed':bool(float_diag['executed']),
            'visual_guidance_used':bool(float_diag['executed'] and accepted),
            'highest_visual_guidance_used':bool(float_diag['executed'] and accepted),
            'field_metrology_qualified':False}
