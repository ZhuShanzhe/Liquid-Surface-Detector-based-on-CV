"""Geometry-only wet-surface rescue from calibrated, currently interior returns.

No RGB, simulator label, true height or scene control enters this estimator.
"""
from dataclasses import replace
import cv2,numpy as np
from liquid_depth.compound_tof_repair import CompoundToFRepair,supported_local_peak
from liquid_depth.ordinary_surface_v35 import interior_ray_roi
from liquid_depth.liquid_height_metrics import world_points
VERSION='dynamic_wet_interior_v14_20261021'
class DynamicInteriorRepair(CompoundToFRepair):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        rim=float(self.base.bundle['rim_z_m'])
        opening=interior_ray_roi(replace(self.cal,bottom_z_m=rim-1e-5),rim,self.dry.shape,.98)
        self.current_opening=cv2.dilate(opening.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    def estimate_surface(self,depth_m,legacy_liquid):
        old_liquid,old_peak,old_diag,ctx=super().estimate_surface(depth_m,legacy_liquid)
        d=np.asarray(depth_m);h=ctx['height'];sigma=ctx['sigma'];xyz=ctx['xyz']
        # Empty-floor pixels are not the only possible wet-liquid pixels. A
        # rising surface can occlude rays that previously struck the vessel wall.
        valid=self.current_opening&ctx['inner']&np.isfinite(d)&(d>=self.policy.minimum_sensor_depth_m)
        valid&=np.isfinite(self.dry)&(self.dry>0)
        margin=np.maximum(.05,4*sigma)
        valid&=(h>=-margin)&(h<=float(self.base.bundle['rim_z_m'])+margin)
        combined=np.sqrt(sigma*sigma+self.dry_sigma*self.dry_sigma)
        # An admitted opaque surface must precede the empty return on its ray.
        changed=valid&((self.dry-d)>np.maximum(.0025,3*combined))
        xy=xyz[...,:2]/np.asarray(self.cal.radii_m);ti=np.clip(((xy+1)*2).astype(int),0,3);tile=ti[...,0]+4*ti[...,1]
        z=h[changed];tiles=int(len(np.unique(tile[changed])));fraction=float(changed.sum()/max(1,valid.sum()))
        proposal=None;mad=None;strong=False;coherent=np.zeros(d.shape,bool)
        if len(z)>=self.policy.minimum_changed_returns and tiles>=self.policy.minimum_changed_tiles and fraction>=self.policy.minimum_changed_fraction:
            center=float(np.median(z));mad=max(float(np.median(abs(z-center))),float(np.median(sigma[changed])))
            coherent=changed&(abs(h-center)<=max(.025,4*mad));tiles=int(len(np.unique(tile[coherent])))
            if coherent.sum()>=self.policy.minimum_changed_returns and tiles>=self.policy.minimum_changed_tiles:
                proposal=float(np.median(h[coherent]));strong=True
        selected=old_liquid;peak=old_peak;replaced=False;old_score=None;new_score=None;peak_support=0
        if strong:
            width=max(.020,2.5*mad,3*float(np.median(sigma[changed])))
            identity=np.where(changed,1.,.15)
            def score(value):return float(np.sum(identity[valid]*np.exp(-.5*((h[valid]-value)/width)**2)))
            new_score=score(proposal);old_score=score(old_liquid) if old_liquid is not None else 0.
            if old_liquid is None or (abs(proposal-old_liquid)>max(self.policy.minimum_replacement_delta_m,4*float(np.median(sigma[valid]))) and new_score>self.policy.replacement_score_ratio*max(old_score,1e-9)):
                selected=proposal;replaced=True
            if selected is not None:
                surface=valid&(abs(h-selected)<=max(.025,4*mad))
                dynamic_peak,peak_support=supported_local_peak(h,surface,sigma)
                if dynamic_peak is not None:peak=dynamic_peak
        dynamic=dict(version=VERSION,valid_returns=int(valid.sum()),changed_positive_returns=int(changed.sum()),
            coherent_returns=int(coherent.sum()),coherent_physical_tiles=tiles,changed_fraction=fraction,
            proposal_m=proposal,mad_m=mad,strong=strong,old_selected_m=old_liquid,replaced=replaced,
            old_score=old_score,new_score=new_score,surface_peak_support=peak_support,
            uses_current_xyz_and_empty_ray_identity=True,uses_rgb_or_truth=False,no_output_clipping=True)
        diagnostics={**old_diag,'dynamic_interior':dynamic,'liquid_replaced':bool(old_diag['liquid_replaced'] or replaced),
            'surface_peak_m':peak,'surface_peak_support':peak_support if strong else old_diag['surface_peak_support']}
        return selected,peak,diagnostics,{**ctx,'dynamic_valid':valid}
