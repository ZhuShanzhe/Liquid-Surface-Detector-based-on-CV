"""Experimental equal-candidate selector with semantic and return evidence separated."""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .empty_geometry_v41 import height_calibration
from .liquid_height_metrics import world_points
from .surface_refinement import StereoNoiseModel, weighted_median


@dataclass(frozen=True)
class EvidencePolicy:
    min_returns: int=24
    min_tiles: int=3
    min_score_margin: float=.15
    max_uninformative_rgb_fraction: float=.95


def symmetric_candidates(heights,xy,noise,semantic=None):
    """Same clustering, scoring and support requirements at every candidate height."""
    h=np.asarray(heights,float);xy=np.asarray(xy,float);noise=np.maximum(np.asarray(noise,float),.001)
    if len(h)<3:return []
    constant_noise_median=np.median(noise)
    step=float(np.clip(constant_noise_median*.35,.005,.05))
    edges=np.linspace(h.min()-step,h.max()+step,min(6000,max(5,int(np.ptp(h)/step)+3)))
    mass=np.histogram(h,edges)[0].astype(float)
    density=gaussian_filter1d(mass,1.)
    # Padding keeps minority clusters at either observed-range boundary eligible.
    # scipy.find_peaks otherwise silently omits those endpoint maxima.
    peaks,_=find_peaks(np.r_[0.,density,0.],prominence=max(.5,density.max()*.02))
    peaks=peaks-1
    # Include the dominant cluster even if it is at a histogram boundary.
    peaks=np.unique(np.r_[peaks,np.argmax(density)])
    candidates=[]
    qx=np.clip((xy[:,0]-xy[:,0].min())/max(np.ptp(xy[:,0]),1e-9)*4,0,3).astype(int)
    qy=np.clip((xy[:,1]-xy[:,1].min())/max(np.ptp(xy[:,1]),1e-9)*4,0,3).astype(int)
    tile=qx+4*qy
    for index in peaks:
        center=float((edges[index]+edges[index+1])*.5)
        keep=abs(h-center)<=np.maximum(.012,2.5*noise)
        if keep.sum()<3:continue
        # Semantic foreground is a prior, never a label of the physical return origin.
        prior=np.ones(len(h))
        if semantic is not None:
            cls=1 if abs(center)<=max(.03,2*constant_noise_median) else 2
            prior=.05+.95*np.asarray(semantic)[cls]
        robust=np.exp(-.5*((h-center)/noise)**2)
        weights=prior*robust
        weights[~keep]=0
        ntiles=len(np.unique(tile[keep]))
        score=float(weights.sum()*np.sqrt(ntiles/16.))
        value=float(weighted_median(h[keep],np.maximum(weights[keep],1e-9)))
        candidates.append({'height_m':value,'score':score,'support_returns':int(keep.sum()),
                           'tiles':ntiles,'mask':keep,'weights':weights,
                           'interpretation':'bottom_compatible' if abs(value)<=max(.03,2*constant_noise_median) else 'nonbottom_candidate'})
    candidates.sort(key=lambda c:c['score'],reverse=True)
    distinct=[]
    for c in candidates:
        if not any(abs(c['height_m']-d['height_m'])<max(.015,constant_noise_median) for d in distinct):distinct.append(c)
    return distinct


class SymmetricVisualMeter:
    def __init__(self,ordinary_meter,*,predictor=None,policy=None):
        self.base=ordinary_meter;self.predictor=predictor;self.policy=policy or EvidencePolicy()

    def measure(self,*,depth_m,rgb_bgr=None,probabilities=None):
        depth=np.asarray(depth_m)
        roi=self.base.ordinary_container_roi
        if depth.shape!=roi.shape:raise ValueError('aligned_depth_required')
        cal=height_calibration(self.base.bundle)
        xyz=world_points(depth,cal)
        valid=roi&np.isfinite(depth)&(depth>0)
        if probabilities is None and self.predictor is not None:
            if rgb_bgr is None:raise ValueError('rgb_required_for_visual_route')
            probabilities=self.predictor(rgb_bgr,roi)
        if probabilities is not None:
            probabilities=np.asarray(probabilities,float)
            if probabilities.shape!=(4,*depth.shape) or not np.isfinite(probabilities).all() or np.any(probabilities<0) or not np.allclose(probabilities.sum(0),1,atol=.002):
                raise ValueError('normalized_aligned_semantic_probabilities_required')
        baseline=self.base.measure('ordinary_diffuse',depth_m=depth)['liquid_depth_m']
        y,x=np.nonzero(valid);h=xyz[...,2][valid]-cal.bottom_z_m
        z=depth[valid]
        sensor=self.base.bundle['sensor_family']
        if sensor=='tof':noise=.0015+.0007*z
        else:
            model=StereoNoiseModel.simulation_proxy(sensor)
            noise=np.sqrt((model.depth_noise_floor_m+model.depth_noise_quadratic*z*z)**2+(model.disparity_sigma_px*z*z/(cal.intrinsics[0,0]*model.baseline_m))**2)
        sem=None if probabilities is None else probabilities[:,valid]
        candidates=symmetric_candidates(h,np.column_stack([x,y]),noise,sem)
        value=baseline;flags=[];eligible=True
        if len(h)<self.policy.min_returns:flags.append('insufficient_current_returns');eligible=False
        if candidates:
            best=candidates[0];value=best['height_m']
            margin=(best['score']-candidates[1]['score'])/max(best['score'],1e-9) if len(candidates)>1 else 1.
            if best['support_returns']<self.policy.min_returns or best['tiles']<self.policy.min_tiles:
                flags.append('insufficient_spatial_support');eligible=False
            if margin<self.policy.min_score_margin:flags.append('competing_layers_ambiguous');eligible=False
        else:
            best=None;margin=0.;flags.append('no_supported_candidate');eligible=False
        if rgb_bgr is not None and roi.any():
            grey=cv2.cvtColor(np.asarray(rgb_bgr),cv2.COLOR_BGR2GRAY)
            uninformative=float(np.mean((grey[roi]<=2)|(grey[roi]>=253)))
            if probabilities is not None and uninformative>=self.policy.max_uninformative_rgb_fraction:
                flags.append('visual_semantics_unobservable');eligible=False
        if (best is not None and probabilities is not None and
                best['interpretation']=='bottom_compatible' and float(np.mean(probabilities[2,valid]))>.7):
            flags.append('liquid_appearance_bottom_returns_conflict');eligible=False
        # Float height uses semantic components and supported local upper quantiles,
        # not the single largest raw point. The actual sampled max is diagnostic only.
        highest=value;rawmax=None;float_components=0
        if probabilities is not None:
            floating=valid&(probabilities[3]>=.7)&(probabilities[3]>probabilities[2])
            n,labels=cv2.connectedComponents(floating.astype(np.uint8))
            upper=[]
            for i in range(1,n):
                mask=labels==i
                if mask.sum()<6:continue
                points=xyz[...,2][mask]-cal.bottom_z_m
                med=cv2.medianBlur(np.where(mask,xyz[...,2]-cal.bottom_z_m,0).astype(np.float32),3)[mask]
                supported=points[np.abs(points-med)<=max(.03,3*float(np.median(noise)) if len(noise) else .03)]
                if len(supported)>=6:
                    upper.append(float(np.percentile(supported,95)));float_components+=1
            if floating.any():rawmax=float(np.max(xyz[...,2][floating]-cal.bottom_z_m))
            if upper:highest=max([v for v in [value,*upper] if v is not None])
        clean=[{k:v for k,v in c.items() if k not in ('mask','weights')} for c in candidates]
        return {'liquid_depth_m':value,'highest_height_m':highest,
                'numeric_output':value is not None,'precision_eligible':eligible,
                'route':'symmetric_visual_evidence_v1','production_qualified':False,
                'diagnostics':{'candidates':clean,'relative_score_margin':margin,
                    'flags':flags,'selected_returns':len(h),'semantic_probabilities_are_return_origin_labels':False,
                    'float_components':float_components,'raw_float_max_diagnostic_m':rawmax,
                    'fallback_value_m':baseline,'eligibility_uses_truth':False}}
