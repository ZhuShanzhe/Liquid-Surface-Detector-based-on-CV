"""Optional RGB-D evidence. Geometric fallback is immutable and always retained."""
import cv2
import numpy as np
import torch
from torch import nn

from .empty_geometry_v41 import height_calibration
from .frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from .liquid_height_metrics import world_points
from .rgbd_evidence_model import observed_features
from .surface_refinement import StereoNoiseModel, censored_clearance, weighted_median
from .symmetric_visual_meter import symmetric_candidates
from .uneven_surface_meter import UnevenSurfaceMeter


class CandidateTrustNet(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(16,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,2))
    def forward(self,x):return self.net(x)


def candidate_features(height,base_height,heights,noise,xy,pred,bundle):
    radius=np.maximum(.015,2.5*noise);keep=abs(heights-height)<=radius
    if not keep.any():keep=np.ones(len(heights),bool)
    cal=height_calibration(bundle);distance=abs(float(cal.camera_to_world_cv[2,3]-cal.bottom_z_m))
    s=pred['semantic'];o=pred['origin'];r=pred['reliability'][0]
    q=np.clip(xy.astype(float),0,None);tiles=np.unique((q[:,0]//40+32*(q[:,1]//40))[keep]).size
    return np.array([height/max(distance,.1),(height-(base_height or 0))/max(float(np.median(noise)),.02),
        np.log1p(len(heights))/10,float(keep.mean()),min(tiles/32,1),float(np.median(noise))/max(distance,.1),
        float(s[2,keep].mean()),float(s[3,keep].mean()),float(s[1,keep].mean()),
        float(o[1,keep].mean()),float(o[2,keep].mean()),float(o[3,keep].mean()),
        float((o[5,keep]+o[6,keep]).mean()),float(r[keep].mean()),
        float(np.median(abs(heights[keep]-height)/np.maximum(noise[keep],.001))),height/max(float(bundle['rim_z_m']),.1)],np.float32)


def build_observed_candidates(base,depth,prediction):
    cal=height_calibration(base.bundle);valid=base.ordinary_container_roi&np.isfinite(depth)&(depth>0)
    y,x=np.nonzero(valid);xyz=world_points(depth,cal);heights=xyz[...,2][valid]-cal.bottom_z_m;z=depth[valid]
    if not len(z):return [],{'valid':valid,'xyz':xyz,'noise':np.array([])}
    pred={k:v[:,valid] for k,v in prediction.items()}
    sensor=base.bundle['sensor_family'];model=None if sensor=='tof' else StereoNoiseModel.simulation_proxy(sensor)
    noise=.0015+.0007*z if model is None else np.sqrt((model.depth_noise_floor_m+model.depth_noise_quadratic*z*z)**2+(model.disparity_sigma_px*z*z/(cal.intrinsics[0,0]*model.baseline_m))**2)
    frozen=FrozenSymmetricGeometryMeter(base).measure(depth_m=depth)['liquid_depth_m']
    proposals=[{'height_m':c['height_m'],'source':'symmetric_peak'} for c in symmetric_candidates(heights,np.column_stack([x,y]),noise)[:4]]
    weights=pred['semantic'][2]*pred['origin'][1]*pred['reliability'][0]
    if (weights>.2).sum()>=24 and weights.sum()>8:
        proposals.append({'height_m':float(weighted_median(heights,weights)),'source':'reliable_liquid_weighted_median'})
        if model is not None:
            rays=np.column_stack(((x-cal.intrinsics[0,2])/cal.intrinsics[0,0],(y-cal.intrinsics[1,2])/cal.intrinsics[1,1],np.ones(len(x))))
            cosine=-(rays@cal.camera_to_world_cv[:3,:3].T)[:,2]
            # All soft-weighted returns are fitted, not a hard height-truncated subset.
            clearance=censored_clearance(z,cosine,weights/weights.sum(),cal.intrinsics[0,0],model)
            if clearance is not None:
                proposals.append({'height_m':float(cal.camera_to_world_cv[2,3]-cal.bottom_z_m-clearance),'source':'reliable_sensor_domain'})
    result=[]
    for c in proposals:
        if not np.isfinite(c['height_m']) or any(abs(c['height_m']-q['height_m'])<.002 for q in result):continue
        c['features']=candidate_features(c['height_m'],frozen,heights,noise,np.column_stack([x,y]),pred,base.bundle)
        result.append(c)
    return result,{'valid':valid,'xyz':xyz,'noise':noise,'baseline_height_m':frozen}


def supported_float_height(prediction,context):
    valid=context['valid'];xyz=context['xyz'];cal=context['calibration']
    sem=prediction['semantic'];origin=prediction['origin']
    mask=valid&(sem[3]>=.6)&(origin[2]>=.3)&(prediction['reliability'][0]>=.6)
    n,labels=cv2.connectedComponents(mask.astype(np.uint8));peaks=[];components=[]
    heights=xyz[...,2]-cal.bottom_z_m
    local=cv2.medianBlur(np.where(valid,heights,0).astype(np.float32),3)
    tolerance=max(.015,3*float(np.median(context['noise'])) if len(context['noise']) else .015)
    for i in range(1,n):
        component=labels==i
        if component.sum()<4:continue
        stable=component&(abs(heights-local)<=tolerance)
        if stable.sum()<3:continue
        peak=float(np.percentile(heights[stable],95));peaks.append(peak)
        yy,xx=np.nonzero(component)
        components.append({'pixels':int(component.sum()),'stable_returns':int(stable.sum()),'height_m':peak,
            'bbox_xyxy_px':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
            'semantic_score':float(sem[3,component].mean()),'source_score':float(origin[2,component].mean()),
            'metric_reliability_score':float(prediction['reliability'][0,component].mean())})
    return max(peaks) if peaks else None,components,mask


class AdaptiveRGBDMeter:
    def __init__(self,base,*,dry_depth_m,predictor,gate_checkpoint=None):
        self.base=base;self.dry=np.asarray(dry_depth_m);self.predictor=predictor
        self.frozen=FrozenSymmetricGeometryMeter(base);self.gate=None;self.threshold=1.01
        self.geometric_peak=UnevenSurfaceMeter(base,self.dry)
        if gate_checkpoint is not None:
            state=torch.load(gate_checkpoint,map_location='cpu',weights_only=False)
            self.gate=CandidateTrustNet().eval();self.gate.load_state_dict(state['model'])
            self.threshold=float(state['threshold']);self.mean=np.asarray(state['mean'],np.float32);self.scale=np.asarray(state['scale'],np.float32)

    def measure(self,*,rgb_bgr,depth_m,mode='auto'):
        if mode not in {'off','on','auto'}:raise ValueError('visual_mode_off_on_auto_required')
        old=self.frozen.measure(depth_m=depth_m)
        if mode=='off':return {**old,'visual_module_executed':False,'visual_guidance_used':False,'mode':mode}
        roi=self.base.ordinary_container_roi
        features=observed_features(rgb_bgr,depth_m,self.dry,roi,self.base.bundle)
        pred=self.predictor.predict(features)
        proposals,ctx=build_observed_candidates(self.base,np.asarray(depth_m),pred)
        ctx['calibration']=height_calibration(self.base.bundle)
        value=old['liquid_depth_m'];selected=None
        if self.gate is not None and proposals:
            x=np.stack([c['features'] for c in proposals]);x=np.clip((x-self.mean)/self.scale,-20,20)
            with torch.inference_mode():trust=self.gate(torch.from_numpy(x)).sigmoid().numpy()
            for c,t in zip(proposals,trust,strict=True):c['improvement_probability']=float(t[0]);c['within_tolerance_probability']=float(t[1])
            selected=max(proposals,key=lambda c:c['improvement_probability']*c['within_tolerance_probability'])
            threshold=self.threshold if mode=='auto' else .5
            if (selected['improvement_probability']>=threshold and selected['within_tolerance_probability']>=.7
                    and (mode=='on' or old['precision_eligible'])):
                value=selected['height_m']
        highest,components,float_mask=supported_float_height(pred,ctx)
        learned_peak=highest
        geometric_peak=None
        if highest is None:
            geometric_peak=self.geometric_peak.measure(np.asarray(depth_m))['highest_visible_height_m']
            highest=geometric_peak
        use_float=bool(components)
        combined=max([v for v in [value,highest] if v is not None]) if value is not None or highest is not None else None
        effective=float((pred['semantic'][2]*pred['origin'][1]*pred['reliability'][0])[ctx['valid']].sum())
        diagnostic={**old['diagnostics'],'adaptive_candidates':[{k:v for k,v in c.items() if k!='features'} for c in proposals],
                    'float_components':components,'float_candidate_pixels':int(float_mask.sum()),'auto_threshold':self.threshold,
                    'effective_liquid_return_support':effective,
                    'liquid_return_support_warning':effective<24,
                    'highest_source':'rgbd_supported_float' if components else 'preserved_geometric_peak',
                    'learned_float_peak_m':learned_peak,'geometric_peak_m':geometric_peak,
                    'semantics_and_return_reliability_separate':True,'truth_used_for_inference':False,
                    'classification_probabilities_are_not_metrological_guarantees':True}
        return {**old,'liquid_depth_m':value,'highest_height_m':combined,'numeric_output':value is not None,
                'route':'adaptive_rgbd_evidence_v2','mode':mode,'visual_module_executed':True,
                'visual_guidance_used':bool(value!=old['liquid_depth_m'] or use_float),
                'liquid_override_used':bool(value!=old['liquid_depth_m']),
                'diagnostics':diagnostic}
