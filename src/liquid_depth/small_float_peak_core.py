"""Highest-only small-object evidence. No liquid prediction or labels at runtime."""
from dataclasses import replace
import cv2,numpy as np
from liquid_depth.rgbd_evidence_model import observed_features,RGBDEvidencePredictor
from liquid_depth.empty_geometry_v41 import height_calibration
from liquid_depth.liquid_height_metrics import world_points
from .geometry_execution import interior_ray_roi

def peak_roi(bundle,depth):
    cal=height_calibration(bundle);rim=float(bundle['rim_z_m'])
    opening=interior_ray_roi(replace(cal,bottom_z_m=rim-1e-5),rim,depth.shape,.999)
    opening=cv2.dilate(opening.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    xyz=world_points(depth,cal);footprint=np.sum(abs(xyz[...,:2]/np.asarray(cal.radii_m))**cal.footprint_exponent,axis=-1)
    return opening&(footprint<=1.03**cal.footprint_exponent)&np.isfinite(depth)&(depth>=.08)

def float_features(rgb,depth,dry,bundle):
    roi=peak_roi(bundle,depth);x=observed_features(rgb,depth,dry,roi,bundle)
    noise=.0015+.0007*np.maximum(depth,0);local=cv2.medianBlur(np.asarray(depth,np.float32),5)
    x[9]=np.where(depth>0,np.tanh((depth-local)/np.maximum(noise,.0015)),0);x[10]=noise
    return x,roi

def dense_candidates(depth,prediction,bundle,liquid):
    cal=height_calibration(bundle);h=world_points(depth,cal)[...,2]-cal.bottom_z_m
    roi=peak_roi(bundle,depth);p=prediction['semantic'][3];rel=prediction['reliability'][0]
    sigma=.0015+.0007*np.maximum(depth,0);out=[];seen=set()
    for threshold in (.25,.5,.75):
        mask=roi&(p>=threshold)&(rel>=.20)
        n,cc,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        for k in range(1,n):
            if stats[k,cv2.CC_STAT_AREA]<3:continue
            mask=cc==k;yy,xx=np.nonzero(mask)
            if np.ptp(xx)<1 or np.ptp(yy)<1:continue
            signature=(int(yy.min()),int(xx.min()),int(yy.max()),int(xx.max()),len(xx))
            if signature in seen:continue
            seen.add(signature);z=h[mask];noise=float(np.median(sigma[mask]));qs=np.percentile(z,[10,50,75,90,95,100]);apex=float(qs[3])
            a=np.column_stack([np.ones(len(xx)),xx-np.median(xx),yy-np.median(yy)]);beta=np.linalg.lstsq(a,z,rcond=None)[0]
            residual=float(np.median(abs(z-a@beta)));w=int(np.ptp(xx)+1);ht=int(np.ptp(yy)+1)
            if residual>max(.015,3*noise) or np.sum(abs(z-apex)<=max(.015,3*noise))<3:continue
            prior=float(liquid) if liquid is not None else float(np.median(h[roi]))
            features=np.array([np.log1p(len(xx)),w,ht,len(xx)/(w*ht),p[mask].mean(),p[mask].max(),np.quantile(p[mask],.1),rel[mask].mean(),rel[mask].min(),
                residual/max(noise,.001),noise,apex,prior,(apex-prior)/max(noise,.001),*(qs-apex)/max(noise,.001),float(np.median(depth[mask])),
                float(xx.mean()/depth.shape[1]),float(yy.mean()/depth.shape[0]),threshold],np.float32)
            if np.isfinite(features).all():out.append(dict(mask=mask,features=features,height_raw_m=apex,noise_m=noise,pixels=len(xx),bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]))
    if out:
        heights=np.array([c['height_raw_m'] for c in out])
        for c in out:c['features']=np.r_[c['features'],(c['height_raw_m']-heights.max())/max(c['noise_m'],.001),np.mean(heights<=c['height_raw_m'])].astype(np.float32)
    return out

def score_candidates(candidates,bundle):
    if not candidates:return []
    x=np.stack([c['features'] for c in candidates])
    p=bundle['object'].predict_proba(x)[:,1];top=bundle['top'].predict_proba(x)[:,1]
    correction=bundle['correction'].predict(x)
    return [{**c,'object_probability':float(a),'top_probability':float(b),'correction_m':float(d*c['noise_m']),
        'height_m':float(c['height_raw_m']+d*c['noise_m'])} for c,a,b,d in zip(candidates,p,top,correction)]

def choose_highest(scored,policy):
    accepted=[]
    for c in scored:
        if c['object_probability']<policy['threshold'] or c['top_probability']<policy['top_threshold']:continue
        if abs(c['correction_m'])>3*c['noise_m']:continue
        accepted.append(c)
    return max(accepted,key=lambda c:c['height_m']) if accepted else None
