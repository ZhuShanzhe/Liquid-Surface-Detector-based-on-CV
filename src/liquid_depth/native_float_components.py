"""Same candidates and features; gather points within component bounds."""
import cv2,numpy as np
from liquid_depth.empty_geometry_v41 import height_calibration
from liquid_depth.liquid_height_metrics import world_points
from liquid_depth.small_float_peak_core import peak_roi
def dense_candidates(depth,prediction,bundle,liquid):
    cal=height_calibration(bundle);h=world_points(depth,cal)[...,2]-cal.bottom_z_m;roi=peak_roi(bundle,depth);p=prediction['semantic'][3];rel=prediction['reliability'][0];sigma=.0015+.0007*np.maximum(depth,0);out=[];seen=set()
    for threshold in (.25,.5,.75):
        mask=roi&(p>=threshold)&(rel>=.20);n,cc,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        for k in range(1,n):
            x0,y0,width,height,area=stats[k]
            if area<3:continue
            yy,xx=np.nonzero(cc[y0:y0+height,x0:x0+width]==k);yy+=y0;xx+=x0
            if np.ptp(xx)<1 or np.ptp(yy)<1:continue
            signature=(int(yy.min()),int(xx.min()),int(yy.max()),int(xx.max()),len(xx))
            if signature in seen:continue
            seen.add(signature);z=h[yy,xx];noise=float(np.median(sigma[yy,xx]));qs=np.percentile(z,[10,50,75,90,95,100]);apex=float(qs[3]);a=np.column_stack([np.ones(len(xx)),xx-np.median(xx),yy-np.median(yy)]);beta=np.linalg.lstsq(a,z,rcond=None)[0];residual=float(np.median(abs(z-a@beta)));w=int(np.ptp(xx)+1);ht=int(np.ptp(yy)+1)
            if residual>max(.015,3*noise) or np.sum(abs(z-apex)<=max(.015,3*noise))<3:continue
            prior=float(liquid) if liquid is not None else float(np.median(h[roi]));pv=p[yy,xx];rv=rel[yy,xx]
            features=np.array([np.log1p(len(xx)),w,ht,len(xx)/(w*ht),pv.mean(),pv.max(),np.quantile(pv,.1),rv.mean(),rv.min(),residual/max(noise,.001),noise,apex,prior,(apex-prior)/max(noise,.001),*(qs-apex)/max(noise,.001),float(np.median(depth[yy,xx])),float(xx.mean()/depth.shape[1]),float(yy.mean()/depth.shape[0]),threshold],np.float32)
            if np.isfinite(features).all():out.append(dict(mask=None,features=features,height_raw_m=apex,noise_m=noise,pixels=len(xx),bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]))
    if out:
        heights=np.array([c['height_raw_m'] for c in out])
        for c in out:c['features']=np.r_[c['features'],(c['height_raw_m']-heights.max())/max(c['noise_m'],.001),np.mean(heights<=c['height_raw_m'])].astype(np.float32)
    return out
