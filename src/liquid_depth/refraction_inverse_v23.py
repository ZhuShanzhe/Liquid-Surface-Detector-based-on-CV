"""Two-dimensional, finite-baseline material calibration and liquid inversion."""
import numpy as np
from scipy.optimize import least_squares,minimize_scalar
from .refraction_images_v23 import bottom_map


def residuals(height,index,bias_px,obs,cal):
    qr=np.asarray(obs['right_q']).copy();qr[:,0]+=bias_px/cal['fx']
    l=bottom_map(obs['left_q'],height,index,cal['bottom_m'])
    r=bottom_map(qr,height,index,cal['bottom_m'],(cal['baseline_m'],0))
    # Approximate pixel units. Not a direct physical uncertainty guarantee.
    return (l-r)*cal['fx']/cal['bottom_m']


def fit_prior(observations,heights,cal):
    heights=np.asarray(heights,float)
    if len(heights)<3 or len(observations)!=len(heights):raise ValueError('At least three known levels required')
    if np.ptp(heights)<max(.03,.15*max(heights)):raise ValueError('Known levels have insufficient span')
    if any(o['regions']<20 for o in observations):raise ValueError('Insufficient matched prior regions')
    def fit(ids):
        def fun(p):
            return np.concatenate([(residuals(h,p[0],p[1],o,cal)*np.sqrt(np.maximum(o['weights'],.02))[:,None]
                /np.sqrt(len(o['weights']))).ravel() for i,(o,h) in enumerate(zip(observations,heights)) if i in ids])
        result=least_squares(fun,[1.333,0],bounds=([1.2,-2],[1.6,2]),loss='soft_l1',f_scale=.025,max_nfev=100)
        if not result.success:raise ValueError('Material optimizer failed')
        return result.x
    n,b=fit(set(range(len(heights))))
    leave=np.array([fit(set(range(len(heights)))-{i}) for i in range(len(heights))])
    sd=np.sqrt((len(heights)-1)/len(heights)*np.sum((leave-leave.mean(0))**2,axis=0))
    if n<=1.205 or n>=1.595 or sd[0]>.04:raise ValueError('Material estimate unstable')
    for i,(obs,h) in enumerate(zip(observations,heights)):
        prediction=infer(obs,dict(index=float(leave[i,0]),bias_px=float(leave[i,1]),index_sd=0.,bias_sd=0.),cal)
        if prediction['height_m'] is None or abs(prediction['height_m']-h)>max(.005,.05*h):
            raise ValueError('Held-out prior check failed')
    return dict(index=float(n),bias_px=float(b),index_sd=float(sd[0]),bias_sd=float(sd[1]),
                calibration_frames=len(heights),research_only=True,production_enabled=False)


def infer(obs,prior,cal):
    count=obs['regions']
    out=dict(height_m=None,observable=False,reasons=[],regions=count,
             visibility=obs.get('visibility',0.),uncertainty_m=None)
    if count<20:
        out['reasons']=['insufficient_matched_regions'];return out
    n,b=prior['index'],prior['bias_px']
    weights=np.maximum(obs['weights'],.02)
    def objective(h):
        e=residuals(h,n,b,obs,cal)
        return float(np.average(np.sqrt(1+(e/.25)**2).sum(1)-2,weights=weights))
    result=minimize_scalar(objective,bounds=(.001,.85*cal['bottom_m']),method='bounded')
    h=float(result.x);e=residuals(h,n,b,obs,cal)
    norms=np.linalg.norm(e,axis=1)
    inliers=norms<max(.25,3*float(np.median(norms)))
    dh=max(1e-6,h*1e-4)
    jac=(residuals(h+dh,n,b,obs,cal)-residuals(h-dh,n,b,obs,cal))/(2*dh)
    denom=np.sum(jac*jac,axis=1)
    corrections=np.sum(e*jac,axis=1)/np.maximum(denom,1e-12)
    # Spatial block effects are not treated as independent pixels.
    blocks=np.floor(obs['left_pixels']/64).astype(int)
    block_c=[]
    for cell in np.unique(blocks,axis=0):
        ids=(blocks==cell).all(1)&inliers
        if ids.any():block_c.append(float(np.median(corrections[ids])))
    scatter=1.4826*float(np.median(abs(np.asarray(block_c)-np.median(block_c)))) if block_c else np.inf
    # Retain a systematic matching floor, even if every block agrees.
    sensitivity=np.sqrt(np.median(denom))
    uncertainty=np.sqrt((scatter/max(np.sqrt(len(block_c)),1))**2+(.015/max(sensitivity,1e-8))**2)
    dn=1e-4
    jn=(residuals(h,n+dn,b,obs,cal)-residuals(h,n-dn,b,obs,cal))/(2*dn)
    ni=abs(np.median(np.sum(jn*jac,axis=1)/np.maximum(denom,1e-12)))
    bi=cal['bottom_m']/cal['fx']/max(np.median(abs(jac[:,0]))*cal['bottom_m']/cal['fx'],1e-8)
    uncertainty=np.sqrt(uncertainty**2+(ni*prior.get('index_sd',0))**2+(bi*prior.get('bias_sd',0))**2)
    reason=[]
    if not result.success or h>=.849*cal['bottom_m']:reason.append('geometry_or_optimizer_limit')
    if inliers.mean()<.60:reason.append('inconsistent_matches')
    if np.median(norms)>.8:reason.append('optical_model_residual')
    out.update(height_m=h,observable=not reason,reasons=reason,uncertainty_m=float(uncertainty),
        median_residual_px=float(np.median(norms)),inlier_fraction=float(inliers.mean()),
        block_count=len(block_c),research_only=True,hardware_qualified=False)
    return out


def raw_height(obs,cal):
    if obs['regions']<20:return None
    disparity=(obs['left_q'][:,0]-obs['right_q'][:,0])*cal['fx']
    good=np.isfinite(disparity)&(disparity>.1)
    if good.sum()<20:return None
    return float(cal['bottom_m']-np.median(cal['fx']*cal['baseline_m']/disparity[good]))
