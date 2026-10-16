"""Small finite-bottom relative tracking and direct registration, no truth inputs."""
import cv2
import numpy as np
from scipy.optimize import least_squares,minimize_scalar
from scipy.ndimage import spline_filter,map_coordinates,distance_transform_edt
from . import transparent_relative_v29 as old


def eroded(mask,radius):
    return cv2.erode((mask>0).astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0

def package(p,q,cal,channel,extra=None):
    p=np.asarray(p,float).reshape(-1,2);q=np.asarray(q,float).reshape(-1,2)
    c=np.array([cal['cx'],cal['cy']])
    result=dict(anchor_q=(p-c)/cal['fx'],current_q=(q-c)/cal['fx'],weights=np.ones(len(p)),regions=len(p),channel=channel)
    if extra:result.update(extra)
    return result


def baseline_pair(a,b,cal,mask,channel):
    # Crop to the fixed empty-bottom ROI before calling unchanged V29. No rim or
    # external background is admitted as liquid evidence. Preserve missing data.
    ys,xs=np.where(mask>0)
    if not len(xs):return package([],[],cal,channel)
    x0,x1=int(xs.min()),int(xs.max())+1;y0,y1=int(ys.min()),int(ys.max())+1
    local=dict(cal,cx=cal['cx']-x0,cy=cal['cy']-y0)
    return old.track_pair(a[y0:y1,x0:x1],b[y0:y1,x0:x1],local,channel)


def small_lk(a,b,cal,mask,channel):
    support=eroded(mask,5);roi=support.astype(np.uint8)*255
    p=cv2.goodFeaturesToTrack(a,400,.01,2,mask=roi,blockSize=3)
    if p is None:return package([],[],cal,channel)
    q,st,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,winSize=(9,9),maxLevel=0,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,80,.0001))
    back,bst,_=cv2.calcOpticalFlowPyrLK(b,a,q,p.copy(),winSize=(9,9),maxLevel=0,flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,80,.0001))
    p=p.reshape(-1,2);q=q.reshape(-1,2);fb=np.linalg.norm(back.reshape(-1,2)-p,axis=1)
    valid=(st.ravel()>0)&(bst.ravel()>0)&(fb<.7)&np.isfinite(q).all(1)
    # The second patch must also stay inside the measured bottom ROI.
    for i,t in enumerate(q):
        x,y=np.round(t).astype(int)
        valid[i]&=0<=x<mask.shape[1] and 0<=y<mask.shape[0] and support[y,x]
    return package(p[valid],q[valid],cal,channel,dict(method='native_small_LK'))


def direct_one(a,b,mask):
    support=eroded(mask,3);y,x=np.where(support)
    if len(x)<30:return None
    # A native-pixel dense similarity fit; no super-resolution, no hidden signal.
    stride=max(1,int(np.sqrt(len(x)/1200)));x=x[::stride].astype(float);y=y[::stride].astype(float)
    centre=np.array([np.mean(x),np.mean(y)]);dx=x-centre[0];dy=y-centre[1]
    ref=a[y.astype(int),x.astype(int)].astype(float)/255
    if np.std(ref)<.006:return None
    # Spline prefilter must not leak external background/rim values into the fit.
    # Extend only measured bottom pixels beyond the ROI, never external pixels.
    indices=distance_transform_edt(mask==0,return_distances=False,return_indices=True)
    filled=b[tuple(indices)].astype(float)/255
    target=spline_filter(filled,order=3)
    def residual(params):
        scale=np.exp(params[0]);xx=centre[0]+dx*scale+params[1];yy=centre[1]+dy*scale+params[2]
        val=map_coordinates(target,[yy,xx],order=3,prefilter=False,mode='nearest')
        # Affine radiometric nuisance eliminated analytically at each iteration.
        v=val-val.mean();r=ref-ref.mean();gain=float(v@r/max(v@v,1e-10))
        return (gain*v-r)
    fit=least_squares(residual,[0.,0.,0.],bounds=([-.12,-1.5,-1.5],[.12,1.5,1.5]),
        loss='soft_l1',f_scale=.012,max_nfev=80,diff_step=1e-4,xtol=1e-10,ftol=1e-10,gtol=1e-10)
    if not fit.success or not np.isfinite(fit.x).all():return None
    scale=np.exp(fit.x[0]);p=np.c_[x,y];q=centre+(p-centre)*scale+fit.x[1:]
    # Disallow support extrapolation into a rim/background. Tight support has
    # radius 3 and bounds 1.5 px; retain only inside-ROI correspondences.
    qi=np.round(q).astype(int);valid=(qi[:,0]>=0)&(qi[:,0]<mask.shape[1])&(qi[:,1]>=0)&(qi[:,1]<mask.shape[0])
    idx=np.where(valid)[0];valid[idx]&=support[qi[idx,1],qi[idx,0]]
    return p[valid],q[valid],dict(log_scale=float(fit.x[0]),photometric_rms=float(np.sqrt(np.mean(residual(fit.x)**2))))


def direct_pair(a,b,cal,mask,channel,symmetric=True):
    first=direct_one(a,b,mask)
    if first is None:return package([],[],cal,channel)
    p,q,extra=first
    if symmetric:
        second=direct_one(b,a,mask)
        if second is not None:
            r,s,_=second;p=np.r_[p,s];q=np.r_[q,r]
    return package(p,q,cal,channel,dict(extra,method='native_direct_similarity',correlated_dense_samples=True))


def valid(pair):
    if pair['regions']<6 or not np.isfinite(pair['anchor_q']).all() or not np.isfinite(pair['current_q']).all():return False
    cov=np.cov(pair['anchor_q'].T)
    return bool(np.linalg.eigvalsh(cov).min()>1e-8)


def fit(pairs,heights,cal,condition,correction=False):
    old.v25.require_target(condition);hs=old.validate_heights(heights,cal)
    ps=[(a,b,p) for a,b,p in pairs if valid(p)]
    if len({b for a,b,p in ps})<3:raise ValueError('insufficient_finite_prior_observations')
    def objective(n):
        return float(np.mean([old.relative_objective([(a,p)],hs,hs[b],n,cal) for a,b,p in ps]))
    result=minimize_scalar(objective,bounds=(1.2,1.6),method='bounded',options={'xatol':1e-8})
    if not result.success:raise ValueError('finite_prior_optimizer_failed')
    prior=dict(index=float(result.x),heights=hs.tolist(),condition=condition,mode='multi',prior_track_groups=len(ps),
        research_only=True,production_enabled=False)
    if correction:
        xs=[];ys=[]
        for j,h in enumerate(hs):
            groups=[]
            for a,b,p in ps:
                if b==j:groups.append((a,p))
                elif a==j:groups.append((b,dict(p,anchor_q=p['current_q'],current_q=p['anchor_q'])))
            pred=infer(groups,prior,cal)
            if pred['height_m'] is not None:xs.append(pred['height_m']);ys.append(h)
        order=np.argsort(xs);x=np.asarray(xs)[order];y=np.asarray(ys)[order]
        if len(x)<3 or np.any(np.diff(x)<=1e-9) or np.any(np.diff(y)<=0):raise ValueError('nonmonotonic_finite_calibration')
        prior['empirical_correction']=dict(x=x.tolist(),y=y.tolist())
    return prior


def infer(groups,prior,cal):
    groups=[(a,p) for a,p in groups if valid(p)]
    if not groups:return dict(height_m=None,reasons=['no_usable_finite_bottom_observations'],qualified=False)
    # Keep the same physically broad search interval as V29, not [0,C] clipping.
    b=cal['bottom_m'];hs=np.asarray(prior['heights'])
    result=minimize_scalar(lambda h:old.relative_objective(groups,hs,h,prior['index'],cal),bounds=(-.05*b,.95*b),
        method='bounded',options={'xatol':max(1e-10,b*1e-9)})
    if not result.success:return dict(height_m=None,reasons=['finite_height_optimizer_failed'],qualified=False)
    raw=float(result.x);value=raw;reasons=[]
    if 'empirical_correction' in prior:value=old.v26.bounded_height(raw,prior['empirical_correction'],cal,1.,.3,'constant')
    if value<0 or value>cal['container_max_height_m']:reasons.append('outside_container_capacity')
    if raw<hs[0] or raw>hs[-1]:reasons.append('outside_prior_span')
    if prior['index']<=1.205 or prior['index']>=1.595:reasons.append('effective_index_at_bound')
    return dict(height_m=float(value),uncorrected_height_m=raw,reasons=reasons,qualified=False,
        diagnostic_only=True,confidence_gate_enabled=False,coverage_gate_enabled=False,
        regions=sum(p['regions'] for a,p in groups),track_groups=len(groups),objective=float(result.fun))


def observed_log_scale(pair):
    p=pair['anchor_q']-np.mean(pair['anchor_q'],axis=0)
    q=pair['current_q']-np.mean(pair['current_q'],axis=0)
    scale=float(np.sum(p*q)/max(np.sum(p*p),1e-12))
    return float(np.log(scale)) if scale>0 else None


def fit_response(pairs,heights,cal,condition,alpha_floor=.005):
    """Calibrate effective observed scale response, NOT the liquid's true IOR.

    A static projector component and pixel integration can attenuate apparent
    texture motion beyond physical-IOR bounds. One monotonic response parameter
    is fit to all five known levels; it is not selected with test liquid height.
    """
    old.v25.require_target(condition);hs=old.validate_heights(heights,cal)
    ps=[(a,b,observed_log_scale(p)) for a,b,p in pairs if valid(p)]
    ps=[(a,b,s) for a,b,s in ps if s is not None]
    if len({b for a,b,s in ps})<3:raise ValueError('insufficient_finite_prior_observations')
    btm=cal['bottom_m']
    def objective(alpha):
        k=alpha/btm
        r=np.array([np.log1p(-k*hs[a])-np.log1p(-k*hs[b])-s for a,b,s in ps])
        return float(np.mean(np.sqrt(1+(r/.0001)**2)-1))
    if not .005<=alpha_floor<=.2:raise ValueError('invalid_response_floor')
    opt=minimize_scalar(objective,bounds=(alpha_floor,.7),method='bounded',options={'xatol':1e-9})
    if not opt.success:raise ValueError('response_prior_optimizer_failed')
    return dict(heights=hs.tolist(),condition=condition,effective_alpha=float(opt.x),response_k=float(opt.x/btm),
        mode='observed_scale_response',prior_track_groups=len(ps),physical_index_identified=False,alpha_floor=alpha_floor,
        research_only=True,production_enabled=False)


def infer_response(groups,prior,cal):
    samples=[];k=prior['response_k'];hs=prior['heights']
    for a,p in groups:
        if not valid(p):continue
        s=observed_log_scale(p)
        if s is not None:samples.append((1-(1-k*hs[a])*np.exp(-s))/k)
    if not samples:return dict(height_m=None,reasons=['no_usable_finite_bottom_observations'],qualified=False)
    h=float(np.median(samples));reasons=[]
    if h<0 or h>cal['container_max_height_m']:reasons.append('outside_container_capacity')
    if prior['effective_alpha']<prior.get('alpha_floor',.005)+.001 or prior['effective_alpha']>.699:reasons.append('effective_response_at_bound')
    return dict(height_m=h,reasons=reasons,qualified=False,diagnostic_only=True,
        confidence_gate_enabled=False,coverage_gate_enabled=False,track_groups=len(samples),
        height_dispersion_m=float(np.std(samples)),physical_index_identified=False)
