"""Fixed-camera bottom-texture relative refraction research; no evaluator inputs.

Not a production route. Requires unchanged camera/pose/bottom texture and a
visible, homogeneous-liquid bottom. No confidence or coverage rejection is used.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from .refraction_images_v23 import bottom_map, inverse_bottom
from .refraction_matching_v23 import normalized
from . import transparent_diagnostic_v25 as v25
from . import transparent_refinement_v26 as v26


def track_pair(anchor, current, cal, channel='left', bidirectional=True):
    """Track observed image features, never projected truth correspondences."""
    if anchor.shape != current.shape or anchor.ndim != 2:
        raise ValueError('incompatible_monochrome_images')
    a, b = normalized(anchor), normalized(current)
    h,w = a.shape
    mask=np.zeros_like(a);mask[28:h-28,28:w-28]=255
    pairs=[]
    directions=[(a,b,False),(b,a,True)] if bidirectional else [(a,b,False)]
    for first,second,reverse in directions:
        p=cv2.goodFeaturesToTrack(first,500,.015,9,mask=mask,blockSize=5)
        if p is None:continue
        q,st,_=cv2.calcOpticalFlowPyrLK(first,second,p,None,winSize=(31,31),maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,.0001))
        back,bst,_=cv2.calcOpticalFlowPyrLK(second,first,q,p.copy(),winSize=(31,31),maxLevel=3,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,.0001))
        p=p.reshape(-1,2);q=q.reshape(-1,2)
        fb=np.linalg.norm(back.reshape(-1,2)-p,axis=1)
        valid=(st.ravel()>0)&(bst.ravel()>0)&(fb<.7)&np.isfinite(q).all(1)
        valid&=(q[:,0]>18)&(q[:,0]<w-18)&(q[:,1]>18)&(q[:,1]<h-18)
        for x,y,e in zip(p[valid],q[valid],fb[valid]):
            pa=cv2.getRectSubPix(first,(15,15),tuple(x)).astype(float).ravel()
            pb=cv2.getRectSubPix(second,(15,15),tuple(y)).astype(float).ravel()
            pa-=pa.mean();pb-=pb.mean()
            corr=float(pa@pb/np.sqrt(max((pa@pa)*(pb@pb),1e-9)))
            if corr>.65:pairs.append((y if reverse else x,x if reverse else y,corr/(1+e)))
    if not pairs:return dict(anchor_q=np.empty((0,2)),current_q=np.empty((0,2)),weights=np.empty(0),regions=0,channel=channel)
    p=np.array([x[0] for x in pairs]);q=np.array([x[1] for x in pairs]);wt=np.array([x[2] for x in pairs])
    # One representative per direction and image cell; reverse observations are
    # correlated and are not counted as independent sensor samples in reporting.
    ids=[];cells=np.floor(p/20).astype(int)
    for cell in np.unique(cells,axis=0):
        hit=np.where((cells==cell).all(1))[0];ids.append(hit[np.argmax(wt[hit])])
    ids=np.asarray(ids);centre=np.array([cal['cx'],cal['cy']])
    return dict(anchor_q=(p[ids]-centre)/cal['fx'],current_q=(q[ids]-centre)/cal['fx'],
        weights=wt[ids],regions=len(ids),channel=channel)


def valid_pair(pair):
    return pair['regions']>=20 and np.isfinite(pair['anchor_q']).all() and np.isfinite(pair['current_q']).all()


def relative_residual(pair, anchor_height, height, index, cal):
    world=bottom_map(pair['anchor_q'],anchor_height,index,cal['bottom_m'])
    predicted=inverse_bottom(world,height,index,cal['bottom_m'])
    residual=(predicted-pair['current_q'])*cal['fx']
    # Per-pair translation nuisance. This is not camera-motion compensation:
    # changes in rotation/scale/pose cannot be identified from this alone.
    residual-=np.median(residual,axis=0)
    return residual


def relative_objective(groups, heights, target_height, index, cal):
    values=[]
    for anchor,pair in groups:
        r=relative_residual(pair,heights[anchor],target_height,index,cal)
        values.append(float(np.average((np.sqrt(1+(r/.04)**2)-1).sum(1),weights=pair['weights'])))
    return float(np.mean(values)) if values else float('inf')


def validate_heights(heights, cal):
    hs=np.asarray(heights,float)
    if len(hs)!=5 or not np.isfinite(hs).all() or hs[0]!=0 or np.any(np.diff(hs)<=0):
        raise ValueError('one_empty_plus_four_increasing_levels_required')
    if hs[-1]>=cal['container_max_height_m']:raise ValueError('known_level_exceeds_capacity')
    return hs


def fit_relative(pairs, heights, cal, condition, mode='multi'):
    v25.require_target(condition);hs=validate_heights(heights,cal)
    training=[(a,b,p) for a,b,p in pairs if valid_pair(p) and (mode!='empty' or a==0)]
    if len({b for a,b,p in training})<3:raise ValueError('insufficient_prior_tracks')
    def objective(index):
        return float(np.mean([relative_objective([(a,p)],hs,hs[b],index,cal) for a,b,p in training]))
    fit=minimize_scalar(objective,bounds=(1.2,1.6),method='bounded',options={'xatol':1e-8})
    if not fit.success:raise ValueError('relative_prior_optimizer_failed')
    return dict(index=float(fit.x),condition=condition,heights=hs.tolist(),mode=mode,
        prior_track_groups=len(training),research_only=True,production_enabled=False,
        shared_geometry=True,fixed_camera_required=True)


def infer_relative(groups, prior, cal, correction=None):
    groups=[(a,p) for a,p in groups if valid_pair(p) and (prior['mode']!='empty' or a==0)]
    if not groups:return dict(height_m=None,reasons=['no_usable_relative_tracks'],qualified=False)
    hs=np.asarray(prior['heights']);b=cal['bottom_m']
    fit=minimize_scalar(lambda h:relative_objective(groups,hs,h,prior['index'],cal),
        bounds=(-.05*b,.95*b),method='bounded',options={'xatol':max(1e-10,b*1e-9)})
    if not fit.success:return dict(height_m=None,reasons=['relative_height_optimizer_failed'],qualified=False)
    raw=float(fit.x);value=raw;reasons=[]
    if correction is not None:
        value=v26.bounded_height(raw,correction,cal,1.,.3,'constant')
    if value<0 or value>cal['container_max_height_m']:reasons.append('outside_container_capacity')
    if raw<hs[0] or raw>hs[-1]:reasons.append('outside_prior_span')
    if prior['index']<=1.205 or prior['index']>=1.595:reasons.append('effective_index_at_bound')
    return dict(height_m=float(value),uncorrected_height_m=raw,reasons=reasons,
        regions=sum(p['regions'] for a,p in groups),track_groups=len(groups),
        residual_objective=float(fit.fun),qualified=False,diagnostic_only=True,
        confidence_gate_enabled=False,coverage_gate_enabled=False)


def fit_absolute_empty(observations, heights, cal, condition, harmonics=2, ridge=.02):
    """Empty stereo pair fixes pixel bias; four nonzero levels fit effective IOR."""
    v25.require_target(condition);hs=validate_heights(heights,cal)
    if not all(v25.usable(o) for o in observations):raise ValueError('unusable_absolute_prior')
    empty=observations[0];count=2*harmonics
    def objective(p):
        r=v26.pixel_residual(0.,1.37,p[0],p[1:],empty,cal)
        return np.r_[(r*np.sqrt(empty['weights'])[:,None]/np.sqrt(empty['regions'])).ravel(),np.sqrt(ridge)*p[1:]]
    fit=least_squares(objective,np.zeros(count+1),bounds=(np.r_[-2.,np.full(count,-.15)],np.r_[2.,np.full(count,.15)]),
        loss='soft_l1',f_scale=.005,max_nfev=180)
    if not fit.success:raise ValueError('empty_bias_optimizer_failed')
    bias=float(fit.x[0]);coeff=fit.x[1:]
    def index_objective(index):
        values=[]
        for h,o in zip(hs[1:],observations[1:]):
            r=v26.pixel_residual(h,index,bias,coeff,o,cal)
            values.append(np.average((np.sqrt(1+(r/.03)**2)-1).sum(1),weights=o['weights']))
        return float(np.mean(values))
    idx=minimize_scalar(index_objective,bounds=(1.2,1.6),method='bounded',options={'xatol':1e-8})
    if not idx.success:raise ValueError('empty_index_optimizer_failed')
    prior=dict(index=float(idx.x),bias_px=bias,pixel_bias_coefficients=coeff.tolist(),condition=condition,
        correction='none',used_frames=list(range(5)),research_only=True,production_enabled=False)
    xp=np.array([v26.solve_height(o,prior,cal)['height_m'] for o in observations])
    if np.any(np.diff(xp)<=1e-9):raise ValueError('nonmonotonic_calibration')
    prior.update(x=xp.tolist(),y=hs.tolist(),prior_residual_m=(xp-hs).tolist())
    return prior


def empirical_correction(pairs, prior, cal):
    """Cross-prior reconstruction, excluding the target image as its own anchor."""
    xs=[];ys=[]
    for j,h in enumerate(prior['heights']):
        groups=[]
        for a,b,p in pairs:
            if b==j:groups.append((a,p))
            elif a==j:
                reverse=dict(p,anchor_q=p['current_q'],current_q=p['anchor_q'])
                groups.append((b,reverse))
        p=infer_relative(groups,prior,cal)
        if p.get('height_m') is not None:xs.append(p['height_m']);ys.append(h)
    order=np.argsort(xs);x=np.asarray(xs)[order];y=np.asarray(ys)[order]
    if len(x)<3 or np.any(np.diff(x)<=1e-9) or np.any(np.diff(y)<=0):raise ValueError('nonmonotonic_relative_calibration')
    return dict(x=x.tolist(),y=y.tolist())
