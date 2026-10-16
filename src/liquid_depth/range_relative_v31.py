"""Research-only native-image range registration and five-capture calibration.

No unknown liquid height, refractive index or extinction is an inference input.
Dense pixels in a registration are correlated, not independent measurements.
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, spline_filter, map_coordinates
from scipy.optimize import least_squares
from . import finite_relative_v30 as base


def _one(a, b, mask):
    support = base.eroded(mask, 3)
    yy, xx = np.where(support)
    if len(xx) < 60:
        return None
    centre = np.array([xx.mean(), yy.mean()])
    # The core remains inside measured bottom support over the entire scale
    # search. Filling outside the mask only protects filtering boundaries.
    use = (abs(xx-centre[0]) < .30*np.ptp(xx)) & (abs(yy-centre[1]) < .30*np.ptp(yy))
    xx, yy = xx[use], yy[use]
    stride = max(1, int(np.ceil(len(xx)/650)))
    xx, yy = xx[::stride].astype(float), yy[::stride].astype(float)
    if len(xx) < 30:
        return None
    nearest = distance_transform_edt(mask == 0, return_distances=False, return_indices=True)
    # Suppress fixed high-frequency projector speckles that do not follow
    # material-bottom motion. This is filtering, not invented resolution.
    aa = cv2.GaussianBlur(a[tuple(nearest)].astype(float)/255, (0,0), 3.)
    bb = cv2.GaussianBlur(b[tuple(nearest)].astype(float)/255, (0,0), 3.)
    ref = aa[yy.astype(int), xx.astype(int)]
    if ref.std() < .003:
        return None
    target = spline_filter(bb, order=3)
    p = np.c_[xx, yy]; delta = p-centre; r = ref-ref.mean()

    def values(v):
        q = centre+delta*np.exp(v[0])+v[1:]
        val = map_coordinates(target, [q[:,1],q[:,0]], order=3, prefilter=False, mode='nearest')
        return q, val-val.mean()

    def residual(v):
        _, z = values(v)
        # Negative photometric gain is not a valid texture correspondence.
        gain = max(0., float(z@r/max(z@z, 1e-12)))
        return gain*z-r

    starts = [np.array([s,0.,0.]) for s in np.linspace(-.48,.48,25)]
    starts.sort(key=lambda s: float(np.mean(residual(s)**2)))
    results = []
    for start in starts[:2]:
        fit = least_squares(residual, start, bounds=([-.50,-5.,-5.],[.50,5.,5.]),
            loss='soft_l1', f_scale=.012, max_nfev=65, diff_step=1e-4,
            xtol=1e-9, ftol=1e-9, gtol=1e-9)
        if np.isfinite(fit.x).all():
            results.append(fit)
    if not results:
        return None
    fit = min(results, key=lambda f: np.mean(residual(f.x)**2))
    q,z = values(fit.x)
    qi = np.round(q).astype(int)
    valid = (qi[:,0]>=0)&(qi[:,0]<mask.shape[1])&(qi[:,1]>=0)&(qi[:,1]<mask.shape[0])
    idx = np.where(valid)[0]; valid[idx] &= support[qi[idx,1],qi[idx,0]]
    corr = float(z@r/max(np.linalg.norm(z)*np.linalg.norm(r),1e-12))
    return p[valid], q[valid], dict(correlation=corr,
        photometric_rms=float(np.sqrt(np.mean(residual(fit.x)**2))),log_scale=float(fit.x[0]))


def wide_pair(a,b,cal,mask,channel):
    first = _one(a,b,mask)
    second = _one(b,a,mask)
    if first is None and second is None:
        return base.package([],[],cal,channel)
    parts=[];quality=[]
    if first is not None:
        p,q,e=first;parts.append((p,q));quality.append(e)
    if second is not None:
        p,q,e=second;parts.append((q,p));quality.append(e)
    # Prefer the more observable direction; unequal-quality symmetric fits
    # should not average a reliable transform with a noise fit.
    order=np.argsort([e['correlation'] for e in quality]);j=int(order[-1])
    p,q=parts[j];e=quality[j]
    return base.package(p,q,cal,channel,dict(e,method='native_wide_similarity',correlated_dense_samples=True))


def quality(pair):
    return max(.001, max(0.,float(pair.get('correlation',1.)))**8)


def _weighted_median(values, weights):
    order=np.argsort(values);x=np.asarray(values)[order];w=np.asarray(weights)[order]
    return float(x[np.searchsorted(np.cumsum(w),w.sum()/2)])


def fit_curve(pairs,heights,cal,condition):
    base.old.v25.require_target(condition)
    hs=base.old.validate_heights(heights,cal)
    edges=[(a,b,base.observed_log_scale(p),quality(p)) for a,b,p in pairs if base.valid(p)]
    edges=[e for e in edges if e[2] is not None]
    active={0}
    for _ in range(5):
        for a,b,s,w in edges:
            if a in active or b in active:active.update((a,b))
    nodes=sorted(active)
    if len(nodes)<3:raise ValueError('insufficient_connected_known_heights')
    hh=hs[nodes];pos={a:i for i,a in enumerate(nodes)}
    edges=[e for e in edges if e[0] in active and e[1] in active]
    btm=cal['bottom_m'];lower=.005/btm*np.diff(hh);upper=.7/btm*np.diff(hh)/(1-.7*hh[1:]/btm)
    initial=np.diff(-np.log1p(-.25*hh/btm))

    def objective(increments):
        latent=np.r_[0.,np.cumsum(increments)]
        return np.array([(latent[pos[b]]-latent[pos[a]]-s)*np.sqrt(w) for a,b,s,w in edges])

    result=least_squares(objective,np.clip(initial,lower*1.001,upper*.999),bounds=(lower,upper),
        loss='soft_l1',f_scale=.0005,max_nfev=150,xtol=1e-11,ftol=1e-11,gtol=1e-11)
    if not result.success:raise ValueError('range_curve_calibration_failed')
    latent=np.r_[0.,np.cumsum(result.x)]
    return dict(heights=hs.tolist(),nodes=nodes,node_heights=hh.tolist(),latent=latent.tolist(),
        condition=condition,research_only=True,production_enabled=False,physical_index_identified=False,
        prior_track_groups=len(edges),calibration_residual=float(np.sqrt(np.mean(objective(result.x)**2))))


def _inverse(x,xs,ys):
    # Linear endpoint extrapolation, never clamp to capacity or test range.
    j=int(np.clip(np.searchsorted(xs,x)-1,0,len(xs)-2))
    return float(ys[j]+(x-xs[j])*(ys[j+1]-ys[j])/(xs[j+1]-xs[j]))


def infer_curve(groups,prior,cal,best=False):
    nodes=prior['nodes'];latent=prior['latent'];mapping=dict(zip(nodes,latent));xs=[];ws=[]
    for a,p in groups:
        if a not in mapping or not base.valid(p):continue
        s=base.observed_log_scale(p)
        if s is not None:xs.append(mapping[a]+s);ws.append(quality(p))
    if not xs:return dict(height_m=None,reasons=['no_usable_finite_bottom_observations'],qualified=False)
    if best:
        order=np.argsort(ws)[-2:];x=_weighted_median(np.asarray(xs)[order],np.asarray(ws)[order])
    else:x=_weighted_median(xs,ws)
    h=_inverse(x,latent,prior['node_heights']);reasons=[]
    if x<latent[0] or x>latent[-1]:reasons.append('outside_known_prior_span')
    if h<0 or h>cal['container_max_height_m']:reasons.append('outside_container_capacity')
    return dict(height_m=h,reasons=reasons,qualified=False,diagnostic_only=True,
        confidence_gate_enabled=False,coverage_gate_enabled=False,physical_index_identified=False,
        track_groups=len(xs),latent_dispersion=float(np.std(xs)),latent_observation=x)
