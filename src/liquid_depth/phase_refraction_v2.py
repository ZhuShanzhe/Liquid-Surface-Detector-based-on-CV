"""Five-prior uniform-liquid inversion for raw iToF / projector-camera phases.

No stereo image adapter, test liquid label, event mask or test refractive index is
accepted by this runtime. This is a bounded physical-family simulator validation,
not a claim that arbitrary vendor depth-only SDKs expose the required raw data.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

C=299792458.
SENSORS=('tof','structured_light')
SCENES=('homogeneous_transparent','homogeneous_translucent')


def decode(correlation):
    a=np.asarray(correlation,dtype=float)
    if a.ndim!=4 or a.shape[1]!=4 or not np.isfinite(a).all():
        raise ValueError('finite_frequency_4phase_height_width_correlations_required')
    return (a[:,0]-a[:,2])+1j*(a[:,3]-a[:,1])


def bottom_map(q,h,n,b):
    a=np.sqrt(n*n+(n*n-1)*np.sum(q*q,axis=-1))
    return (b-h)*q+h*q/a[...,None]


def projector_delta(q,h,n,b,baseline,fx):
    """Snell camera ray -> bottom hit -> inverse projector ray, flat interface."""
    target=bottom_map(q,h,n,b)-np.array([baseline,0.])
    p=target/b
    for _ in range(7):
        a=np.sqrt(n*n+(n*n-1)*np.sum(p*p,axis=-1))
        residual=(b-h)*p+h*p/a[...,None]-target
        diagonal=b-h+h/a
        factor=h*(n*n-1)/(a**3)
        jxx=diagonal-factor*p[...,0]**2;jyy=diagonal-factor*p[...,1]**2
        jxy=-factor*p[...,0]*p[...,1];det=jxx*jyy-jxy*jxy
        p-=np.stack(((jyy*residual[...,0]-jxy*residual[...,1])/det,
                    (jxx*residual[...,1]-jxy*residual[...,0])/det),axis=-1)
    return fx*(p[...,0]-(q[...,0]-baseline/b))


def relative_phasor(sensor,q,h,parameters,profile):
    n,rho,ext,scatter,offset=parameters
    if sensor=='tof':
        k=4*np.pi*np.asarray(profile['frequencies_hz'])/C
        cosa=(1+np.sum(q*q,axis=-1))**-.5
        cost=np.sqrt(1-(1-cosa*cosa)/(n*n))
        surface=-h/cosa;bottom=h*(n/cost-1/cosa)
        ab=np.exp(-2*ext*h/cost);ass=rho*np.ones_like(ab)
        ph=ab[None]*np.exp(1j*k[:,None]*bottom[None])+ass[None]*np.exp(1j*k[:,None]*surface[None])
        for t in (.125,.375,.625,.875):
            amp=scatter*(1-np.exp(-2*ext*h/cost))/4
            delta=-h/cosa+n*h*t/cost
            ph+=amp[None]*np.exp(1j*k[:,None]*delta[None])
        ph*=np.exp(1j*k[:,None]*offset)
    else:
        b=profile['bottom_m'];fx=profile['fx_px'];baseline=profile['baseline_m']
        k=2*np.pi/np.asarray(profile['fringe_periods_px'])
        delta=projector_delta(q,h,n,b,baseline,fx)
        surface=-fx*baseline/(b-h)+fx*baseline/b
        ab=np.exp(-2*ext*h)*np.ones(len(q))
        ph=ab[None]*np.exp(1j*k[:,None]*delta[None])+rho*np.exp(1j*k[:,None]*surface)
        # A broad subsurface component has low contrast; residual blur is left as model mismatch.
        ph+=scatter*(1-np.exp(-2*ext*h))*np.exp(1j*k[:,None]*(delta[None]+surface)/2)
        ph*=np.exp(1j*k[:,None]*offset)
    return ph


def model_feature(sensor,q,h,parameters,profile):
    z=relative_phasor(sensor,q,h,parameters,profile)
    return np.mean(z/np.maximum(abs(z),1e-12),axis=1)


@dataclass
class PhasePriorMeter:
    sensor:str
    scene:str
    profile:dict
    q:np.ndarray
    dry:np.ndarray
    roi:np.ndarray
    parameters:np.ndarray
    known:np.ndarray
    calibration_residual:float

    @classmethod
    def fit(cls,sensor,scene,profile,ray_coordinates,roi,prior_correlations,known_heights_m):
        if sensor not in SENSORS:raise ValueError('only_tof_or_structured_light_supported')
        if scene not in SCENES:raise ValueError('uniform_transparent_or_mild_translucent_required')
        known=np.asarray(known_heights_m,float)
        capacity=float(profile['container_max_height_m']);b=float(profile['bottom_m'])
        if known.shape!=(5,) or not np.isfinite(known).all() or known[0]!=0 or np.any(known[1:]<=0) or len(np.unique(known))!=5 or np.any(np.diff(known)<=0):
            raise ValueError('one_empty_plus_four_increasing_distinct_known_heights_required')
        if capacity<=0 or b<=capacity or known[-1]>capacity:raise ValueError('calibration_outside_container')
        if len(prior_correlations)!=5:raise ValueError('five_capture_groups_required')
        decoded=[decode(c) for c in prior_correlations];shape=decoded[0].shape
        if any(z.shape!=shape for z in decoded):raise ValueError('prior_capture_shape_changed')
        expected=3 if sensor=='tof' else 2
        if shape[0]!=expected:raise ValueError('wrong_modulation_channel_count')
        field='frequencies_hz' if sensor=='tof' else 'fringe_periods_px'
        if len(profile[field])!=expected or np.any(np.asarray(profile[field])<=0):raise ValueError('invalid_sensor_modulation_profile')
        mask=np.asarray(roi,bool)
        q=np.asarray(ray_coordinates,float)
        if mask.shape!=shape[1:] or q.shape!=(*mask.shape,2):raise ValueError('aligned_empty_roi_and_camera_rays_required')
        # Empty/operator ROI only. Common valid prior modulation, not a true-liquid pixel mask.
        mask=mask&np.all(np.stack([abs(z)>8 for z in decoded]),axis=(0,1))
        if mask.sum()<64:raise ValueError('insufficient_prior_modulation_support')
        # Deterministic spatial sampling reduces fit cost without inventing pixels.
        idx=np.flatnonzero(mask)[::max(1,int(mask.sum()//128))][:160]
        fitroi=np.zeros(mask.size,bool);fitroi[idx]=True;fitroi=fitroi.reshape(mask.shape)
        rays=q[fitroi];dry=decoded[0]
        def feature(z):
            good=fitroi&np.all(abs(z)>8,axis=0)
            if good.sum()<64:raise ValueError('insufficient_current_modulation_support')
            ratio=z[:,fitroi]/np.maximum(abs(z[:,fitroi]),1e-12)*np.conj(dry[:,fitroi]/np.maximum(abs(dry[:,fitroi]),1e-12))
            return np.mean(ratio,axis=1)
        observations=np.stack([feature(z) for z in decoded])
        def residual(v):
            value=np.stack([model_feature(sensor,rays,h,v,profile) for h in known])
            diff=value-observations
            return np.r_[diff.real.ravel(),diff.imag.ravel()]
        off=.015 if sensor=='tof' else .3
        bounds=([1.15,0.,0.,0.,-off],[1.6,.4,.6,1.,off])
        best=None
        for n in (1.25,1.36,1.49):
            fitted=least_squares(residual,[n,.06,.04 if scene.endswith('translucent') else .005,.12,0.],
                bounds=bounds,loss='soft_l1',f_scale=.01,max_nfev=180,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            cost=float(np.mean(residual(fitted.x)**2))
            if best is None or cost<best[0]:best=(cost,fitted)
        return cls(sensor,scene,dict(profile),rays,dry,fitroi,best[1].x,known,float(np.sqrt(best[0])))

    def measure(self,correlations):
        z=decode(correlations)
        if z.shape!=self.dry.shape:raise ValueError('capture_shape_changed_recalibrate')
        valid=np.all(abs(z[:,self.roi])>8,axis=0)
        support=int(valid.sum())
        if support<64:return self._result(None,{'reasons':['insufficient_current_modulation_support'],'support':support})
        ratio=z[:,self.roi]/np.maximum(abs(z[:,self.roi]),1e-12)*np.conj(self.dry[:,self.roi]/np.maximum(abs(self.dry[:,self.roi]),1e-12))
        observed=np.mean(ratio[:,valid],axis=1);q=self.q[valid]
        def score(h):
            diff=model_feature(self.sensor,q,h,self.parameters,self.profile)-observed
            return float(np.mean(abs(diff)**2))
        # Search beyond nominal range so endpoints cannot become forced zero/capacity.
        # Numeric output is not rejected by fit error or confidence thresholds.
        cap=self.profile['container_max_height_m'];margin=max(.05,.02*cap)
        upper=min(cap+margin,self.profile['bottom_m']-.02)
        grid=np.linspace(-margin,upper,151);costs=np.array([score(h) for h in grid]);i=int(costs.argmin())
        lo=grid[max(0,i-1)];hi=grid[min(len(grid)-1,i+1)]
        best=minimize_scalar(score,bounds=(lo,hi),method='bounded',options={'xatol':1e-7})
        h=float(best.x);step=max(1e-4,cap*1e-4)
        a=model_feature(self.sensor,q,h+step,self.parameters,self.profile)
        b=model_feature(self.sensor,q,h-step,self.parameters,self.profile)
        sensitivity=float(np.linalg.norm(a-b)/(2*step))
        ordered=np.sort(costs);risk=float(np.sqrt(best.fun)/max(sensitivity,1e-12))
        return self._result(h,dict(support=support,model_residual=float(np.sqrt(best.fun)),
            calibration_residual=self.calibration_residual,local_sensitivity_per_m=sensitivity,
            residual_to_sensitivity_m=risk,coarse_second_cost=float(ordered[1]),
            extrapolated=bool(h<self.known[0] or h>self.known[-1]),
            outside_container_unclipped=bool(h<0 or h>cap),parameters=self.parameters.tolist(),
            confidence_used_to_reject=False,optimizer_search_bounds_m=[-margin,upper],
            constraint_boundary_reached=bool(h<grid[0]+step or h>grid[-1]-step)))

    def _result(self,h,diagnostics):
        return dict(schema_version=2,scene=self.scene,sensor_family=self.sensor,route='five_prior_raw_phase_'+self.sensor,
            liquid_depth_m=h,numeric_output=h is not None,diagnostics=diagnostics,
            field_metrology_qualified=False,visual_module_executed=False,liquid_visual_guidance_used=False,
            highest_visual_guidance_used=False,visual_guidance_used=False,liquid_override_used=False,
            reasons=[] if h is not None else diagnostics['reasons'])
