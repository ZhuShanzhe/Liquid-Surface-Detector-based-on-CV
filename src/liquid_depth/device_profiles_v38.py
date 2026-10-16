"""Source-backed geometry; explicit, unqualified noise sensitivity assumptions.

This is NOT a vendor digital twin. No profile promises liquid-level accuracy.
Pure NumPy so Blender can import it without torch/cv2 package initialization.
"""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class Profile:
    name: str
    sensor: str
    width: int
    height: int
    hfov_deg: float
    vfov_spec_deg: float
    fps: float
    min_z: float
    max_z: float
    wavelength_nm: int
    baseline_m: float
    source: str
    @property
    def fx(self): return self.width/(2*np.tan(np.deg2rad(self.hfov_deg/2)))
    def manifest(self):
        return dict(asdict(self), fx=self.fx, projection='rectified square pixels from horizontal FOV; vertical FOV approximate',
                    hardware_qualified=False, noise_parameters='assumptions; nominal/adverse are sensitivity brackets, not probabilities')

PROFILES=(
    Profile('blaze101_long','tof',640,480,67.,51.,20.,.3,10.,940,0.,
            'https://docs.baslerweb.com/blaze-101'),
    Profile('gemini2xl_dense','active_stereo',1280,800,91.,66.,10.,.4,10.,850,.1,
            'https://orbbec.com/wp-content/uploads/2023/12/ORBBEC_Datasheet_Gemini-2-XL.pdf'),
)
SCENES=('ordinary','low_light','glare','floating_objects','uneven_surface','depth_failure',
        'transparent','translucent','multilayer','compound')
BANDS=((0.,0.),(.01,.1),(0.,1.))+tuple((float(i),float(i+1)) for i in range(1,9))

def spatial_field(rng, shape, block=16):
    a=rng.normal(size=((shape[0]+block-1)//block,(shape[1]+block-1)//block))
    return np.repeat(np.repeat(a,block,0),block,1)[:shape[0],:shape[1]]

def noisy_depth(z, profile, seed, frame, regime='nominal', ray_norm=None):
    if regime not in ('nominal','adverse'): raise ValueError('unknown noise assumption')
    z=np.asarray(z,float); valid=np.isfinite(z)&(z>0)
    rng=np.random.default_rng(np.random.SeedSequence([seed,frame,381]))
    global_rng=np.random.default_rng(np.random.SeedSequence([seed,382]))
    shared=spatial_field(global_rng,z.shape)
    # Common standardized innovations make the two sensitivity runs paired.
    e=rng.normal(size=z.shape)
    if regime=='adverse':e=np.sqrt(.5)*e+np.sqrt(.5)*shared
    if profile.sensor=='tof':
        rn=1. if ray_norm is None else np.asarray(ray_norm)
        distance=z*rn
        sigma=.0008+.0012*np.maximum(distance-5.5,0)**2
        if regime=='adverse':sigma*=3
        # Device/installation offset shared across the four frames, not averaged away.
        bias=global_rng.uniform(-.005,.005)+global_rng.uniform(-.002,.002)*np.maximum(distance-5.5,0)
        if regime=='adverse':bias+=global_rng.uniform(-.01,.01)*np.maximum(distance-5.5,0)
        result=(distance+bias+sigma*e)/rn
    else:
        fb=profile.fx*profile.baseline_m
        sigma=.18 if regime=='nominal' else .36
        bias=global_rng.uniform(-.025,.025)*(2 if regime=='adverse' else 1)
        disparity=fb/np.maximum(z,1e-6)+bias+sigma*e
        disparity=np.round(disparity*16)/16
        result=fb/np.maximum(disparity,1e-6)
    result=np.round(result/.001)*.001
    measured_range=result*(rn if profile.sensor=='tof' else 1.)
    return np.where(valid&(measured_range>=profile.min_z)&(measured_range<=profile.max_z),result,0).astype(np.float32)

def metric_summary(rows):
    out=[r for r in rows if r.get('predicted_height_m') is not None]
    e=np.array([abs(r['predicted_height_m']-r['true_height_m']) for r in out])
    h=np.array([r['true_height_m'] for r in out]);positive=np.array([not r.get('empty',False) and r['true_height_m']>0 for r in out],dtype=bool)
    rel=e[positive]/h[positive]*100
    ae=[0,5,10,20,50,100,200,float('inf')];re=[0,.5,1,2,3,5,10,float('inf')]
    def hist(x,edges):
        ns=np.histogram(x,edges)[0]
        return dict(counts=ns.tolist(),pct_output=(ns*100/max(len(x),1)).tolist(),
                    pct_all=(ns*100/max(len(rows),1)).tolist())
    return dict(frames=len(rows),outputs=len(out),no_output=len(rows)-len(out),
        clusters=len({r['cluster'] for r in rows}),numeric_output_pct=100*len(out)/max(len(rows),1),
        mae_mm=float(e.mean()*1000) if len(e) else None,p95_mm=float(np.percentile(e,95)*1000) if len(e) else None,
        signed_bias_mm=float(np.mean([r['predicted_height_m']-r['true_height_m'] for r in out])*1000) if out else None,
        absrel_positive_pct=float(rel.mean()) if len(rel) else None,relative_outputs=len(rel),
        relative_undefined_empty=sum(r.get('empty',False) for r in rows),
        pass_pct=100*int(np.sum(e<=np.maximum(.005,.02*h)))/max(len(rows),1),
        absolute_bins_mm=ae[:-1],relative_bins_pct=re[:-1],absolute=hist(e*1000,ae),relative=hist(rel,re))
