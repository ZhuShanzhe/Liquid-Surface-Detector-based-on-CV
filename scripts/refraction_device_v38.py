"""Gemini-geometry IR-like proxy; unchanged five-prior inference.

Finite bottom texture, Snell mapping, attenuation, pixel integration. No device
projector pattern, spectral BRDF or proprietary matcher is claimed reproduced.
"""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import cv2,numpy as np
from liquid_depth.device_profiles_v38 import PROFILES,metric_summary
from liquid_depth.refraction_images_v23 import bottom_map,inverse_bottom,texture
from liquid_depth.range_runtime_v33 import FivePriorImageMeter

class DeviceScene:
    def __init__(self,seed,bottom,capacity,width,index,grain,extinction,condition):
        p=PROFILES[1];self.seed=seed;self.index=index;self.extinction=extinction;self.condition=condition
        f=p.fx;b=p.baseline_m;d=width*bottom/f
        xlo=int(np.floor(-width/2-b*f/bottom))-8;xhi=int(np.ceil(width/2))+8
        ylo=int(np.floor(-.4*width))-8;yhi=int(np.ceil(.4*width))+8
        self.cal=dict(width=xhi-xlo,height=yhi-ylo,fx=f,cx=-xlo,cy=-ylo,bottom_m=bottom,baseline_m=b,
            container_max_height_m=capacity,container_width_m=d,container_length_m=.8*d,empty_width_px=width,
            sensor_width=p.width,sensor_height=p.height,crop_x0=p.width/2+xlo,crop_y0=p.height/2+ylo,
            fixed_camera=True,operator_roi_source='fixed_empty_bottom_geometry',glass_wall_raytracing=False,spectral_sensor_calibrated=False)
        yy,xx=np.indices((self.cal['height'],self.cal['width']));q=np.stack(((xx+xlo)/f,(yy+ylo)/f),-1);self.masks={}
        for ch,c in [('left',(0.,0.)),('right',(b,0.))]:
            v=bottom*q+c;self.masks[ch]=((abs(v[...,0])<d/2)&(abs(v[...,1])<.4*d)).astype(np.uint8)*255
        ss=3;self.ss=ss;yy,xx=np.indices((self.cal['height']*ss,self.cal['width']*ss))
        self.q=np.stack((((xx+.5)/ss-.5+xlo)/f,((yy+.5)/ss-.5+ylo)/f),-1)
        size=1536;self.step=1.4*d/size;self.size=size
        tex=np.random.default_rng(seed).random((size,size)).astype(np.float32)
        tex=cv2.GaussianBlur(tex,(0,0),max(.6,grain/self.step))
        self.albedo=np.clip((tex-tex.mean())/tex.std()*.15+.5,.05,.95);self.pattern=texture(seed+117,1024)
    def capture(self,height,capture_id):
        if not 0<=height<=9:raise ValueError('height_out_of_range')
        ca=self.cal;f=ca['fx'];b=ca['baseline_m'];bottom=ca['bottom_m'];d=ca['container_width_m']
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,capture_id,3833]));ims={}
        exposure=rng.normal(1,.01);offset=rng.normal(0,.0003)
        for ch,c in [('left',(0.,0.)),('right',(b,0.))]:
            world=bottom_map(self.q,height,self.index,bottom,c);qp=inverse_bottom(world,height,self.index,bottom,(b,0.))
            alb=cv2.remap(self.albedo,(world[...,0]/self.step+self.size/2).astype(np.float32),
                (world[...,1]/self.step+self.size/2).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            pat=cv2.remap(self.pattern,(qp[...,0]*f+512).astype(np.float32),(qp[...,1]*f+512).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            path=2*height*np.sqrt(1+np.sum(self.q*self.q,-1)/self.index**2);trans=np.exp(-self.extinction*path)
            signal=.025+alb*(.4+.5*pat)*trans+(1-trans)*.13
            if self.condition=='translucent_mild':signal=cv2.GaussianBlur(signal,(0,0),(.2+height*self.extinction)*self.ss)
            opening=np.array(c)+self.q*(bottom-ca['container_max_height_m'])
            at_open=(abs(opening[...,0])<d/2)&(abs(opening[...,1])<.4*d)
            at_bottom=(abs(world[...,0])<d/2)&(abs(world[...,1])<.4*d)
            signal=np.where(at_open,np.where(at_bottom,signal,.075),.14).astype(np.float32)
            signal=cv2.GaussianBlur(signal,(0,0),.4*self.ss)
            signal=cv2.resize(signal,(ca['width'],ca['height']),interpolation=cv2.INTER_AREA)
            signal=signal*exposure+offset+rng.normal(0,.003 if self.condition=='transparent' else .006,signal.shape)
            signal*=.975 if ch=='left' else 1.035
            ims[ch]=np.uint8(np.clip(signal,0,1)*255)
        return ims

def job(spec,data,out):
    cv2.setNumThreads(1);seed,bottom,capacity,width,index,grain,extinction=spec
    condition='transparent' if extinction==0 else 'translucent_mild'
    scene=DeviceScene(*spec[:-1],extinction,condition);folder=data/str(seed);folder.mkdir(exist_ok=False)
    known=np.array([0,.15,.4,.65,.9])*capacity;priors=[]
    for j,h in enumerate(known):
        ims=scene.capture(h,j);priors.append(ims)
        for ch,im in ims.items():cv2.imwrite(str(folder/f'prior_{j}_{ch}.png'),im)
    rng=np.random.default_rng(seed+17);known=known+np.r_[0,rng.normal(0,.0005+.001*known[1:])]
    try:
        meter=FivePriorImageMeter(scene.cal,scene.masks,priors,known,condition,
                                 method='baseline_v32' if extinction==0 else 'midpoint_dense')
        prior_error=None
    except ValueError as exc:
        meter=None;prior_error=str(exc)
    levels=[(float(h),'uniform') for h in np.linspace(.125,8.875,36)]+[(0.,'empty')]+[(x,'near_zero_enrichment') for x in (.005,.02,.05)]
    rows=[]
    for j,(h,pop) in enumerate(levels):
        ims=scene.capture(h,1000+j)
        for ch,im in ims.items():cv2.imwrite(str(folder/f'test_{j}_{ch}.png'),im)
        t=time.perf_counter()
        pred=meter.measure(ims) if meter is not None else dict(height_m=None,reasons=['prior_failed',prior_error])
        elapsed=time.perf_counter()-t
        value=pred.get('height_m');value=float(value) if value is not None and np.isfinite(value) else None
        rows.append(dict(cluster=str(seed),profile='gemini2xl_dense',scenario=condition,method='five_prior',regime='image_proxy',variant='base',
            population=pop,empty=h==0,true_height_m=h,predicted_height_m=value,absolute_error_m=None if value is None else abs(value-h),
            camera_to_bottom_m=bottom,camera_to_surface_m=bottom-h,extinction_per_m=extinction,grain_m=grain,width_px=width,
            band=0 if h==0 else (1 if pop=='near_zero_enrichment' else 2+min(int(h),8)),computation_s=elapsed,diagnostics=pred))
    (folder/'calibration.json').write_text(json.dumps(dict(cal=scene.cal,known_heights=known.tolist(),true_prior_heights=(np.array([0,.15,.4,.65,.9])*capacity).tolist()),indent=2))
    (out/(str(seed)+'.json')).write_text(json.dumps(rows,allow_nan=False));return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    a.data.mkdir(parents=True,exist_ok=True);a.output.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for i,geo in enumerate([(9.65,9.2,128),(9.95,9.4,176)]):
      for k,n in enumerate([1.31,1.445]):
       for g,grain in enumerate([.037,.081]):
        for e,ext in enumerate([0.,.033,.105]):jobs.append((202610123811+i*1009+k*137+g*17+e,*geo,n,grain,ext))
    if a.pilot:jobs=jobs[:1]
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(job,s,a.data,a.output) for s in jobs]):
            rows+=f.result();print('REFRACTION',len(rows),len(jobs)*40,flush=True)
    summary={}
    for sc in ['transparent','translucent_mild']:
        rr=[r for r in rows if r['scenario']==sc];summary[sc]={'all':metric_summary(rr)}
        for pop in ['empty','uniform','near_zero_enrichment']:summary[sc][pop]=metric_summary([r for r in rr if r['population']==pop])
        for b in range(11):summary[sc]['band_'+str(b)]=metric_summary([r for r in rr if r['band']==b])
    (a.output/'test-raw.json').write_text(json.dumps(dict(records=rows),allow_nan=False));(a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
