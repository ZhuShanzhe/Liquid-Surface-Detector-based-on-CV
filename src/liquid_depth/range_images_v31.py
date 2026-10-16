"""Finite vessel image proxy for actual liquid heights 0--9 m.

Fixed native-resolution crop from a 480x320 sensor. Geometry and images are
separate from evaluator-only height, index and extinction. Not a device model.
"""
import cv2
import numpy as np
from .refraction_images_v23 import bottom_map,inverse_bottom,texture


class RangeScene:
    def __init__(self,seed,bottom,capacity,empty_width_px,index,grain_m,extinction,condition):
        if not 9<capacity<bottom<=10:raise ValueError('requires_capacity_above_9_and_camera_at_most_10')
        if not 96<=empty_width_px<=180:raise ValueError('small_vessels_out_of_scope')
        if condition not in ('transparent','translucent_mild'):raise ValueError('outside_scope')
        self.seed=seed;self.index=index;self.extinction=extinction;self.condition=condition
        f=450.;baseline=.055;d=empty_width_px*bottom/f
        xlo=int(np.floor(-empty_width_px/2-baseline*f/bottom))-8;xhi=int(np.ceil(empty_width_px/2))+8
        ylo=int(np.floor(-.4*empty_width_px))-8;yhi=int(np.ceil(.4*empty_width_px))+8
        if xlo<-240 or xhi>240 or ylo<-160 or yhi>160:raise ValueError('crop_outside_sensor')
        self.cal=dict(width=xhi-xlo,height=yhi-ylo,fx=f,cx=-xlo,cy=-ylo,bottom_m=bottom,baseline_m=baseline,
            container_max_height_m=capacity,container_width_m=d,container_length_m=.8*d,empty_width_px=empty_width_px,
            sensor_width=480,sensor_height=320,crop_x0=240+xlo,crop_y0=160+ylo,fixed_camera=True,
            operator_roi_source='fixed_empty_bottom_geometry',glass_wall_raytracing=False,spectral_sensor_calibrated=False)
        y,x=np.indices((self.cal['height'],self.cal['width']))
        native=np.stack(((x+xlo)/f,(y+ylo)/f),-1);self.masks={}
        for ch,centre in [('left',(0.,0.)),('right',(baseline,0.))]:
            p=bottom*native+centre
            self.masks[ch]=((np.abs(p[...,0])<d/2)&(np.abs(p[...,1])<.4*d)).astype(np.uint8)*255
        ss=3;self.ss=ss;y,x=np.indices((self.cal['height']*ss,self.cal['width']*ss))
        self.q=np.stack((((x+.5)/ss-.5+xlo)/f,((y+.5)/ss-.5+ylo)/f),-1)
        size=1536;self.step=1.4*d/size;self.texture_size=size
        tex=np.random.default_rng(seed).random((size,size)).astype(np.float32)
        tex=cv2.GaussianBlur(tex,(0,0),max(.6,grain_m/self.step))
        self.albedo=np.clip((tex-tex.mean())/max(float(tex.std()),1e-6)*.15+.5,.05,.95)
        self.pattern=texture(seed+117,1024)

    def capture(self,height,capture_id):
        if not 0<=height<=9:raise ValueError('actual_height_outside_0_9')
        b=self.cal['bottom_m'];c=self.cal['container_max_height_m'];d=self.cal['container_width_m'];f=self.cal['fx']
        # Independent capture noise even when test and empty prior both have H=0.
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,int(capture_id),773]))
        exposure=float(rng.normal(1.,.01));offset=float(rng.normal(0.,.0003));images={}
        for ch,centre in [('left',(0.,0.)),('right',(.055,0.))]:
            world=bottom_map(self.q,height,self.index,b,centre)
            qp=inverse_bottom(world,height,self.index,b,(.055,0.))
            alb=cv2.remap(self.albedo,(world[...,0]/self.step+self.texture_size/2).astype(np.float32),
                (world[...,1]/self.step+self.texture_size/2).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            pat=cv2.remap(self.pattern,(qp[...,0]*f+512).astype(np.float32),(qp[...,1]*f+512).astype(np.float32),
                cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            path=2*height*np.sqrt(1+np.sum(self.q*self.q,axis=-1)/self.index**2)
            trans=np.exp(-self.extinction*path)
            signal=.025+alb*(.4+.5*pat)*trans+(1-trans)*.13
            if self.condition=='translucent_mild':
                signal=cv2.GaussianBlur(signal,(0,0),(.20+height*self.extinction)*self.ss)
            opening=np.asarray(centre)+self.q*(b-c)
            at_open=(np.abs(opening[...,0])<d/2)&(np.abs(opening[...,1])<.4*d)
            at_bottom=(np.abs(world[...,0])<d/2)&(np.abs(world[...,1])<.4*d)
            signal=np.where(at_open,np.where(at_bottom,signal,.075),.14)
            signal=cv2.GaussianBlur(signal.astype(np.float32),(0,0),.40*self.ss)
            signal=cv2.resize(signal,(self.cal['width'],self.cal['height']),interpolation=cv2.INTER_AREA)
            sigma=.003 if self.condition=='transparent' else .006
            signal=signal*exposure+offset+rng.normal(0,sigma,signal.shape)
            signal*=.975 if ch=='left' else 1.035
            images[ch]=np.uint8(np.clip(signal,0,1)*255)
        return images,dict(height_m=float(height),index=self.index,extinction_per_m=self.extinction,
            centre_ballistic_transmission=float(np.exp(-2*self.extinction*height)),capture_id=capture_id)
