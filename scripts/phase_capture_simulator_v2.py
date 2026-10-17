"""Independent observation generator: 4-phase quantization, transport, nuisance.

Forward projector inversion is bisection (runtime uses Newton); scattering uses
24 slices (runtime uses4). Pixel albedo, blur, coherent drift, harmonic distortion,
dropout and spatially correlated noise are not supplied as hidden inverse inputs.
Still a controlled analytic sensor-family simulation, not rendered spectral BRDF.
"""
import cv2,numpy as np

C=299792458.


def rays(profile):
    yy,xx=np.indices((48,64));f=profile['fx_px']
    return np.stack(((xx-31.5)/f+profile['center_q'][0],(yy-23.5)/f+profile['center_q'][1]),-1)


def generate(sensor,profile,physical,h,capture_id):
    rng=np.random.default_rng(np.random.SeedSequence([physical['seed'],capture_id,1 if sensor=='tof' else 2]))
    q=rays(profile);qx=q[...,0];qy=q[...,1];n=physical['index'];ext=physical['extinction']
    b=profile['bottom_m'];rho=physical['surface_ratio'];scatter=physical['scatter_strength']
    p=np.random.default_rng(physical['seed']+3)
    texture=cv2.GaussianBlur(p.uniform(.5,1.5,qx.shape).astype(np.float32),(0,0),1.5)
    texture/=texture.mean()
    ca=(1+qx*qx+qy*qy)**-.5;ct=np.sqrt(1-(1-ca*ca)/(n*n))
    if sensor=='tof':
        k=4*np.pi*np.asarray(profile['frequencies_hz'])/C
        base=b/ca
        bottom=(b-h)/ca+n*h/ct;surface=(b-h)/ca
        z=texture[None]*np.exp(-2*ext*h/ct)[None]*np.exp(1j*k[:,None,None]*bottom[None])
        z+=rho*np.exp(1j*k[:,None,None]*surface[None])
        for i in range(24):
            t=(i+.5)/24
            amplitude=scatter*(2*ext*h/ct)*np.exp(-2*ext*h*t/ct)/24
            z+=amplitude[None]*np.exp(1j*k[:,None,None]*((b-h)/ca+n*h*t/ct)[None])
        common_distance=rng.normal(0,.0007+.00015*b)
        spatial=cv2.resize(rng.normal(0,.001,(4,4)).astype(np.float32),(64,48),interpolation=cv2.INTER_CUBIC)
        z*=np.exp(1j*k[:,None,None]*(common_distance+spatial)[None])
        noise=1.5+.18*b
    else:
        baseline=profile['baseline_m'];fx=profile['fx_px'];k=2*np.pi/np.asarray(profile['fringe_periods_px'])
        a=np.sqrt(n*n+(n*n-1)*(qx*qx+qy*qy))
        target=np.stack(((b-h)*qx+h*qx/a-baseline,(b-h)*qy+h*qy/a),-1)
        norm=np.linalg.norm(target,axis=-1);unit=target/np.maximum(norm[...,None],1e-15)
        lo=np.zeros_like(norm);hi=np.maximum(norm/max(b-h,.02),1.)
        # Solve radial tan(theta) from Snell's law, independently of runtime Newton.
        for _ in range(40):
            t=(lo+hi)/2
            length=(b-h)*t+h*t/np.sqrt(n*n+(n*n-1)*t*t)
            lo=np.where(length<norm,t,lo);hi=np.where(length>=norm,t,hi)
        qp=unit*((lo+hi)/2)[...,None]
        bottom_u=fx*qp[...,0]+32.;surface_u=fx*(qx-baseline/(b-h))+32.
        z=texture[None]*np.exp(-2*ext*h)*np.exp(1j*k[:,None,None]*bottom_u[None])
        z+=rho*np.exp(1j*k[:,None,None]*surface_u[None])
        z+=scatter*(1-np.exp(-2*ext*h))*np.exp(1j*k[:,None,None]*((bottom_u+surface_u)/2)[None])
        drift=rng.normal(0,.0015+.0001*b)
        spatial=cv2.resize(rng.normal(0,.004,(4,4)).astype(np.float32),(64,48),interpolation=cv2.INTER_CUBIC)
        z*=np.exp(1j*k[:,None,None]*(drift+spatial)[None]);noise=2.+.3*b
    amplitude=420.*rng.uniform(.96,1.04)
    # DC/exposure noise, third harmonic, subpixel low-pass and12-bit digitization.
    result=[]
    for f in range(len(k)):
        frames=[]
        for shift in np.arange(4)*np.pi/2:
            image=1200.+amplitude*np.real(z[f]*np.exp(1j*shift))
            image+=amplitude*.0025*abs(z[f])*np.cos(3*(np.angle(z[f])+shift)+.4)
            sigma=(.25+ext*h*.5) if sensor=='structured_light' else .15
            image=cv2.GaussianBlur(image.astype(np.float32),(0,0),sigma)
            image+=rng.normal(0,noise,image.shape)
            frames.append(np.uint16(np.clip(np.rint(image),0,4095)))
        result.append(frames)
    result=np.asarray(result)
    dropout=rng.random(qx.shape)<(.02+.025*ext*h)
    result[:,:,dropout]=1200
    # Declared acquisition ROI comes from the empty image, not the current H.
    roi=(abs(qx-profile['center_q'][0])<.06)&(abs(qy-profile['center_q'][1])<.045)
    return result,roi
