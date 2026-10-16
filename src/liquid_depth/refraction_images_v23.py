"""Image-valued Snell ray tracer with monochrome IR and scattering proxies.

Flat horizontal liquid and bottom, finite calibrated baseline, no glass wall.
This is NOT a spectral renderer or a device-calibrated radiative-transfer solver.
Ground-truth correspondences are returned separately for offline scoring only.
"""
import cv2
import numpy as np


def bottom_map(q, height, index, bottom, centre=(0., 0.)):
    q = np.asarray(q, float)
    a = np.sqrt(index**2+(index**2-1)*np.sum(q*q, -1))
    return np.asarray(centre) + q*(bottom-height+height/a)[..., None]


def inverse_bottom(x, height, index, bottom, centre=(0., 0.)):
    x = np.asarray(x)-np.asarray(centre)
    radius = np.linalg.norm(x, axis=-1)
    r = radius/(bottom-height+height/index)
    for _ in range(12):
        a = np.sqrt(index**2+(index**2-1)*r*r)
        length = bottom-height+height/a
        derivative = length-height*(index**2-1)*r*r/a**3
        r -= (r*length-radius)/derivative
    return x*np.divide(r, radius, out=np.zeros_like(r), where=radius>1e-12)[..., None]


def grid(width=480, height=320, fx=450., cx=None, cy=None):
    y, x = np.indices((height, width))
    return np.stack(((x-(width/2 if cx is None else cx))/fx,
                     (y-(height/2 if cy is None else cy))/fx), -1)


def texture(seed, size=1024):
    rng = np.random.default_rng(seed)
    noise = rng.random((size, size)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.1)
    noise = np.clip((noise-.5)*4+.5, 0, 1)
    # Aperiodic speckles with low-frequency albedo variation.
    coarse = cv2.resize(rng.random((32,32)).astype(np.float32), (size,size))
    return np.clip(.15+.7*noise+.15*coarse, 0, 1)


def render_capture(seed, height_m, index, bottom_m, baseline_m, condition, family,
                   width=480, height=320, fx=450.):
    """Returns observation images and a separate evaluator-only truth object."""
    q = grid(width,height,fx)
    tex = texture(seed)
    projector_width, projector_height = 512, 384
    condition_params = {'transparent': (0., .0, .003),
                        'translucent_mild': (1.2, .65, .006),
                        'translucent_strong': (4., 1.6, .012)}
    extinction, blur_strength, noise = condition_params[condition]
    rng = np.random.default_rng(seed+int(round(height_m/bottom_m*1e6))*101+7919)
    d = bottom_m-height_m

    def geometry(centre):
        x = bottom_map(q,height_m,index,bottom_m,centre)
        qp = inverse_bottom(x,height_m,index,bottom_m,(baseline_m,0))
        a = np.sqrt(index**2+(index**2-1)*np.sum(q*q,-1))
        ap = np.sqrt(index**2+(index**2-1)*np.sum(qp*qp,-1))
        # Phase-space path lengths for the camera and illuminating projector.
        path = height_m*np.sqrt(1+np.sum((q/a[...,None])**2,-1))
        path += height_m*np.sqrt(1+np.sum((qp/ap[...,None])**2,-1))
        transmission = np.exp(-extinction*path/bottom_m)
        mapx = (x[...,0]/bottom_m*900+512).astype(np.float32)
        mapy = (x[...,1]/bottom_m*900+512).astype(np.float32)
        albedo = cv2.remap(tex,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
        return x, qp, transmission, albedo

    left_geo = geometry((0,0))
    right_geo = geometry((baseline_m,0))
    x, qp, transmission, albedo = left_geo
    projector_x = qp[...,0]*fx+projector_width/2
    projector_y = qp[...,1]*fx+projector_height/2
    right_q = inverse_bottom(x,height_m,index,bottom_m,(baseline_m,0))
    truth = dict(right_u=(right_q[...,0]*fx+width/2).astype(np.float32),
                 right_v=(right_q[...,1]*fx+height/2).astype(np.float32),
                 projector_u=projector_x.astype(np.float32),
                 projector_v=projector_y.astype(np.float32),height_m=height_m,index=index)

    def image(signal, geo, channel):
        xx, qq, trans, alb = geo
        # Beer-Lambert ballistic term, blurred forward-scatter proxy and veil.
        ballistic = alb*(.12+.78*signal)*trans
        blurred = cv2.GaussianBlur(alb*(.12+.78*signal), (0,0),
                                   max(.15,blur_strength*(height_m/bottom_m)*2))
        out = .025+ballistic+(1-trans)*(.12+.30*blurred)
        if condition != 'transparent':
            out = cv2.GaussianBlur(out,(0,0),max(.15,blur_strength*height_m/bottom_m))
        out += rng.normal(0,noise,out.shape)
        out *= 1+(.035 if channel else -.025)
        return np.uint8(np.clip(out,0,1)*255)

    if family == 'active_stereo':
        # One shared projector pattern illuminates the same physical bottom.
        proj = texture(seed+117,1024)
        def projected(geo):
            qq=geo[1]
            return cv2.remap(proj,(qq[...,0]*1100+512).astype(np.float32),
                             (qq[...,1]*1100+512).astype(np.float32),cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        images = {'left': image(.35+.65*projected(left_geo),left_geo,0),
                  'right': image(.35+.65*projected(right_geo),right_geo,1)}
    elif family == 'structured_light':
        inside_projector=(projector_x>=0)&(projector_x<projector_width)&(projector_y>=0)&(projector_y<projector_height)
        images = {'black':image(np.zeros(q.shape[:2]),left_geo,0),
                  'white':image(inside_projector.astype(float),left_geo,0)}
        for axis, coordinate, size in [('x',projector_x,projector_width),('y',projector_y,projector_height)]:
            inside=inside_projector
            cell=np.floor(coordinate/16).astype(np.int32)
            gray=cell^(cell>>1)
            for bit in range(5):
                pattern=((gray>>bit)&1).astype(float)*inside
                images[f'{axis}_g{bit}']=image(pattern,left_geo,0)
                images[f'{axis}_i{bit}']=image((1-pattern)*inside,left_geo,0)
            for phase in range(4):
                signal=(.5+.5*np.cos(2*np.pi*coordinate/16+phase*np.pi/2))*inside
                images[f'{axis}_p{phase}']=image(signal,left_geo,0)
    else:
        raise ValueError('Unsupported triangulation family')
    calibration = dict(width=width,height=height,fx=fx,fy=fx,cx=width/2,cy=height/2,
        reference_cx=width/2 if family=='active_stereo' else projector_width/2,
        reference_cy=height/2 if family=='active_stereo' else projector_height/2,
        bottom_m=bottom_m,baseline_m=baseline_m,sensor_family=family,
        geometry_model='fixed_horizontal_bottom_flat_uniform_liquid',
        monochrome_ir_proxy=True,spectral_sensor_calibrated=False)
    return images, calibration, truth
