"""Camera observations: explicit units and factory calibration, no liquid priors."""
import cv2
import numpy as np

def _matrix(value,shape,name):
    result=np.asarray(value,dtype=np.float64)
    if result.shape!=shape or not np.isfinite(result).all():raise ValueError(f'Invalid {name}')
    return result

def prepare_observation(rgb_bgr,raw_depth,camera):
    """camera fields come from the capture SDK/export, never from a liquid label.
    Already registered: depth_alignment='color', depth_scale_to_m.
    Unregistered: depth_alignment='native', depth_camera_matrix, color_camera_matrix,
    depth_to_color_rotation, depth_to_color_translation_m, optional distortion arrays.
    The RGB and depth resolutions must match their exported calibration.
    """
    if rgb_bgr.ndim!=3 or rgb_bgr.shape[2]!=3 or raw_depth.ndim!=2:
        raise ValueError('Expected HWC BGR and HW depth')
    scale=float(camera['depth_scale_to_m'])
    if not np.isfinite(scale) or scale<=0:raise ValueError('A positive SDK depth scale is required')
    depth=raw_depth.astype(np.float64)*scale
    valid=np.isfinite(depth)&(depth>0)
    if camera.get('depth_alignment')=='color':
        if depth.shape!=rgb_bgr.shape[:2]:raise ValueError('Registered depth must match RGB dimensions')
        aligned=np.where(valid,depth,0)
    elif camera.get('depth_alignment')=='native':
        kd=_matrix(camera['depth_camera_matrix'],(3,3),'depth intrinsics')
        kc=_matrix(camera['color_camera_matrix'],(3,3),'color intrinsics')
        rot=_matrix(camera['depth_to_color_rotation'],(3,3),'rotation')
        trans=_matrix(camera['depth_to_color_translation_m'],(3,),'translation in meters')
        if not np.allclose(rot@rot.T,np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(rot),1,atol=1e-4):raise ValueError('Expected a rigid camera rotation')
        if min(kd[0,0],kd[1,1],kc[0,0],kc[1,1])<=0:raise ValueError('Invalid focal length')
        if list(depth.shape[::-1])!=camera['depth_resolution'] or list(rgb_bgr.shape[1::-1])!=camera['color_resolution']:
            raise ValueError('Use factory intrinsics for the actual capture resolutions')
        y,x=np.nonzero(valid);z=depth[y,x]
        aligned=np.full(rgb_bgr.shape[:2],np.inf,np.float64)
        if len(z):
            uv=np.stack([x,y],1).astype(np.float64)
            rays=cv2.undistortPoints(uv[:,None],kd,np.asarray(camera.get('depth_distortion',[0,0,0,0,0]),np.float64))[:,0]
            points=np.column_stack([rays*z[:,None],z])@rot.T+trans
            good=np.isfinite(points).all(1)&(points[:,2]>0)
            points=points[good]
            if len(points):
                projected,_=cv2.projectPoints(points,np.zeros(3),np.zeros(3),kc,np.asarray(camera.get('color_distortion',[0,0,0,0,0]),np.float64))
                pixels=np.rint(projected[:,0]).astype(np.int64);u,v=pixels.T
                h,w=aligned.shape;keep=(u>=0)&(u<w)&(v>=0)&(v<h)
                np.minimum.at(aligned,(v[keep],u[keep]),points[keep,2])
        aligned[~np.isfinite(aligned)]=0
    else:raise ValueError('SDK export must declare depth_alignment=color or native')
    rgb=cv2.cvtColor(rgb_bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return rgb,aligned.astype(np.float32)
