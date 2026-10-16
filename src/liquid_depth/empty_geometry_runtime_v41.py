"""Explicit opt-in panel/CLI bridge from empty-vessel geometry to V40 metrology."""
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from .io import load_frame,write_json
from .system_runtime import load_system_profile,save_system_profile,resolve_profile_path
from .empty_geometry_v41 import build_geometry,height_calibration
from .ordinary_surface_v40 import prepare,calibrate,estimate


def preview_empty_geometry(payload):
    frame=load_frame(payload['frame_dir'])
    scale=float(payload['depth_scale_to_m'])
    if not np.isfinite(scale) or scale<=0:raise ValueError('explicit_positive_depth_scale_to_m_required')
    bundle=build_geometry(frame.depth.astype(np.float32)*scale,frame.camera_matrix,payload['geometry'])
    # A plausible primitive is not enough: the selected ordinary route must
    # actually observe its dry reference through the inferred opening.
    ctx=prepare(height_calibration(bundle),bundle['rim_z_m'],frame.depth.shape,bundle['baseline_m'])
    ref=calibrate(frame.depth.astype(np.float32)*scale,ctx)
    bundle['quality']['ordinary_reference_observed_pixels']=ref.observed_pixels
    return frame,scale,bundle


def create_empty_geometry_profile(payload):
    frame,scale,bundle=preview_empty_geometry(payload)
    profile=Path(payload['profile_path']).expanduser().resolve()
    if profile.suffix.lower() not in ('.yaml','.yml'):raise ValueError('profile_requires_yaml_suffix')
    if profile.exists():raise ValueError('profile_exists_choose_a_new_name_to_preserve_existing_calibration')
    assets=profile.with_name(profile.stem+'_assets')
    if assets.exists():raise ValueError('calibration_assets_already_exist_choose_new_profile_name')
    assets.mkdir(parents=True,exist_ok=False)
    write_json(assets/'geometry.json',bundle)
    np.save(assets/'empty_depth_m.npy',frame.depth.astype(np.float32)*scale,allow_pickle=False)
    if not cv2.imwrite(str(assets/'empty_rgb.png'),frame.rgb_bgr):raise OSError('empty_rgb_save_failed')
    spec=dict(schema_version=1,name=profile.stem,
        measurement=dict(mode='empty_geometry_diffuse',geometry_path=str(assets/'geometry.json'),
            empty_depth_path=str(assets/'empty_depth_m.npy'),reference_image_path=str(assets/'empty_rgb.png'),
            research_only=True,production_qualified=False),
        camera=dict(camera_matrix=frame.camera_matrix.tolist(),image_size=[frame.depth.shape[1],frame.depth.shape[0]],
            distortion=[],source='rectified_frame_sdk_intrinsics',depth_scale_to_m=scale,depth_registered_to_color=True),
        deployment=dict(camera_motion_policy='recalibrate_after_move',ordinary_diffuse_only=True,
            ideal_geometry_assumed=False,independent_field_validation_required=True))
    save_system_profile(profile,spec)
    return profile,bundle


class EmptyGeometryLiquidDepthSystem:
    def __init__(self,profile_path,*,device=None,temporal=False):
        self.profile=load_system_profile(profile_path)
        m=self.profile.get('measurement',{})
        if m.get('mode')!='empty_geometry_diffuse':raise ValueError('not_empty_geometry_profile')
        self.bundle=json.loads(resolve_profile_path(self.profile,m['geometry_path']).read_text())
        self.cal=height_calibration(self.bundle)
        self.empty=np.load(resolve_profile_path(self.profile,m['empty_depth_path']),allow_pickle=False)
        self.scale=float(self.profile['camera']['depth_scale_to_m'])
        self.ctx=prepare(self.cal,self.bundle['rim_z_m'],self.empty.shape,self.bundle['baseline_m'])
        self.ref=calibrate(self.empty,self.ctx)
        self.temporal_requested=bool(temporal)

    def measure_depth(self,metric_depth):
        out=estimate(np.asarray(metric_depth),self.ctx,self.ref,'paired')
        out.update(geometry_source='empty_rgbd_plus_metric_anchor',geometry_quality=self.bundle['quality'],
            production_qualified=False,fixed_camera_required=True,temporal_enabled=False)
        return out

    def measure(self,frame_dir,output_dir=None):
        started=perf_counter();frame=load_frame(frame_dir)
        if frame.depth.shape!=self.empty.shape or not np.allclose(frame.camera_matrix,self.cal.intrinsics,rtol=0,atol=1e-5):
            raise ValueError('camera_intrinsics_or_shape_changed_recalibration_required')
        out=self.measure_depth(frame.depth.astype(np.float32)*self.scale)
        candidate=None if out['statistics'] is None else out['statistics']['mean_m']
        elapsed=(perf_counter()-started)*1000
        reasons=['research_geometry_requires_field_validation']+out['reasons']
        if self.temporal_requested:reasons.append('temporal_not_enabled_for_new_geometry_route')
        result=dict(schema_version=1,measurement_mode='empty_geometry_diffuse',frame_id=frame.frame_id,
            accepted=False,liquid_depth_m=None,liquid_depth_candidate_m=candidate,
            uncertainty_m=None,confidence_uncalibrated=None,rejection_reasons=reasons,
            message='空槽几何实验链路：候选液深仅供调试；现场独立验收和机位有效性检查尚未完成。',
            geometry_quality=self.bundle['quality'],diagnostics=out,temporal=None,
            latency=dict(inference_total_ms=elapsed,budget_ms=500.,within_budget=elapsed<=500.,
                budget_scope='frame_load_and_cpu_meter_excludes_capture_and_serialization'))
        if output_dir is not None:
            target=Path(output_dir);target.mkdir(parents=True,exist_ok=True)
            write_json(target/'depth_result.json',result)
            overlay=frame.rgb_bgr.copy()
            for field,color in [('projected_bottom_outline_px',(0,200,255)),('projected_rim_outline_px',(0,200,0))]:
                cv2.polylines(overlay,[np.round(self.bundle[field]).astype('int32')],True,color,2)
            label='RESEARCH '+('NO OUTPUT' if candidate is None else f'H={candidate*1000:.2f} mm')
            cv2.putText(overlay,label,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,230),2)
            cv2.imwrite(str(target/'measurement_overlay.png'),overlay)
        return result
