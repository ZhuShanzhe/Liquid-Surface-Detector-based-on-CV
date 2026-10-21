"""Loopback-only industrial demonstration console; no camera/PLC control.

The simulation adapter owns truth and rendering. The meter receives only the
public ToFObservation/ToFInstallation contracts. API callers cannot supply code,
paths, shell arguments or ground-truth candidate masks.
"""
import argparse, base64, hashlib, json, math, os, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np
from .runtime_paths import resource_root

ROOT=resource_root()
CONSOLE_VERSION='scene_console_v14_desktop_fixed10mm_20261021'
INSTALLATION_KEYS=('material','container_height_m','camera_to_bottom_m','container_shape',
    'container_radius_x_m','container_radius_y_m','camera_offset_x_m','camera_offset_y_m',
    'calibration_noise_mode','refractive_index','extinction','scatter_strength','surface_ratio')

def framing_parameters(container_height_m,camera_to_bottom_m,auto_frame,*,radius_x_m=.55,
                       radius_y_m=.55,camera_xy_m=(0.,0.)):
    """Full rim with 12% margin; never substitute viewport zoom for intrinsics."""
    if not auto_frame:return float(camera_to_bottom_m),54.
    span=(max(radius_x_m,radius_y_m)+np.linalg.norm(camera_xy_m))*1.12*(480/270)
    clearance=span/math.tan(math.radians(100/2))
    required=float(container_height_m)+clearance
    bottom=float(camera_to_bottom_m) if camera_to_bottom_m>=required else math.ceil(required*1000)/1000
    if bottom>10.+1e-9:raise ValueError('full_view_cannot_fit_within_10m_and_100deg')
    needed=math.degrees(2*math.atan(span/(bottom-container_height_m)))
    return bottom,float(np.clip(needed,54.,100.))


def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def validate_controls(value):
    if not isinstance(value,dict):raise ValueError('controls_object_required')
    allowed={'height_m','material','darkness','glare','wave_m','float_count','loss',
             'container_height_m','camera_to_bottom_m','auto_frame',*INSTALLATION_KEYS}
    if set(value)-allowed:raise ValueError('unknown_control')
    defaults=dict(height_m=.6,material='ordinary_diffuse',darkness=0.,glare=0.,wave_m=0.,float_count=0,loss=0.,
                  container_height_m=1.5,camera_to_bottom_m=4.,auto_frame=True,
                  container_shape='cylinder',container_radius_x_m=.55,container_radius_y_m=.55,
                  camera_offset_x_m=0.,camera_offset_y_m=0.,calibration_noise_mode='fixed_10mm',
                  refractive_index=1.333,extinction=.045,scatter_strength=.16,surface_ratio=.07)
    out={**defaults,**value}
    if out['material'] not in ('ordinary_diffuse','homogeneous_transparent','homogeneous_translucent'):
        raise ValueError('unsupported_material')
    if out['container_shape'] not in ('cylinder','ellipse','rectangle'):raise ValueError('unsupported_container_shape')
    if out['calibration_noise_mode'] not in ('ideal','fixed_10mm'):raise ValueError('unsupported_calibration_noise_mode')
    for key,maximum in [('height_m',9.),('container_height_m',9.8),('camera_to_bottom_m',10.),
                        ('darkness',.98),('glare',1.),('wave_m',.12),('float_count',24),('loss',.5)]:
        x=out[key]
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or not 0<=x<=maximum:
            raise ValueError('out_of_range:'+key)
    ranges={'container_radius_x_m':(.1,3.),'container_radius_y_m':(.1,3.),
            'camera_offset_x_m':(-.75,.75),'camera_offset_y_m':(-.75,.75),
            'refractive_index':(1.31,1.45),'extinction':(.02,.08),
            'scatter_strength':(.08,.25),'surface_ratio':(.03,.12)}
    for key,(lo,hi) in ranges.items():
        x=out[key]
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or not lo<=x<=hi:
            raise ValueError('out_of_range:'+key)
    if out['container_shape']=='cylinder' and abs(out['container_radius_x_m']-out['container_radius_y_m'])>1e-8:
        raise ValueError('cylinder_requires_equal_radii')
    if any(abs(out['camera_offset_'+axis+'_m'])>.25*out['container_radius_'+axis+'_m']+1e-9 for axis in ('x','y')):
        raise ValueError('top_camera_offset_exceeds_quarter_radius')
    if int(out['float_count'])!=out['float_count']:raise ValueError('integer_float_count_required')
    if out['container_height_m']<.1:raise ValueError('container_height_minimum_0.1m')
    if not isinstance(out['auto_frame'],bool):raise ValueError('boolean_auto_frame_required')
    out['camera_to_bottom_m'],_=framing_for_controls(out)
    if out['camera_to_bottom_m']-out['container_height_m']<.1-1e-9:
        raise ValueError('top_camera_requires_at_least_0.1m_above_rim')
    if out['height_m']>out['container_height_m']:raise ValueError('liquid_exceeds_container_height')
    if out['wave_m']>min(out['height_m']*.3,(out['container_height_m']-out['height_m'])*.5):
        raise ValueError('wave_exceeds_physical_freeboard_or_liquid_depth')
    if out['height_m']==0 and out['float_count']:
        raise ValueError('floating_objects_require_liquid')
    if out['material']!='ordinary_diffuse' and (out['wave_m']>0 or out['float_count']>0):
        raise ValueError('透明/半透明与漂浮物或起伏的组合尚未验证，请先选择普通液体，或移除漂浮物与起伏')
    return out


def framing_for_controls(controls):
    return framing_parameters(controls['container_height_m'],controls['camera_to_bottom_m'],controls['auto_frame'],
        radius_x_m=controls['container_radius_x_m'],radius_y_m=controls['container_radius_y_m'],
        camera_xy_m=(controls['camera_offset_x_m'],controls['camera_offset_y_m']))


def validation_envelope(controls):
    """Physical validity is not equivalent to coverage by the published holdout."""
    rx,ry=controls['container_radius_x_m'],controls['container_radius_y_m']
    reasons=[]
    if not .1<=controls['height_m']<=9.:reasons.append('empty_or_sub_10cm_diagnostic_only')
    if not .3<=rx<=1.5:reasons.append('radius_outside_published_sampling')
    if not .65<=ry/rx<=1.45:reasons.append('aspect_outside_published_sampling')
    if any(abs(controls['camera_offset_'+axis+'_m'])>.22*rx+1e-9 for axis in ('x','y')):
        reasons.append('offset_outside_published_sampling')
    if controls['camera_to_bottom_m']<3.53206:reasons.append('camera_distance_below_actual_evaluated_minimum')
    if controls['material']!='ordinary_diffuse' and any(controls[k]>0 for k in ('darkness','glare','loss')):
        reasons.append('refractive_photometric_or_dropout_combination_exploratory')
    return dict(within_published_parameter_bounds=not reasons,exploratory=bool(reasons),reasons=reasons,
        not_a_new_performance_evaluation=True,field_metrology_qualified=False,
        published_liquid_range_m=[.1,9.],published_actual_camera_bottom_range_m=[3.53206,9.99906],
        published_radius_m=[.3,1.5],published_aspect=[.65,1.45],published_offset_fraction_radius=[-.22,.22])


def simulation_request(controls,*,seed,index,fov,empty=False):
    rx,ry=controls['container_radius_x_m'],controls['container_radius_y_m']
    request=dict(**controls,seed=seed,index=index,bottom_m=controls['camera_to_bottom_m'],
        capacity_m=controls['container_height_m'],radius_m=rx,aspect=ry/rx,
        shape={'cylinder':'cylindrical','ellipse':'elliptical','rectangle':'rectangular_tank'}[controls['container_shape']],
        camera_mount='top_offset',camera_xy_m=[controls['camera_offset_x_m'],controls['camera_offset_y_m']],
        horizontal_fov_deg=fov,sensor_model='formal_diverse_v15',sensor_severity=.18)
    if empty:request.update(height_m=0.,material='ordinary_diffuse',darkness=0.,glare=0.,wave_m=0.,float_count=0,loss=0.)
    return request


def depth_preview(depth):
    valid=np.isfinite(depth)&(depth>0)
    if not valid.any():return None,None
    lo,hi=map(float,np.percentile(depth[valid],[1,99]))
    normalized=np.where(valid,np.clip((depth-lo)/max(hi-lo,1e-6),0,1),0)
    preview=cv2.applyColorMap(np.uint8(normalized*255),cv2.COLORMAP_TURBO)
    preview[~valid]=0
    _,encoded=cv2.imencode('.png',preview)
    return 'data:image/png;base64,'+base64.b64encode(encoded).decode(),[lo,hi]


def simulated_dimension_geometry(dry,cal,bottom,capacity,radius,spec):
    """Legacy diagnostic helper; NOT used by the desktop/formal initialization.

    Nothing here consumes a wet frame or current liquid height. Field use would
    require a measured vessel diameter, bottom distance and camera pose; this is
    not claimed to recover an unobserved complete boundary from an image.
    """
    from .empty_geometry_v41 import _plane
    from .liquid_height_metrics import world_points
    points=world_points(np.full(dry.shape,bottom),cal)
    mask=(points[:,:,:2]**2).sum(-1)<(radius*.75)**2
    up,raw_distance,quality=_plane(dry,cal.intrinsics,mask)
    known_up=np.asarray(cal.camera_to_world_cv[2,:3])
    disagreement=float(np.degrees(np.arccos(np.clip(up@known_up,-1,1))))
    if disagreement>8 or abs(raw_distance-bottom)>max(.03,.03*bottom):
        raise ValueError('empty_bottom_disagrees_with_independent_installation')
    quality.update(camera_axis_tilt_deg=0.,normal_source='independent_simulated_gravity',
                   full_boundary_observed=False,independent_container_dimensions_required=True,
                   raw_vs_metric_distance_delta_m=bottom-raw_distance,
                   depth_plane_gravity_disagreement_deg=disagreement)
    angle=np.linspace(0,2*np.pi,64,endpoint=False)
    circle=np.column_stack((radius*np.cos(angle),radius*np.sin(angle),np.zeros(64)))
    def project(vertices):
        cam=(vertices-cal.camera_to_world_cv[:3,3])@cal.camera_to_world_cv[:3,:3]
        p=cam@cal.intrinsics.T
        return (p[:,:2]/p[:,2:]).tolist()
    return dict(schema_version=1,source='empty_rgbd_plus_independent_simulated_installation_dimensions',
        shape='circle',sensor_family='tof',baseline_m=0.,rim_z_m=capacity,
        image_size=[dry.shape[1],dry.shape[0]],
        calibration=dict(intrinsics=cal.intrinsics.tolist(),camera_to_world_cv=cal.camera_to_world_cv.tolist(),
                         bottom_z_m=0.,radii_m=[radius,radius],footprint_exponent=2.),
        bottom_plane_camera=dict(up=known_up.tolist(),perpendicular_distance_m=bottom),
        internal_dimensions_m=dict(width=radius*2,length=radius*2,height=capacity),
        projected_bottom_outline_px=project(circle),projected_rim_outline_px=project(circle+np.array([0,0,capacity])),
        operator_inputs=spec,quality=quality,production_qualified=False,continuous_motion_supported=False,
        supported_routes=['ordinary_diffuse_v40'],
        warnings=['simulation_independent_geometry_not_image_only_inference',
                  'field_use_requires_measured_dimensions_and_independent_pose'])


class Console:
    def __init__(self,work,blender,checkpoint,*,compound_repair=True):
        from .scene_classifier import SceneClassifier
        # Blender resolves relative render.filepath against its own blend-file
        # directory, not Python cwd. Every capture path must be absolute.
        self.work=Path(work).resolve();self.work.mkdir(parents=True,exist_ok=True)
        self.blender=blender;self.classifier=SceneClassifier(checkpoint)
        self.worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='simulation')
        self.lock=threading.Lock();self.jobs={};self.state={'status':'not_initialized','result':None}
        self.meter=None;self.initialization=None;self.counter=0
        self.controls=validate_controls({});self.token=uuid.uuid4().hex
        self.compound_repair=bool(compound_repair)
        self.render_engine=os.environ.get('LIQUID_RENDER_ENGINE','eevee')
        if self.render_engine not in ('eevee','cycles_cpu'):raise ValueError('unsupported_render_engine')

    def submit(self,controls,initialize=False):
        requested=dict(controls)
        controls=validate_controls(controls)
        with self.lock:
            if any(v['status'] in ('queued','running') for v in self.jobs.values()):
                raise ValueError('simulation_busy')
            job=uuid.uuid4().hex;self.jobs[job]=dict(status='queued',started=time.time(),
                requested_camera_to_bottom_m=requested.get('camera_to_bottom_m',4.))
        self.worker.submit(self.run,job,controls,initialize)
        return job

    def capture(self,request,folder):
        folder=Path(folder).resolve()
        request={**request,'render_engine':self.render_engine}
        request_path=folder.parent/(folder.name+'-request.json')
        request_path.write_text(json.dumps(request))
        with (folder.parent/(folder.name+'-render.log')).open('w') as log:
            p=subprocess.run([self.blender,'--factory-startup','-b','-t','4','-P',str(ROOT/'scripts/render_scene_console.py'),
                              '--','--request',str(request_path),'--out',str(folder)],
                             stdout=log,stderr=subprocess.STDOUT,timeout=600,cwd=ROOT,
                             creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if p.returncode or not (folder/'complete.json').exists():
            raise RuntimeError('simulation_capture_failed_check_server_render_log')

    def initialize(self,controls,horizontal_fov_deg):
        from .unified_tof import ToFInstallation
        sys.path.insert(0,str(ROOT/'scripts'))
        from measure_release import validated_profile
        from tof_phase_simulator import generate,rays
        name='installation-'+uuid.uuid4().hex[:10];folder=self.work/name
        folder.mkdir()
        material=controls['material'];capacity=float(controls['container_height_m']);bottom=float(controls['camera_to_bottom_m'])
        # The initialization seed depends on installation parameters, never H.
        installation_controls={k:controls[k] for k in INSTALLATION_KEYS}
        # Toggling annotation uncertainty must not change the physical texture
        # or forward sensor realization of the same installation.
        seed_controls={k:v for k,v in installation_controls.items() if k!='calibration_noise_mode'}
        installation_seed=int(hashlib.sha256(json.dumps(seed_controls,sort_keys=True).encode()).hexdigest()[:8],16)
        request=simulation_request(controls,seed=installation_seed,index=0,fov=horizontal_fov_deg,empty=True)
        self.capture(request,folder/'empty')
        metadata=json.loads((folder/'empty/metadata.json').read_text())
        # The simulated independent ruler/outline is made from EMPTY acquisition
        # only; current liquid H and current semantic truth never enter this step.
        (folder/'installation/00000').mkdir(parents=True);(folder/'dry/0000').mkdir(parents=True)
        (folder/'installation/00000/metadata.json').write_text(json.dumps(metadata))
        dry=np.load(folder/'empty/dry_depth_m.npy',allow_pickle=False)
        np.save(folder/'dry/0000/dry_depth_m.npy',dry)
        from .liquid_height_metrics import HeightCalibration
        from .simulation_calibration import empty_calibration,phase_annotations,phase_modulation_qualification
        cal=HeightCalibration.from_simulation(metadata)
        with np.load(folder/'empty/offline_geometry.npz',allow_pickle=False) as empty_geometry:
            empty_visible=empty_geometry['semantic']==2
        bundle,audit=empty_calibration(dry,cal,empty_visible,shape=controls['container_shape'],
            radii_m=(controls['container_radius_x_m'],controls['container_radius_y_m']),capacity_m=capacity,
            bottom_m=bottom,noise_mode=controls['calibration_noise_mode'],seed=installation_seed)
        item=dict(bundle=bundle,audit=audit)
        profile=folder/'portable-profile.json';profile.write_text(json.dumps(validated_profile(ROOT)))
        physical=dict(seed=installation_seed,index=controls['refractive_index'],
                      extinction=controls['extinction'] if material.endswith('translucent') else 0.,
                      scatter_strength=controls['scatter_strength'] if material.endswith('translucent') else 0.,
                      surface_ratio=controls['surface_ratio'])
        phase_profile=dict(bottom_m=bottom,container_max_height_m=capacity,fx_px=float(cal.intrinsics[0,0]),center_q=[0.,0.],
                           frequencies_hz=[20e6,30e6,50e6],fixed_camera=True,sensor_profile='amcw_itof_phase3_formal_v1')
        known=np.array([0.,.1,.35,.65,.95])*capacity
        captures=[]
        for j,h in enumerate(known):
            c,roi=generate('tof',phase_profile,physical,float(h),j);captures.append(c)
        # The phase ROI is the same native central crop used in the formal
        # diverse evaluation. It cannot invent phase evidence behind a wall.
        roi=roi&empty_visible[111:159,208:272]
        phase=dict(profile=phase_profile,ray_coordinates=rays(phase_profile),roi=roi,
                   prior_correlations=captures,known_heights_m=known)
        phase_audit=None
        if material!='ordinary_diffuse':
            if int(roi.sum())<64:raise ValueError('insufficient_empty_projected_phase_support')
            phase,phase_audit=phase_annotations(phase,audit,noise_mode=controls['calibration_noise_mode'],seed=installation_seed+1)
            prior_support=phase_modulation_qualification(phase)
            if not prior_support['eligible']:raise ValueError('insufficient_prior_modulation_support')
            phase_audit['modulation_support']=prior_support
        installation=ToFInstallation(name,item['bundle'],dry,confirmed_material=material,phase=phase,
             provenance=dict(mode='simulation_empty_plus_independent_ruler_and_five_known_priors',
                             current_test_truth_used=False,field_calibration_requires_operator_measurements=True),
             confirmed_refractive_planar_unoccluded=material!='ordinary_diffuse')
        from .formal_tof_runtime import create_formal_meter
        self.meter=create_formal_meter(installation,classifier=self.classifier,profile_path=profile,
                                  compound_repair=self.compound_repair)
        preload_ms=self.meter.preload_optional_peak_models() if material=='ordinary_diffuse' else 0.
        warmup_ms=self.meter.warm_optional_peak_models() if material=='ordinary_diffuse' and self.compound_repair else 0.
        # Fit phase calibration at initialization, not inside the reported warm inference latency.
        if material!='ordinary_diffuse':
            from .tof_scene_meter import ToFSceneMeter
            self.meter.phase_cache[material]=ToFSceneMeter.from_phase_priors(sensor='tof',scene=material,**phase)
        self.initialization=dict(id=name,folder=str(folder),audit=item['audit'],
             **installation_controls,camera_mount='top_offset',
             camera_above_rim_m=bottom-capacity,
             horizontal_fov_deg=horizontal_fov_deg,intrinsics=cal.intrinsics.tolist(),
             phase_known_heights_m=phase['known_heights_m'].tolist(),phase_calibration_precomputed=material!='ordinary_diffuse',
             phase_prior_annotation=phase_audit,
             simulated_ruler_is_exact=controls['calibration_noise_mode']=='ideal',current_truth_used_for_measurement=False,
             optional_peak_asset_preload_ms=preload_ms,optional_peak_assets_preloaded=material=='ordinary_diffuse')
        self.initialization['formal_runtime']=self.meter.formal_runtime
        self.initialization['compound_repair_requested']=self.compound_repair
        self.initialization['compound_repair_enabled']=self.meter.compound_repair is not None
        self.initialization['optional_peak_kernel_warmup_ms']=warmup_ms
        self.initialization['warmup_predictions_used_as_measurement_or_calibration']=False
        (folder/'geometry.json').write_text(json.dumps(item['bundle']))
        self.forward_physical,self.forward_profile=physical,phase_profile
        # Display-only producer pose: fitted ellipse/rectangle local axes may
        # be rotated 90 degrees. Never pass this truth pose to the meter.
        self.display_calibration=cal

    def run(self,job,controls,initialize):
        from .unified_tof import ToFObservation
        begin=time.perf_counter()
        with self.lock:self.jobs[job]['status']='running';self.state['status']='simulating'
        try:
            _,fov=framing_for_controls(controls)
            changed=self.initialization is None or any(self.initialization.get(k)!=controls[k]
                        for k in INSTALLATION_KEYS)
            changed=changed or self.initialization.get('horizontal_fov_deg')!=fov
            initialization_ms=0.
            if initialize or self.meter is None or changed:
                initialization_start=time.perf_counter()
                self.initialize(controls,fov)
                initialization_ms=(time.perf_counter()-initialization_start)*1000
            self.counter+=1;folder=self.work/f'frame-{self.counter:06d}-{job[:6]}'
            request=simulation_request(controls,seed=202610191901,index=self.counter,fov=fov)
            render_start=time.perf_counter();self.capture(request,folder)
            render_ms=(time.perf_counter()-render_start)*1000
            rgb=cv2.imread(str(folder/'rgb.png'));depth=np.load(folder/'depth_m.npy',allow_pickle=False)
            correlations=None
            if controls['material']!='ordinary_diffuse':
                from tof_phase_simulator import generate
                correlations,_=generate('tof',self.forward_profile,self.forward_physical,controls['height_m'],100+self.counter)
                # Controlled modulation dropout; never fabricate absent samples.
                rng=np.random.default_rng(100+self.counter)
                absent=rng.random(correlations.shape[-2:])<controls['loss']
                correlations[:,:,absent]=1200
            from .sim_input_qualification import qualify_simulated_input,measure_if_eligible
            # Producer-only visibility is an offline SIMULATION eligibility check.
            # Neither it nor current truth/height is an estimator input.
            with np.load(folder/'offline_geometry.npz',allow_pickle=False) as visibility:
                semantic=visibility['semantic']
                if correlations is not None:
                    correlations[:,:,(semantic[111:159,208:272]!=2)]=1200
                minimum=self.meter.compound_repair.policy.minimum_sensor_depth_m if self.meter.compound_repair is not None else 0.
                qualification=qualify_simulated_input(depth,semantic,
                    material=controls['material'],has_floats=controls['float_count']>0,
                    correlations=correlations,phase_roi=self.meter.installation.phase['roi'],minimum_depth_m=minimum)
                if correlations is not None:
                    from .simulation_calibration import phase_modulation_qualification
                    phase_check=phase_modulation_qualification(self.meter.installation.phase,correlations)
                    reasons=qualification['reasons']+phase_check['reasons']
                    qualification.update(phase_check)
                    qualification.update(reasons=reasons,eligible=not reasons)
                # Same producer-only genuine-return checks as the formal
                # holdout. These visibility labels never enter the meter.
                if controls['material']=='ordinary_diffuse':
                    with np.load(folder/'offline_return_origin.npz',allow_pickle=False) as origins:
                        genuine=(origins['error_type']==0)&(depth>minimum)&np.isfinite(depth)
                    support=(semantic==2)&genuine
                    yy,xx=np.nonzero(support)
                    span=min(int(np.ptp(yy))+1,int(np.ptp(xx))+1) if len(xx) else 0
                    qualification.update(genuine_liquid_return_pixels=int(support.sum()),genuine_liquid_span_px=span)
                    if support.sum()<30 or span<3:qualification['reasons'].append('insufficient_genuine_liquid_support')
                    if controls['float_count']:
                        n=int((visibility['highest_instance_visible']&genuine).sum())
                        qualification['highest_instance_genuine_return_pixels']=n
                        if n<3:qualification['reasons'].append('highest_instance_not_observable')
                    qualification['eligible']=not qualification['reasons']
            observation=ToFObservation(job,rgb,depth,time.time(),correlations,
                installation_id=self.meter.installation.installation_id,capture_validity=np.isfinite(depth)&(depth>0))
            result=measure_if_eligible(self.meter,observation,qualification)
            if result is None:
                result=dict(liquid_depth_m=None,highest_height_m=None,numeric_output=False,
                    inference_executed=False,pipeline_ms=0.,precision_eligible=False,
                    classification=dict(material=controls['material'],material_score=0.,tags=[],
                                        model_id='not_executed_input_rejected'),
                    module_plan=dict(tags=[],modules=['simulation_input_validation'],executed_modules=['simulation_input_validation'],
                        blocked_reasons=qualification['reasons'],warnings=[],composite=False,visual_peak=False,liquid_route='not_executed'))
            else:result['inference_executed']=True
            # Only AFTER inference, load current truth for display/scoring.
            truth=json.loads((folder/'truth.json').read_text())
            actual={k:truth[k] for k in ('liquid_height_m','highest_height_m','lowest_height_m')}
            estimate=dict(liquid_height_m=result.get('liquid_depth_m'),highest_height_m=result.get('highest_height_m'))
            from .empty_geometry_v41 import height_calibration
            from .liquid_height_metrics import world_points
            points=[];colors=[]
            if qualification['eligible']:
                xyz=world_points(depth,self.display_calibration)
                good=(depth>0)&self.meter.roi
                yy,xx=np.nonzero(good);stride=max(1,len(yy)//5500);yy,xx=yy[::stride],xx[::stride]
                points=xyz[yy,xx].tolist();colors=rgb[yy,xx,::-1].tolist()
            _,encoded=cv2.imencode('.jpg',rgb,[cv2.IMWRITE_JPEG_QUALITY,84])
            depth_image,depth_range=depth_preview(depth)
            payload=dict(status='ready',software_version=CONSOLE_VERSION,frame_id=job,controls=controls,actual=actual,estimate=estimate,
                input_qualification=qualification,processing_status='measured' if qualification['eligible'] else 'input_rejected',
                framing=dict(auto=controls['auto_frame'],horizontal_fov_deg=fov,
                             requested_camera_to_bottom_m=self.jobs[job]['requested_camera_to_bottom_m'],
                             effective_camera_to_bottom_m=controls['camera_to_bottom_m'],
                             camera_distance_adjusted=abs(controls['camera_to_bottom_m']-self.jobs[job]['requested_camera_to_bottom_m'])>1e-9,
                             radii_m=[controls['container_radius_x_m'],controls['container_radius_y_m']],margin_fraction=.12),
                result=plain(result),initialization=self.initialization,
                viewer=dict(meshes=truth['meshes'],observed_points=points,observed_colors=colors,
                            radius_m=controls['container_radius_x_m'],shape=controls['container_shape'],
                            radii_m=[controls['container_radius_x_m'],controls['container_radius_y_m']],
                            camera_xy_m=[controls['camera_offset_x_m'],controls['camera_offset_y_m']],
                            capacity_m=controls['container_height_m'],camera_to_bottom_m=controls['camera_to_bottom_m'],
                            camera_mount='top_offset',horizontal_fov_deg=fov,image_aspect=480/270),
                rgb='data:image/jpeg;base64,'+base64.b64encode(encoded).decode(),
                depth_preview=depth_image,depth_preview_range_m=depth_range,
                validation_envelope=validation_envelope(controls),
                rendering=dict(engine=self.render_engine,device='CPU' if self.render_engine=='cycles_cpu' else 'graphics',
                    published_renderer_unchanged=self.render_engine=='eevee',
                    cpu_port_validation_only=self.render_engine=='cycles_cpu',
                    cpu_renderer_performance_report_not_applicable=self.render_engine=='cycles_cpu'),
                timing=dict(render_ms=render_ms,pipeline_ms=result['pipeline_ms'],initialization_ms=initialization_ms,
                            total_ms=(time.perf_counter()-begin)*1000),
                simulation_only=True,
                caveats=['not_a_hardware_digital_twin','RGB_render_and_AMCW_transport_are_separate_controlled_proxies',
                         'compound_scenes_require_independent_end_to_end_validation',
                         'V8_highest_tail_error_not_fixed_by_scene_routing'])
            (folder/'result.json').write_text(json.dumps(plain({k:v for k,v in payload.items() if k not in ('viewer','rgb')}),indent=2))
            with self.lock:
                self.controls=controls;self.state=payload;self.jobs[job].update(status='complete',finished=time.time())
        except Exception as error:
            import traceback
            traceback.print_exc()
            with self.lock:
                self.state=dict(status='error',message=str(error));self.jobs[job].update(status='error',message=str(error))


def serve(console,port,*,ready_callback=None,stop_event=None):
    class Handler(BaseHTTPRequestHandler):
        def send(self,value,status=200,kind='application/json; charset=utf-8'):
            body=value if isinstance(value,bytes) else json.dumps(plain(value),ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(body)

        def do_GET(self):
            if self.headers.get('Host','').split(':')[0] not in ('127.0.0.1','localhost'):
                return self.send({'error':'loopback_host_required'},403)
            if self.path=='/':return self.send((ROOT/'src/liquid_depth/ui/scene_console.html').read_bytes(),kind='text/html; charset=utf-8')
            assets={'/console.js':('scene_console.js','text/javascript; charset=utf-8'),'/console.css':('scene_console.css','text/css; charset=utf-8')}
            if self.path in assets:
                name,kind=assets[self.path];return self.send((ROOT/'src/liquid_depth/ui'/name).read_bytes(),kind=kind)
            if self.path=='/api/status':
                with console.lock:return self.send(dict(state=console.state,jobs=console.jobs,csrf=console.token,
                    software_version=CONSOLE_VERSION,supports_dynamic_installation=True))
            return self.send({'error':'not_found'},404)

        def do_POST(self):
            if self.path not in ('/api/apply','/api/initialize'):return self.send({'error':'not_found'},404)
            if self.headers.get('X-Console-Token')!=console.token:return self.send({'error':'invalid_request_token'},403)
            # Bind only loopback. Reject cross-site mutation (including DNS rebinding).
            host=self.headers.get('Host','').split(':')[0]
            if host not in ('127.0.0.1','localhost'):return self.send({'error':'loopback_host_required'},403)
            origin=self.headers.get('Origin')
            if origin and origin not in ('http://'+self.headers.get('Host',''),):return self.send({'error':'same_origin_required'},403)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('invalid_request_length')
                value=json.loads(self.rfile.read(length));job=console.submit(value,self.path=='/api/initialize')
                return self.send({'job':job},202)
            except (ValueError,TypeError,json.JSONDecodeError) as error:return self.send({'error':str(error)},400)

        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    if ready_callback is not None:ready_callback(server)
    if stop_event is not None:
        def shutdown_when_requested():
            stop_event.wait();server.shutdown()
        threading.Thread(target=shutdown_when_requested,daemon=True).start()
    print('SCENE_CONSOLE_READY',json.dumps(dict(port=server.server_port,binding='127.0.0.1')),flush=True)
    try:server.serve_forever()
    finally:server.server_close()


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--auto-initialize',action='store_true',help='Warm the default independent installation after server startup')
    p.add_argument('--compound-repair',action='store_true',default=True,help='Formal v2 evidence repair (default)')
    p.add_argument('--legacy-geometry',dest='compound_repair',action='store_false',help='Diagnostic old numeric path; not the selected release')
    p.add_argument('--blender',default='blender',help='Blender 4.2 executable or bundled runtime path')
    a=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    console=Console(a.work,a.blender,a.checkpoint,compound_repair=a.compound_repair)
    def ready(server):
        if a.auto_initialize:console.submit({},initialize=True)
    serve(console,a.port,ready_callback=ready)


if __name__=='__main__':main()
