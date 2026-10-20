"""Loopback-only industrial demonstration console; no camera/PLC control.

The simulation adapter owns truth and rendering. The meter receives only the
public ToFObservation/ToFInstallation contracts. API callers cannot supply code,
paths, shell arguments or ground-truth candidate masks.
"""
import argparse, base64, hashlib, json, math, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, torch
from .scene_classifier import SceneClassifier
from .unified_tof import ToFObservation, ToFInstallation, UnifiedToFMeter

ROOT=Path(__file__).resolve().parents[2]


def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def validate_controls(value):
    if not isinstance(value,dict):raise ValueError('controls_object_required')
    allowed={'height_m','material','darkness','glare','wave_m','float_count','loss'}
    if set(value)-allowed:raise ValueError('unknown_control')
    defaults=dict(height_m=.6,material='ordinary_diffuse',darkness=0.,glare=0.,wave_m=0.,float_count=0,loss=0.)
    out={**defaults,**value}
    if out['material'] not in ('ordinary_diffuse','homogeneous_transparent','homogeneous_translucent'):
        raise ValueError('unsupported_material')
    for key,maximum in [('height_m',1.5),('darkness',.98),('glare',1.),('wave_m',.12),('float_count',24),('loss',.5)]:
        x=out[key]
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or not 0<=x<=maximum:
            raise ValueError('out_of_range:'+key)
    if int(out['float_count'])!=out['float_count']:raise ValueError('integer_float_count_required')
    if out['wave_m']>min(out['height_m']*.3,(1.5-out['height_m'])*.5):
        raise ValueError('wave_exceeds_physical_freeboard_or_liquid_depth')
    if out['height_m']==0 and out['float_count']:
        raise ValueError('floating_objects_require_liquid')
    if out['material']!='ordinary_diffuse' and (out['wave_m']>0 or out['float_count']>0):
        raise ValueError('透明/半透明与漂浮物或起伏的组合尚未验证，请先选择普通液体，或移除漂浮物与起伏')
    return out


class Console:
    def __init__(self,work,blender,checkpoint,*,compound_repair=True):
        self.work=work;work.mkdir(parents=True,exist_ok=True)
        self.blender=blender;self.classifier=SceneClassifier(checkpoint)
        self.worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='simulation')
        self.lock=threading.Lock();self.jobs={};self.state={'status':'not_initialized','result':None}
        self.meter=None;self.initialization=None;self.counter=0
        self.controls=validate_controls({});self.token=uuid.uuid4().hex
        self.compound_repair=bool(compound_repair)

    def submit(self,controls,initialize=False):
        controls=validate_controls(controls)
        with self.lock:
            if any(v['status'] in ('queued','running') for v in self.jobs.values()):
                raise ValueError('simulation_busy')
            job=uuid.uuid4().hex;self.jobs[job]=dict(status='queued',started=time.time())
        self.worker.submit(self.run,job,controls,initialize)
        return job

    def capture(self,request,folder):
        request_path=folder.parent/(folder.name+'-request.json')
        request_path.write_text(json.dumps(request))
        with (folder.parent/(folder.name+'-render.log')).open('w') as log:
            p=subprocess.run([self.blender,'-b','-t','4','-P',str(ROOT/'scripts/render_scene_console.py'),
                              '--','--request',str(request_path),'--out',str(folder)],
                             stdout=log,stderr=subprocess.STDOUT,timeout=180,cwd=ROOT)
        if p.returncode or not (folder/'complete.json').exists():
            raise RuntimeError('simulation_capture_failed_check_server_render_log')

    def initialize(self,material):
        sys.path.insert(0,str(ROOT/'scripts'))
        from measure_release import validated_profile
        from tof_phase_simulator import generate,rays
        name='installation-'+uuid.uuid4().hex[:10];folder=self.work/name
        folder.mkdir()
        request=dict(seed=202610191901,index=0,bottom_m=4.,capacity_m=1.5,height_m=0.,radius_m=.55,
                     material=material,darkness=0.,glare=0.,wave_m=0.,float_count=0,loss=0.)
        self.capture(request,folder/'empty')
        metadata=json.loads((folder/'empty/metadata.json').read_text())
        # The simulated independent ruler/outline is made from EMPTY acquisition
        # only; current liquid H and current semantic truth never enter this step.
        (folder/'installation/00000').mkdir(parents=True);(folder/'dry/0000').mkdir(parents=True)
        (folder/'installation/00000/metadata.json').write_text(json.dumps(metadata))
        dry=np.load(folder/'empty/dry_depth_m.npy',allow_pickle=False)
        np.save(folder/'dry/0000/dry_depth_m.npy',dry)
        from .liquid_height_metrics import HeightCalibration
        from .empty_scene_routes_v44 import build_scene_geometry
        # Ideal demonstration calibration: exact independent simulated ruler,
        # gravity and EMPTY outline. No current liquid truth is accessible here.
        cal=HeightCalibration.from_simulation(metadata)
        angle=np.linspace(0,2*np.pi,12,endpoint=False);radius=.55
        def project(points):
            camera=(np.asarray(points)-cal.camera_to_world_cv[:3,3])@cal.camera_to_world_cv[:3,:3]
            p=camera@cal.intrinsics.T
            return (p[:,:2]/p[:,2:]).tolist()
        outline=np.column_stack((radius*np.cos(angle),radius*np.sin(angle),np.zeros(12)))
        spec=dict(shape='circle',bottom_outline_px=project(outline),wall_height_m=1.5,
            metric_anchor=dict(kind='perpendicular_distance',distance_m=4.),
            up_camera=cal.camera_to_world_cv[2,:3].tolist(),sensor_family='tof',baseline_m=0.,
            check_spans=[dict(pixels=project([[-radius*.5,0,0],[radius*.5,0,0]]),length_m=radius)],
            confirmed_empty=True,horizontal_flat_bottom=True,vertical_walls=True,fixed_camera=True,rectified_aligned=True)
        rim_spec={**spec,'rim_outline_px':project(outline+np.array([0,0,1.5])),
                  'measured_rim_dimensions_m':[radius*2,radius*2]}
        bundle=build_scene_geometry(dry,cal.intrinsics,spec,rim_spec=rim_spec)
        prior_error=abs(bundle['bottom_plane_camera']['perpendicular_distance_m']-4.)
        if prior_error>1e-6:raise RuntimeError('ideal_empty_metric_anchor_verification_failed')
        item=dict(bundle=bundle,audit=dict(mode='ideal_simulated_independent_empty_ruler',
                 bottom_distance_error_mm=prior_error*1000,quality=bundle['quality']))
        profile=folder/'portable-profile.json';profile.write_text(json.dumps(validated_profile(ROOT)))
        physical=dict(seed=202610191919,index=1.333,extinction=.045 if material.endswith('translucent') else 0.,
                      scatter_strength=.16 if material.endswith('translucent') else 0.,surface_ratio=.07)
        phase_profile=dict(bottom_m=4.,container_max_height_m=1.5,fx_px=470.94,center_q=[0.,0.],
                           frequencies_hz=[20e6,30e6,50e6],fixed_camera=True,sensor_profile='amcw_itof_phase3_formal_v1')
        known=np.array([0.,.1,.35,.65,.95])*1.5
        captures=[]
        for j,h in enumerate(known):
            c,roi=generate('tof',phase_profile,physical,float(h),j);captures.append(c)
        phase=dict(profile=phase_profile,ray_coordinates=rays(phase_profile),roi=roi,
                   prior_correlations=captures,known_heights_m=known)
        installation=ToFInstallation(name,item['bundle'],dry,confirmed_material=material,phase=phase,
             provenance=dict(mode='simulation_empty_plus_independent_ruler_and_five_known_priors',
                             current_test_truth_used=False,field_calibration_requires_operator_measurements=True),
             confirmed_refractive_planar_unoccluded=material!='ordinary_diffuse')
        self.meter=UnifiedToFMeter(installation,self.classifier,profile_path=profile,
                                  compound_repair=self.compound_repair)
        preload_ms=self.meter.preload_optional_peak_models() if material=='ordinary_diffuse' else 0.
        warmup_ms=self.meter.warm_optional_peak_models() if material=='ordinary_diffuse' and self.compound_repair else 0.
        # Fit phase calibration at initialization, not inside the reported warm inference latency.
        if material!='ordinary_diffuse':
            from .tof_scene_meter import ToFSceneMeter
            self.meter.phase_cache[material]=ToFSceneMeter.from_phase_priors(sensor='tof',scene=material,**phase)
        self.initialization=dict(id=name,folder=str(folder),audit=item['audit'],material=material,
             phase_known_heights_m=known.tolist(),phase_calibration_precomputed=material!='ordinary_diffuse',
             simulated_ruler_is_exact=True,current_truth_used_for_measurement=False,
             optional_peak_asset_preload_ms=preload_ms,optional_peak_assets_preloaded=material=='ordinary_diffuse')
        self.initialization['compound_repair_requested']=self.compound_repair
        self.initialization['compound_repair_enabled']=self.meter.compound_repair is not None
        self.initialization['optional_peak_kernel_warmup_ms']=warmup_ms
        self.initialization['warmup_predictions_used_as_measurement_or_calibration']=False
        (folder/'geometry.json').write_text(json.dumps(item['bundle']))
        self.forward_physical,self.forward_profile=physical,phase_profile

    def run(self,job,controls,initialize):
        begin=time.perf_counter()
        with self.lock:self.jobs[job]['status']='running';self.state['status']='simulating'
        try:
            if initialize or self.meter is None or self.initialization['material']!=controls['material']:
                self.initialize(controls['material'])
            self.counter+=1;folder=self.work/f'frame-{self.counter:06d}-{job[:6]}'
            request=dict(**controls,seed=202610191901,index=self.counter,bottom_m=4.,capacity_m=1.5,radius_m=.55)
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
            observation=ToFObservation(job,rgb,depth,time.time(),correlations,
                                       installation_id=self.meter.installation.installation_id)
            result=self.meter.measure(observation)
            # Only AFTER inference, load current truth for display/scoring.
            truth=json.loads((folder/'truth.json').read_text())
            actual={k:truth[k] for k in ('liquid_height_m','highest_height_m','lowest_height_m')}
            estimate=dict(liquid_height_m=result.get('liquid_depth_m'),highest_height_m=result.get('highest_height_m'))
            from .empty_geometry_v41 import height_calibration
            from .liquid_height_metrics import world_points
            xyz=world_points(depth,height_calibration(self.meter.installation.geometry))
            good=(depth>0)&self.meter.roi
            yy,xx=np.nonzero(good);stride=max(1,len(yy)//5500);yy,xx=yy[::stride],xx[::stride]
            points=xyz[yy,xx].tolist();colors=rgb[yy,xx,::-1].tolist()
            _,encoded=cv2.imencode('.jpg',rgb,[cv2.IMWRITE_JPEG_QUALITY,84])
            payload=dict(status='ready',frame_id=job,controls=controls,actual=actual,estimate=estimate,
                result=plain(result),initialization=self.initialization,
                viewer=dict(meshes=truth['meshes'],observed_points=points,observed_colors=colors,radius_m=.55,capacity_m=1.5),
                rgb='data:image/jpeg;base64,'+base64.b64encode(encoded).decode(),
                timing=dict(render_ms=render_ms,pipeline_ms=result['pipeline_ms'],total_ms=(time.perf_counter()-begin)*1000),
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


def serve(console,port):
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
                with console.lock:return self.send(dict(state=console.state,jobs=console.jobs,csrf=console.token))
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
    print('SCENE_CONSOLE_READY',json.dumps(dict(port=server.server_port,binding='127.0.0.1')),flush=True)
    server.serve_forever()


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--compound-repair',action='store_true',default=True,help='Formal v2 evidence repair (default)')
    p.add_argument('--legacy-geometry',dest='compound_repair',action='store_false',help='Diagnostic old numeric path; not the selected release')
    p.add_argument('--blender',default='/root/autodl-tmp/tools/blender/blender-4.2.0-linux-x64/blender')
    a=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    console=Console(a.work,a.blender,a.checkpoint,compound_repair=a.compound_repair);serve(console,a.port)


if __name__=='__main__':main()
