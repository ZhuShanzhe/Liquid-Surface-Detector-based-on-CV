"""Frozen V14, independent diverse holdout, paired empty calibration."""
import collections,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path


def fresh_parity_diagnostic(row, observation, factory, synchronize):
    """Never mutate a valid primary result; diagnostics have a separate status."""
    fresh = None
    try:
        fresh = factory()
        again = fresh.measure(observation)
        synchronize()
        def equal(x, y):
            return (x is None and y is None) or (x is not None and y is not None and abs(float(x)-float(y)) < 1e-7)
        check = dict(frame=row['frame'], variant=row['calibration_variant'],
                     liquid_equal=equal(row['liquid'], again.get('liquid_depth_m')),
                     highest_equal=equal(row['highest'], again.get('highest_height_m')),
                     tags_equal=row['classification']['tags'] == again['classification']['tags'])
        if not all(check[key] for key in ('liquid_equal','highest_equal','tags_equal')):
            return check, dict(frame=row['frame'], variant=row['calibration_variant'],
                               reason='fresh_parity_mismatch', check=check)
        return check, None
    except Exception as exc:
        return None, dict(frame=row['frame'], variant=row['calibration_variant'],
                          reason='fresh_parity_diagnostic_exception', error=repr(exc),
                          exception_type=type(exc).__name__, traceback=traceback.format_exc())
    finally:
        del fresh


def parity_self_test():
    row=dict(frame='fixture',calibration_variant='ideal',liquid=.5,highest=.6,classification=dict(tags=[]))
    original=json.loads(json.dumps(row))
    class Fake:
        def measure(self, observation):
            return dict(liquid_depth_m=.5,highest_height_m=.6,classification=dict(tags=[]))
    check,failure=fresh_parity_diagnostic(row,None,Fake,lambda:None)
    assert failure is None and check['liquid_equal'] and row==original
    def fail_factory():raise RuntimeError('fresh-only failure')
    check,failure=fresh_parity_diagnostic(row,None,fail_factory,lambda:None)
    assert check is None and failure['reason']=='fresh_parity_diagnostic_exception' and row==original
    class Different(Fake):
        def measure(self, observation):return {**super().measure(observation),'liquid_depth_m':.8}
    check,failure=fresh_parity_diagnostic(row,None,Different,lambda:None)
    assert failure['reason']=='fresh_parity_mismatch' and not check['liquid_equal'] and row==original
    print('PARITY_DIAGNOSTIC_SELF_TEST_OK')


if '--self-test-parity' in sys.argv:
    parity_self_test();sys.exit(0)

from range_protocol import ROOT,BANDS,QUOTAS,band_of
REPO=Path('/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV')
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import cv2,numpy as np,torch
from threadpoolctl import threadpool_limits
from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.unified_tof import ToFInstallation,ToFObservation
from liquid_depth.formal_tof_runtime import create_formal_meter
from liquid_depth.tof_scene_meter import ToFSceneMeter
from liquid_depth.sim_input_qualification import qualify_simulated_input,measure_if_eligible
from measure_release import validated_profile
from tof_phase_simulator import generate,rays
from diverse_geometry import make_bundle,phase_inverse_inputs
TAGS=('low_light','glare','uneven_surface','floating_objects');SMOKE='--smoke' in sys.argv
OUT=ROOT/'smoke' if SMOKE else ROOT;OUT.mkdir(exist_ok=True)
frozen=json.loads((ROOT.parent/'formal-v14-compounds-20261021/frozen-formal-source.json').read_text())
def verify():
    for name,digest in frozen['source_sha256'].items():assert hashlib.sha256((REPO/'src/liquid_depth'/name).read_bytes()).hexdigest()==digest,name
verify();torch.set_num_threads(2);cv2.setNumThreads(1);guard=threadpool_limits(limits=2)
profile=OUT/'portable-profile.json';profile.write_text(json.dumps(validated_profile(REPO)))
protocol=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
 evaluation_revision='range_0p1_9m_fixed_10mm',measurement_range_m=[.1,9.],calibration_sigma_m=.01/3,
 source_sha256=frozen['source_sha256'],model_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (REPO/'assets/models').rglob('*') if p.is_file()},
 algorithm_changed=False,qualification_frozen_before_inference=True,exact_tags_gate=False,field_metrology_qualified=False,
 calibration_variants=['ideal','operator_noise'],primary_variant='operator_noise',minimum_qualified_per_category=100,
 generation_seed='reused immutable old inputs; new geometry 20261021170039+gid*104729; scene 20261021190011+category*10000019+attempt*99991',
 geometry=dict(shapes=['cylindrical','elliptical','rectangular_tank'],radius_m=[.3,1.5],aspect=[.65,1.45],cylinder_aspect=1.,
 camera_bottom_m=[.9,10.],camera_xy_fraction_radius=[-.22,.22],fov_deg=[45,100],capacity='new H+0.15..0.5 m, upper 8..9 m band H+0.08..0.12 m; below camera; reused prior geometry unchanged',
 sampling='prespecified depth-band quotas [12,11,11,11,11,11,11,11,11], range .1..9; sample H first then feasible geometry; full rim visibility not required',
 new_radius_policy='draw from intersection [0.3,1.5] and feasible rim FOV; never shrink an existing container; input visibility independently checked',bands_m=BANDS,quotas=QUOTAS),
 operator_calibration=dict(seed='2026102197701+gid*9973',error_limit_m=.01,sigma_m=.01/3,
 distribution='zero-mean Gaussian sigma=T/3, rejection-sampled within [-T,+T]',applies_to='bottom distance, independently measured geometry dimensions, four known liquid-height priors',
 empty_prior='confirmed empty H=0; datum uncertainty carried by bottom anchor, not invented nonzero liquid in empty frame',
 click_sigma_px=.3,gravity_component_sigma=.001,max_reannotation_attempts=6,phase_label_max_reannotation_attempts=100,reannotation_gate='geometry/prior input validity only, never output error'),
 sensor_sensitivity=dict(families=['base iid','plus spatial','plus frame common drift','plus both'],
 common_sigma_m='0.001+0.0005*(B-H)',spatial_each_of_four_sigma_m='0.0006+0.00015*(B-H)',
 definition='additional frame-specific drift relative to empty acquisition; not installation-fixed bias, not measured hardware parameters',
 rgb_ir_decoupled=True,glare_rgb='three-light glare renderer, primary gain .7+3.3*glare, source size .38-.27*glare, liquid roughness=max(.008,.22*(1-glare))',glare_depth_loss='local ellipse 8-30 percent visible bbox radii; dropout probability .25+.65*glare',
 legacy_bug_reintroduced=False),
 phase_parameters=dict(index=[1.31,1.45],surface_ratio=[.03,.12],translucent_extinction=[.02,.08],translucent_scatter=[.08,.25],
 priors_fraction=[0,.1,.35,.65,.95],phase_shape=[3,4,48,64],roi='central native 64x48 crop intersects visible empty floor and analytic acquisition ROI',
 unknown_physical_parameters_given_to_inverse=False,limitation='separate near-vertical planar analytic phase model, not coherent spectral Blender ray tracing'),
 qualification='input-only simulator return provenance: >=30 genuine liquid points and span>=3px; invalid<=50%; highest instance >=3 genuine points; phase finite support>=30 and preflight common-prior/current selected modulation support>=64 in actual empty projected ROI',
 timing='serial warmed qualification+classification+measurement with CUDA sync; excludes render, IO, per-install initialization, network, UI',
 liquid_tolerance='max(20mm,2% true bottom-relative depth); actual uneven max(50mm,5%)',highest_tolerance='max(50mm,5% true highest)',
 interpretation='depth-band-balanced 0.1..9 m simulation with fixed 10 mm calibration budget, not field accuracy or manufacturer specification')
selection=json.loads((ROOT/'selection-manifest.json').read_text())
protocol['selection_provenance']={**selection,'categories':[{k:v for k,v in x.items() if not k.endswith('_frames')} for x in selection['categories']]}
protocol['selection_manifest_sha256']=hashlib.sha256((ROOT/'selection-manifest.json').read_bytes()).hexdigest()
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
if '--protocol-only' in sys.argv:print('PROTOCOL_FROZEN');sys.exit(0)
states=[json.loads(p.read_text()) for p in sorted(ROOT.glob('capture-??.json'))]
if not SMOKE:
    assert len(states)==25 and all(len(s['qualified'])==100 for s in states)
    for state in states:
        assert [sum(x['depth_band']==j for x in state['qualified']) for j in range(9)]==QUOTAS
items=[(s,x) for s in states for x in s['qualified']]
items.sort(key=lambda z:(z[1]['installation_id'],z[0]['category_id'],z[1]['attempt']))
if SMOKE:items=[next((s,x) for s,x in items if s['category_id']==i) for i in [0,8,12,13] if any(s['category_id']==i for s,x in items)]
protocol['generated_single_target']=sum(s['cohort']=='single' for s,x in items);protocol['generated_composite_target']=sum(s['cohort']=='composite' for s,x in items)
protocol['evaluation_provenance']=dict(reused_input_images=sum(x['source']=='reused_input' for s,x in items),
    new_input_images=sum(x['source']=='new_input' for s,x in items),unique_input_images=len(items),new_measurement_runs=2*len(items),reused_measurement_runs=0,
    old_results_policy='no old measured values reused; Gaussian calibration distribution and sequence changed',attempts_includes_reused=True)
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));(OUT/'evaluated-inputs.json').write_text(json.dumps([dict(frame=x['frame'],category=s['category'],installation_id=x['installation_id']) for s,x in items]))
counts=collections.Counter();current_gid=None;meters={};geometry={};phase_by_material={};inits={};parity=[];diagnostic_failures=[]
for marker in ('measurements.complete','report-ready.complete'):
    (OUT/marker).unlink(missing_ok=True)
with (OUT/'measurements.jsonl').open('w') as stream:
 for idx,(state,item) in enumerate(items):
    gid=item['installation_id'];folder=Path(item['frame']);req=json.loads((folder/'request.json').read_text());material=req['material'];bits=state['bits']
    if gid!=current_gid:meters.clear();geometry.clear();phase_by_material.clear();inits.clear();current_gid=gid
    rgb=cv2.imread(str(folder/'rgb.png'));raw=np.load(folder/'depth_m.npy');valid=np.load(folder/'capture_valid.npy');sem=np.load(folder/'offline_geometry.npz')['semantic']
    phase=None;correlations=None
    if material!='ordinary_diffuse':
        if material not in phase_by_material:
            rng=np.random.default_rng(2026102183731+gid*7727+(material.endswith('translucent')))
            physical=dict(seed=2026102183731+gid*7727+(material.endswith('translucent')),index=float(rng.uniform(1.31,1.45)),surface_ratio=float(rng.uniform(.03,.12)),extinction=float(rng.uniform(.02,.08)) if material.endswith('translucent') else 0.,scatter_strength=float(rng.uniform(.08,.25)) if material.endswith('translucent') else 0.)
            empty=ROOT/'empty'/f'{gid:04d}';cal=HeightCalibration.from_simulation(json.loads((empty/'metadata.json').read_text()))
            pp=dict(bottom_m=req['bottom_m'],container_max_height_m=req['capacity_m'],fx_px=float(cal.intrinsics[0,0]),center_q=[0.,0.],frequencies_hz=[20e6,30e6,50e6],fixed_camera=True,sensor_profile='amcw_itof_phase3_formal_v1')
            known=np.array([0,.1,.35,.65,.95])*req['capacity_m'];captures=[]
            for j,h in enumerate(known):c,roi=generate('tof',pp,physical,float(h),j);captures.append(c)
            roi=roi&(np.load(empty/'offline_geometry.npz')['semantic'][111:159,208:272]==2)
            phase_by_material[material]=(dict(profile=pp,ray_coordinates=rays(pp),roi=roi,prior_correlations=captures,known_heights_m=known),physical)
        phase,physical=phase_by_material[material];correlations,_=generate('tof',phase['profile'],physical,req['height_m'],1001+item['attempt'])
        correlations=correlations.copy();correlations[:,:,(sem[111:159,208:272]!=2)]=1200
        np.savez_compressed(folder/'phase_observations.npz',priors=np.asarray(phase['prior_correlations']),current=correlations,roi=phase['roi'],known_heights_m=phase['known_heights_m'])
    qualification=dict(item['qualification'])
    if phase is not None:
        pq=qualify_simulated_input(raw,sem,material=material,correlations=correlations,phase_roi=phase['roi'])
        qualification.update(pq)
    for variant in ('ideal','operator_noise'):
        row=dict(frame=str(folder),installation_id=gid,group=gid,calibration_variant=variant,cohort=state['cohort'],category=state['category'],bits=bits,sensor='tof',target_tags=[t for j,t in enumerate(TAGS) if bits&(1<<j)],controls=req,qualification=qualification,inference_executed=False,liquid=None,highest=None,wall_ms=None,initialization_ms=None,classification=dict(tags=[]),field_metrology_qualified=False)
        row.update(input_source=item['source'],depth_band=item['depth_band'],source_frame=item.get('source_frame'),preflight_phase_initialization=item.get('preflight_phase_initialization'))
        if phase is not None:
            row['phase_test_above_highest_prior']=bool(req['height_m']>max(phase['known_heights_m']))
        if qualification['eligible']:
            try:
                begin=time.perf_counter();key=(variant,material)
                was_created=key not in meters
                if variant not in geometry:geometry[variant]=make_bundle(ROOT,gid,variant)
                bundle,dry,audit=geometry[variant];row['calibration_audit']=audit
                if key not in meters:
                    ip=phase
                    if phase is not None:
                        ip,pa=phase_inverse_inputs(phase,audit,gid,material,variant)
                        row['phase_prior_annotation']=pa
                    inst=ToFInstallation(f'diverse-{gid}-{variant}-{material}',bundle,dry,confirmed_material=material,phase=ip,provenance=dict(mode=variant,current_test_truth_used=False),confirmed_refractive_planar_unoccluded=phase is not None)
                    meter=create_formal_meter(inst,profile_path=profile)
                    if ip is not None:meter.phase_cache[material]=ToFSceneMeter.from_phase_priors(sensor='tof',scene=material,**ip)
                    meters[key]=meter;torch.cuda.synchronize();inits[key]=(time.perf_counter()-begin)*1000
                meter=meters[key];row['initialization_ms']=inits[key] if was_created else None;row['initialization_reused']=not was_created
                obs=ToFObservation(folder.name,rgb,raw,float(idx),correlations,installation_id=meter.installation.installation_id,capture_validity=valid)
                torch.cuda.synchronize();start=time.perf_counter();row['inference_executed']=True
                checked=qualify_simulated_input(raw,sem,material=material,has_floats=bool(bits&8),correlations=correlations,phase_roi=None if phase is None else phase['roi'],minimum_depth_m=.08 if phase is None else 0.)
                assert checked['eligible'],'capture_qualification_disagrees_with_runtime_input_check'
                result=measure_if_eligible(meter,obs,qualification);torch.cuda.synchronize()
                row.update(liquid=result.get('liquid_depth_m'),highest=result.get('highest_height_m'),wall_ms=(time.perf_counter()-start)*1000,classification=result['classification'],module_plan=result['module_plan'],formal_runtime=meter.formal_runtime,diagnostics=result.get('diagnostics',{}))
                if (gid<3 and req['repeat']==1 or gid>=10000 and gid%1000==0) and state['category_id'] in (0,3,6,8,12,13):
                    row['fresh_parity_checked']=True
                    check,diagnostic=fresh_parity_diagnostic(
                        row,obs,lambda:create_formal_meter(meter.installation,profile_path=profile),torch.cuda.synchronize)
                    if check is not None:parity.append(check)
                    if diagnostic is not None:
                        diagnostic_failures.append(diagnostic)
                        row['diagnostic_error']=diagnostic
            except Exception as exc:row.update(error=repr(exc),traceback=traceback.format_exc())
        if row.get('fresh_parity_checked'):
            # Audit-file IO failure is not a primary measurement exception;
            # fail the run instead of retaining an ambiguous row.error.
            (OUT/'meter-reuse-parity.json').write_text(json.dumps(parity,indent=2))
            (OUT/'diagnostic-failures.json').write_text(json.dumps(diagnostic_failures,indent=2))
        truth=json.loads((folder/'truth.json').read_text());row['truth']={k:truth[k] for k in ('liquid_height_m','highest_height_m','lowest_height_m')}
        assert band_of(row['truth']['liquid_height_m'])==item['depth_band'],'out-of-protocol liquid depth must be resolved before evaluation'
        stream.write(json.dumps(row,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n');stream.flush();counts[variant+':'+state['category']]+=1
    (OUT/'measurement-progress.json').write_text(json.dumps(dict(completed=(idx+1)*2,total=len(items)*2,per_category=dict(counts))))
    print('DIVERSE',idx+1,len(items),state['category'],row.get('error',''),flush=True)
verify()
assert not diagnostic_failures,'fresh-parity diagnostic failures: primary results retained, full evaluation not complete'
assert all(r['liquid_equal'] and r['highest_equal'] and r['tags_equal'] for r in parity),'meter reuse changes results'
(OUT/'measurements.complete').write_text('true')
