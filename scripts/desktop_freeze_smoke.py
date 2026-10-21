"""Real frozen/source desktop acceptance, plus explicitly labelled fault injection.

The 12 primary cases use real Blender captures and the formal meter. Lifecycle
and fault cases are separate software-integration checks, not accuracy samples.
No extra case is discarded because of its numerical error.
"""
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np


SCENARIOS = [
    ('ordinary', {}, True),
    ('ellipse', dict(container_shape='ellipse',container_radius_x_m=.6,
                     container_radius_y_m=.45,camera_offset_x_m=.05), True),
    ('rectangle', dict(container_shape='rectangle',container_radius_x_m=.6,
                       container_radius_y_m=.5,camera_offset_y_m=-.04), True),
    ('low_light', dict(darkness=.8), True),
    ('glare', dict(glare=.8), False),
    ('loss', dict(loss=.35), False),
    ('waves', dict(wave_m=.04), False),
    ('floating', dict(float_count=5), False),
    ('compound', dict(darkness=.6,glare=.6,wave_m=.025,float_count=5), False),
    ('transparent', dict(material='homogeneous_transparent'), True),
    ('translucent', dict(material='homogeneous_translucent'), True),
    ('high_level', dict(height_m=9.,container_height_m=9.1,camera_to_bottom_m=10.,
                       container_radius_x_m=.3,container_radius_y_m=.3), True),
]


def await_case(console, name, controls, expected_reinitialized, *, timeout_s=600.):
    previous=(console.initialization or {}).get('id')
    job=console.submit(controls)
    deadline=time.monotonic()+timeout_s
    while console.jobs[job]['status'] in ('queued','running'):
        if time.monotonic()>deadline:
            raise TimeoutError('self_test_case_timeout:'+name)
        time.sleep(.15)
    state=console.state
    row=dict(name=name,job=dict(console.jobs[job]),status=state['status'],
             previous_installation_id=previous,expected_reinitialized=expected_reinitialized)
    if state['status']!='ready':
        row.update(message=state.get('message'),eligible_numeric=False,lifecycle_pass=False)
        return row
    row.update(controls=state['controls'],qualification=state['input_qualification'],
        estimate=state['estimate'],actual=state['actual'],timing=state['timing'],
        rendering=state.get('rendering'),validation_envelope=state.get('validation_envelope'),
        initialization=state['initialization'],result=state['result'])
    current=state['initialization']['id']
    reinitialized=current!=previous
    row.update(installation_id=current,reinitialized=reinitialized,
        lifecycle_pass=reinitialized==expected_reinitialized)
    h,truth=state['estimate']['liquid_height_m'],state['actual']['liquid_height_m']
    row['liquid_error_mm']=None if h is None else abs(h-truth)*1000
    row['liquid_signed_error_mm']=None if h is None else (h-truth)*1000
    row['liquid_relative_error']=None if h is None or truth==0 else abs(h-truth)/abs(truth)
    row['eligible_numeric']=state['input_qualification']['eligible'] and h is not None
    liquid_tol=max(.05,.05*truth) if state['controls']['wave_m']>0 else max(.02,.02*truth)
    row['liquid_tolerance_m']=liquid_tol
    row['liquid_pass']=h is not None and abs(h-truth)<=liquid_tol
    highest_needed=state['controls']['wave_m']>0 or state['controls']['float_count']>0
    peak,truth_peak=state['estimate']['highest_height_m'],state['actual']['highest_height_m']
    row['highest_required']=highest_needed
    row['highest_tolerance_m']=max(.05,.05*truth_peak) if highest_needed else None
    row['highest_error_mm']=None if peak is None else abs(peak-truth_peak)*1000
    row['highest_pass']=(peak is not None and abs(peak-truth_peak)<=row['highest_tolerance_m']) if highest_needed else None
    row['functional_pass']=row['liquid_pass'] and (not highest_needed or row['highest_pass'])
    return row


def fault_case(console, name, controls, *, phase=False):
    """Real capture, then explicit observation fault; spy the public meter call.

    This deliberately corrupted frame is NOT included in the 12 accuracy cases.
    The test verifies Console.run's actual pre-inference branch, not only a helper.
    """
    if console.meter is None:
        raise RuntimeError('fault_case_requires_existing_valid_installation')
    original_capture=console.capture
    meter=console.meter
    original_measure=meter.measure
    calls=[]
    def forbidden_measure(observation):
        calls.append(observation.frame_id)
        raise AssertionError('unqualified_input_reached_formal_meter')
    def fault_capture(request,folder):
        original_capture(request,folder)
        if phase:
            path=folder/'offline_geometry.npz'
            with np.load(path,allow_pickle=False) as source:
                arrays={key:source[key] for key in source.files}
            # No visible current liquid in the phase crop: producer generates
            # zero-modulation capture there before the measurement contract.
            arrays['semantic']=np.zeros_like(arrays['semantic'])
            np.savez_compressed(path,**arrays)
        else:
            raw=np.load(folder/'depth_m.npy',allow_pickle=False)
            np.save(folder/'depth_m.npy',np.zeros_like(raw))
            np.save(folder/'capture_valid.npy',np.zeros(raw.shape,bool))
        (folder/'test_fault_injection.json').write_text(json.dumps(dict(
            integration_test_only=True,not_a_performance_sample=True,
            fault='no_visible_phase_surface' if phase else 'all_depth_returns_absent')))
    console.capture=fault_capture
    meter.measure=forbidden_measure
    try:
        row=await_case(console,name,controls,False)
        result=row.get('result',{})
        qualification=row.get('qualification',{})
        passed=(row['status']=='ready' and not qualification.get('eligible',True)
                and result.get('inference_executed') is False
                and row.get('estimate',{}).get('liquid_height_m') is None
                and not calls and console.meter is meter)
        return dict(name=name,integration_test_only=True,not_a_performance_sample=True,
            passed=passed,measurement_call_count=len(calls),meter_unchanged=console.meter is meter,
            status=row['status'],qualification=qualification,inference_executed=result.get('inference_executed'),
            estimate=row.get('estimate'),timing=row.get('timing'),message=row.get('message'),
            job=row['job'])
    finally:
        console.capture=original_capture
        meter.measure=original_measure


def summarize(rows):
    def statistics(values):
        a=np.asarray([v for v in values if v is not None],float)
        if not len(a):return dict(n=0)
        return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
                    p95=float(np.percentile(a,95)),max=float(a.max()))
    return dict(primary_cases=len(rows),eligible_numeric=sum(r.get('eligible_numeric',False) for r in rows),
        liquid_pass=sum(r.get('liquid_pass',False) for r in rows),
        functional_pass=sum(r.get('functional_pass',False) for r in rows),
        liquid_absolute_error_mm=statistics([r.get('liquid_error_mm') for r in rows]),
        highest_absolute_error_mm=statistics([r.get('highest_error_mm') for r in rows if r.get('highest_required')]),
        inference_ms=statistics([r.get('timing',{}).get('pipeline_ms') for r in rows]),
        render_and_capture_ms=statistics([r.get('timing',{}).get('render_ms') for r in rows]),
        end_to_end_without_reinitialization_ms=statistics([r.get('timing',{}).get('total_ms') for r in rows
                                                          if r.get('timing',{}).get('initialization_ms')==0]),
        scope='12-case portability smoke, not statistical accuracy validation or published GPU report')


def run(args, *, version, configure_runtime, make_console, user_data_root):
    """Windowed PyInstaller executables may start with both streams set to None.

    Give third-party runtime imports and background exception reporting a real
    UTF-8 log stream, then restore only streams that this harness replaced.
    """
    work=Path(args.work or user_data_root()/('self-test-'+time.strftime('%Y%m%d-%H%M%S'))).resolve()
    from copy import copy
    active_args=copy(args);active_args.work=work
    work.mkdir(parents=True,exist_ok=True)
    original_stdout,original_stderr=sys.stdout,sys.stderr
    fallback=None
    if original_stdout is None or original_stderr is None:
        fallback=(work/'self-test-console.log').open('a',encoding='utf-8',buffering=1)
        if original_stdout is None:sys.stdout=fallback
        if original_stderr is None:sys.stderr=fallback
    try:
        return _run_with_stdio(active_args,version=version,configure_runtime=configure_runtime,
            make_console=make_console,user_data_root=user_data_root,
            stdio_log=str(work/'self-test-console.log') if fallback else None)
    finally:
        if original_stdout is None:sys.stdout=original_stdout
        if original_stderr is None:sys.stderr=original_stderr
        if fallback is not None:fallback.close()


def _run_with_stdio(args, *, version, configure_runtime, make_console, user_data_root,stdio_log):
    from liquid_depth.scene_console_web import plain,validate_controls
    work=Path(args.work or user_data_root()/('self-test-'+time.strftime('%Y%m%d-%H%M%S'))).resolve()
    output=Path(args.output or work/'desktop-self-test.json').resolve()
    work.mkdir(parents=True,exist_ok=True);output.parent.mkdir(parents=True,exist_ok=True)
    console=None;root=None;records=[];lifecycle=[];faults=[];rejected=[];fatal=None
    scenarios=SCENARIOS[:1] if args.quick else SCENARIOS
    try:
        root=configure_runtime()
        console=make_console(root,work,args.blender)
        for name,controls,expect_changed in scenarios:
            row=await_case(console,name,controls,expect_changed)
            records.append(row)
            if row['status']!='ready':continue
            if name=='ordinary':
                faults.append(fault_case(console,'reject_all_depth_missing',controls))
            if name=='transparent':
                faults.append(fault_case(console,'reject_absent_phase_surface',controls,phase=True))
                lifecycle.append(await_case(console,'optical_index_recalibration',
                    dict(material='homogeneous_transparent',refractive_index=1.37),True))
            if name=='translucent':
                lifecycle.append(await_case(console,'optical_scatter_recalibration',
                    dict(material='homogeneous_translucent',scatter_strength=.22),True))
            if name=='high_level':
                changed={**controls,'height_m':8.8}
                lifecycle.append(await_case(console,'height_only_reuses_calibration',changed,False))
                lifecycle.append(await_case(console,'ideal_mode_recalibration',
                    {**changed,'calibration_noise_mode':'ideal'},True))
                lifecycle.append(await_case(console,'fixed10mm_mode_recalibration',changed,True))
        for controls in [dict(material='homogeneous_transparent',float_count=1),dict(loss=.6),
                         dict(height_m=1.,container_height_m=.5),
                         dict(camera_to_bottom_m=1.,container_height_m=2.,auto_frame=False)]:
            try:validate_controls(controls)
            except ValueError:rejected.append(True)
            else:rejected.append(False)
    except Exception as exc:
        fatal=dict(error=repr(exc),exception_type=type(exc).__name__,traceback=traceback.format_exc())
    finally:
        if console is not None:
            console.worker.shutdown(wait=True,cancel_futures=True)
    complete=len(records)==len(scenarios) and fatal is None
    expected_faults=1 if args.quick else 2
    expected_lifecycle=0 if args.quick else 5
    integration_ok=(len(faults)==expected_faults and all(f['passed'] for f in faults)
        and len(lifecycle)==expected_lifecycle
        and all(r.get('lifecycle_pass',False) and r.get('eligible_numeric',False) for r in lifecycle)
        and all(r.get('lifecycle_pass',False) for r in records))
    result=dict(version=version,frozen=bool(getattr(sys,'frozen',False)),resource_root=None if root is None else str(root),
        work=str(work),output=str(output),network_required=False,python_version=platform.python_version(),
        windowed_stdio_fallback_log=stdio_log,
        render_engine=None if console is None else console.render_engine,
        invalid_controls_rejected=rejected,cases=records,integration_lifecycle=lifecycle,
        integration_faults=faults,summary=summarize(records),fatal_error=fatal,
        complete=complete,software_integration_success=integration_ok,
        success=complete and integration_ok and all(r.get('eligible_numeric',False) for r in records)
                and len(rejected)==4 and all(rejected),
        functional_success=complete and all(r.get('functional_pass',False) for r in records),
        success_definition='operational run, valid input outputs, lifecycle and pre-inference rejection; accuracy is functional_success separately',
        thread_pool_shutdown=True)
    output.write_text(json.dumps(plain(result),ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if result['success'] else 1
