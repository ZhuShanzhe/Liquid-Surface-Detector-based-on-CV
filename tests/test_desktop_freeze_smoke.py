import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def harness():
    path=Path(__file__).resolve().parents[1]/'scripts/desktop_freeze_smoke.py'
    spec=importlib.util.spec_from_file_location('desktop_freeze_smoke_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_self_test_always_shuts_down_worker_and_writes_failure_evidence(tmp_path):
    module=harness()
    calls=[]
    class Console:
        render_engine='cycles_cpu'
        initialization=None
        worker=SimpleNamespace(shutdown=lambda **kwargs:calls.append(kwargs))
        def submit(self,controls):raise RuntimeError('intentional_capture_error')
    args=SimpleNamespace(work=tmp_path/'work',output=tmp_path/'result.json',quick=True,blender=None)
    code=module.run(args,version='test',configure_runtime=lambda:tmp_path,
                    make_console=lambda *args:Console(),user_data_root=lambda:tmp_path)
    data=json.loads(args.output.read_text())
    assert code==1 and not data['success'] and not data['complete']
    assert calls==[dict(wait=True,cancel_futures=True)]
    assert data['thread_pool_shutdown'] and data['fatal_error']['exception_type']=='RuntimeError'
    assert Path(data['work']).is_absolute() and Path(data['output']).is_absolute()


def test_self_test_startup_failure_still_writes_diagnostics(tmp_path):
    module=harness()
    def fail():raise RuntimeError('missing bundled resource')
    args=SimpleNamespace(work=tmp_path/'work',output=tmp_path/'result.json',quick=False,blender=None)
    assert module.run(args,version='test',configure_runtime=fail,
        make_console=lambda *args:None,user_data_root=lambda:tmp_path)==1
    data=json.loads(args.output.read_text())
    assert data['cases']==[] and data['resource_root'] is None
    assert 'missing bundled resource' in data['fatal_error']['error']


def test_primary_matrix_has_twelve_cases_separate_from_integration():
    module=harness()
    assert len(module.SCENARIOS)==12
    assert len({name for name,_,_ in module.SCENARIOS})==12
    for name,controls,changed in module.SCENARIOS:
        assert name not in ('reject_all_depth_missing','reject_absent_phase_surface')
        assert isinstance(changed,bool)


def test_summary_distinguishes_operational_outputs_from_accuracy():
    module=harness()
    result=module.summarize([dict(eligible_numeric=True,liquid_pass=False,functional_pass=False,
        liquid_error_mm=50.,timing=dict(pipeline_ms=30.,render_ms=2000.,total_ms=2030.,initialization_ms=0.))])
    assert result['eligible_numeric']==1 and result['functional_pass']==0
    assert result['liquid_absolute_error_mm']['max']==50.
    assert result['end_to_end_without_reinitialization_ms']['mean']==2030.


def test_windowed_none_standard_streams_are_temporarily_logged(tmp_path,monkeypatch):
    module=harness()
    args=SimpleNamespace(work=tmp_path/'work',output=tmp_path/'result.json',quick=True,blender=None)
    def startup():
        print('windowed runtime stdout')
        sys.stderr.write('windowed runtime stderr\n')
        sys.stdout.flush();sys.stderr.flush()
        raise RuntimeError('intentional runtime failure')
    with monkeypatch.context() as changed:
        changed.setattr(sys,'stdout',None);changed.setattr(sys,'stderr',None)
        assert module.run(args,version='test',configure_runtime=startup,
            make_console=lambda *args:None,user_data_root=lambda:tmp_path)==1
        assert sys.stdout is None and sys.stderr is None
    result=json.loads(args.output.read_text())
    log=Path(result['windowed_stdio_fallback_log'])
    assert 'windowed runtime stdout' in log.read_text() and 'windowed runtime stderr' in log.read_text()
