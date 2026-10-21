"""Export a path-redacted receipt and checksums for an already-tested bundle."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--measurement', type=Path, required=True)
    parser.add_argument('--gui', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    measured = json.loads(args.measurement.read_text(encoding='utf-8'))
    gui = json.loads(args.gui.read_text(encoding='utf-8'))
    if not measured['frozen'] or not measured['success'] or not gui['frozen'] or not gui['success']:
        raise RuntimeError('Frozen executable acceptance must pass before making delivery receipt')
    rows = []
    for row in measured['cases']:
        rows.append({k: row.get(k) for k in ('name', 'controls', 'eligible_numeric',
            'liquid_error_mm', 'liquid_relative_error', 'liquid_tolerance_m', 'liquid_pass',
            'highest_required', 'highest_error_mm', 'highest_tolerance_m', 'highest_pass',
            'functional_pass', 'timing', 'lifecycle_pass')})
    receipt = dict(version=measured['version'],created_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        formal_baseline_commit='2d1a6e9ecc6a06eec29fa5450b44e9e7c4e720ed',
        platform='Windows 11 x64',cpu='Intel Core Ultra 9 285K (24 cores)',
        inference='PyTorch 2.11.0+cpu, CUDA hidden, four inference threads',
        renderer='Blender 4.2.0 Cycles CPU, four threads',
        environment='Desktop Chinese path; PATH limited to Windows/System32; PYTHONPATH/PYTHONHOME empty',
        exe_sha256=digest(bundle/'LiquidDepthSimulation.exe'),
        executable_acceptance_pass=measured['success'],functional_smoke_pass=measured['functional_success'],
        measurement_summary=measured['summary'],cases=rows,
        lifecycle_checks=[{k:r.get(k) for k in ('name','expected_reinitialized','reinitialized','lifecycle_pass','eligible_numeric')}
                          for r in measured['integration_lifecycle']],
        fault_checks=[{k:r.get(k) for k in ('name','passed','measurement_call_count','inference_executed','not_a_performance_sample')}
                      for r in measured['integration_faults']],
        invalid_control_checks=measured['invalid_controls_rejected'],
        gui_checks=gui['checks'],gui_javascript_errors=gui.get('js_console_errors',[]),
        limitations=['Portability smoke only, not a new statistical performance report',
                     'Cycles CPU RGB differs from the EEVEE/GPU published evaluation domain',
                     'No physical camera or lower-end PC qualification was performed',
                     'Unsigned research demonstration; no industrial metrology certification'])
    args.evidence.mkdir(parents=True,exist_ok=True)
    text=json.dumps(receipt,ensure_ascii=False,indent=2)+'\n'
    (args.evidence/'acceptance.json').write_text(text,encoding='utf-8')
    (bundle/'验收记录.json').write_text(text,encoding='utf-8')
    for screenshot in gui.get('screenshots',[]):
        shutil.copy2(screenshot['path'],args.evidence/(screenshot['tab']+'.png'))
    entries=[]
    for path in sorted(bundle.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name=='SHA256SUMS.txt':
            continue
        entries.append(f'{digest(path)}  {path.relative_to(bundle).as_posix()}')
    (bundle/'SHA256SUMS.txt').write_text('\n'.join(entries)+'\n',encoding='utf-8')
    print(json.dumps({'exe_sha256':receipt['exe_sha256'],'files_checksummed':len(entries),
                     'manifest_sha256':digest(bundle/'SHA256SUMS.txt'),
                     'evidence':str(args.evidence)},ensure_ascii=False))


if __name__=='__main__':
    main()
