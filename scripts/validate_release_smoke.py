"""Check portable packaging against the unchanged archived public CLI outputs."""
import json,os,subprocess,sys,time
from pathlib import Path

repo=Path('/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV')
base=Path('/root/autodl-tmp/liquid-depth-artifacts/determined-all-scenes-v1-20261016/no-usb-example')
out=Path('/root/autodl-tmp/liquid-depth-artifacts/publication-20261016')
env=dict(os.environ,PYTHONPATH='src:scripts',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
checks=[]
for file in sorted(base.glob('*/command.json')):
    command=json.loads(file.read_text());command[0]=sys.executable;command[1]='scripts/measure_release.py'
    start=time.monotonic()
    proc=subprocess.run(command,cwd=repo,env=env,capture_output=True,text=True,check=True)
    value=json.loads(proc.stdout);old=json.loads((file.parent/'output.json').read_text())
    scene=file.parent.name
    for key in ('liquid_depth_m','highest_height_m'):
        if old.get(key) is None:assert value.get(key) is None,(scene,key)
        else:assert abs(value[key]-old[key])<1e-7,(scene,key,value[key],old[key])
    assert not value['liquid_visual_guidance_used']
    assert value['visual_module_executed']==(scene=='floating_objects')
    checks.append(dict(scene=scene,matched_archived_cli=True,wall_s=time.monotonic()-start,
                       liquid_depth_m=value['liquid_depth_m'],highest_height_m=value.get('highest_height_m'),
                       vision_executed=value['visual_module_executed']))
    if scene=='floating_objects':
        command.remove('--enable-floating-peak-vision')
        off=json.loads(subprocess.run(command,cwd=repo,env=env,capture_output=True,text=True,check=True).stdout)
        assert not off['visual_module_executed']
        assert off['liquid_depth_m']==value['liquid_depth_m']
assert len(checks)==8
(out/'portable-cli-smoke.json').write_text(json.dumps(checks,indent=2))
print('PORTABLE_CLI_SMOKE_OK',len(checks))
