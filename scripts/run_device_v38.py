"""Freeze then generate independent native tests; never mutate existing datasets."""
import hashlib,json,subprocess,sys,time,os
from pathlib import Path
from liquid_depth.device_profiles_v38 import PROFILES
REPO=Path(__file__).resolve().parents[1]
OUT=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/device-v38-test-20261012')
DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/device-v38-test-20261012')
BLENDER='/root/autodl-tmp/tools/blender/blender-4.2.0-linux-x64/blender'
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=False);DATA.mkdir(exist_ok=False)
    files=list((REPO/'src/liquid_depth').rglob('*.py'))+list((REPO/'scripts').glob('*.py'))+list((REPO/'tests').glob('*v3*.py'))
    files+=[Path('/root/autodl-tmp/liquid-depth-artifacts/training/universal-liquid-v8-ordinary-sensor-anchor/best.pth')]
    hashes={str(p):sha(p) for p in files};dump(OUT/'frozen-source.json',hashes)
    dump(OUT/'protocol.json',dict(seed=202610123899,profiles=[p.manifest() for p in PROFILES],native_rgb_scenes=660,
        expected_sensor_frames=6864,heights_m=[0,9],maximum_camera_to_bottom_m=10,prior_groups=5,uniform_bins=9,
        empty_and_near_zero_separate=True,main_population='uniform only; 3 independent scenes per metre per profile and scenario, four static repeat frames',
        no_training=True,gate='relaxed diagnostic outputs, no zero snap or liquid height clipping',
        noise_assumptions=dict(stereo_disparity_sigma_px=[.18,.36],stereo_shared_disparity_offset_px=[.025,.05],
          tof_temporal_sigma_m='.0008+.0012*max(radial_distance-5.5,0)^2; adverse x3',
          tof_near_common_offset_m=.005,tof_far_extra_slope_m_per_m=[.002,.012],
          adverse_spatial_shared_variance_fraction=.5,adverse_shared_map_static=True,
          material_dropout_wrong_echo='legacy scenario proxy, not vendor liquid characterization',
          infrared_darkness='visible low light alone does not create additional depth dropout'),
        ideal_calibration='known bottom plane, intrinsics, pose, vessel footprint; blaze has ideal externally registered RGB',
        scope='frozen V8 + native surface research estimators; transparent Gemini additionally V33 five-prior IR proxy; no ToF refraction inverse, no multilayer dedicated adapter, no long-memory qualification'))
    start=time.time();ps=[];hs=[]
    env=dict(os.environ,PYTHONPATH='src:scripts',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    dump(OUT/'progress.json',dict(stage='regression',elapsed_s=0))
    with (OUT/'regression.log').open('w') as log:
        subprocess.run([sys.executable,'-m','pytest','-q'],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    for w in range(4):
        log=(OUT/f'generate-{w}.log').open('w');hs.append(log)
        ps.append(subprocess.Popen([BLENDER,'-b','-t','4','--python-exit-code','1','--python','scripts/generate_device_v38.py','--',
            '--root',str(DATA),'--seed','202610123899','--worker',str(w),'--workers','4'],stdout=log,stderr=subprocess.STDOUT,env=env))
    while any(p.poll() is None for p in ps):
        dump(OUT/'progress.json',dict(stage='native_generation',done=len(list(DATA.glob('*/frames.json'))),total=660,elapsed_s=time.time()-start));time.sleep(5)
    for h in hs:h.close()
    if any(p.returncode for p in ps):raise RuntimeError('native_generation_failed')
    frames=[r for p in DATA.glob('*/frames.json') for r in json.loads(p.read_text())]
    if len(frames)!=6864 or len(list(DATA.glob('*/frames.json')))!=660:raise RuntimeError('incomplete_generation')
    for stage,script,data in [('native_evaluation','evaluate_device_v38.py',DATA),('five_prior','refraction_device_v38.py',DATA/'five-prior')]:
        dump(OUT/'progress.json',dict(stage=stage,elapsed_s=time.time()-start))
        dst=OUT if stage=='native_evaluation' else OUT/'five-prior'
        with (OUT/(stage+'.log')).open('w') as log:
            subprocess.run([sys.executable,'scripts/'+script,'--data',str(data),'--output',str(dst)],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    verification={p:sha(Path(p))==h for p,h in hashes.items()};dump(OUT/'verification.json',verification)
    if not all(verification.values()):raise RuntimeError('frozen_source_changed')
    dump(OUT/'progress.json',dict(stage='completed',native_frames=len(frames),five_prior_frames=960,source_unchanged=True,elapsed_s=time.time()-start))
if __name__=='__main__':main()
