# Build with: LIQUID_BLENDER_DIR=<verified Blender directory> python -m PyInstaller packaging/liquid_simulation.spec
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

repo=Path(SPECPATH).resolve().parent
vendor=Path(os.environ['LIQUID_BLENDER_DIR']).resolve()
assert (vendor/'blender.exe').is_file(), 'Verified Windows Blender distribution required'
scripts=['render_scene_console.py','generate_synthetic_liquid.py','generate_fullframe_v16.py',
    'generate_special_scene_strata_v1.py','generate_other_scenes_v34.py','special_scene_protocol_v1.py',
    'measure_release.py','measure_determined.py','tof_phase_simulator.py','desktop_freeze_smoke.py',
    'desktop_gui_smoke.py']
datas=[(str(repo/'assets'),'assets'),(str(repo/'configs/release_assets.json'),'configs'),
       (str(repo/'src/liquid_depth/simulation.py'),'src/liquid_depth'),
       (str(repo/'src/liquid_depth/ui'),'src/liquid_depth/ui'),
       (str(repo/'src/liquid_depth/native_forest.cpp'),'liquid_depth'),
       (str(repo/'src/liquid_depth/native/forest-win_amd64.json'),'liquid_depth/native'),
       (str(repo/'docs/CPU_DESKTOP_GUIDE.rst'),'docs'),
       (str(repo/'docs/cpu_desktop_help.html'),'docs'),
       (str(vendor),'vendor/blender')]
datas.extend((str(repo/'scripts'/p),'scripts') for p in scripts)
for distribution in ('torch','torchvision','scikit-learn','numpy','scipy','joblib','PySide6','Pillow','PyYAML'):
    datas.extend(copy_metadata(distribution))
# Include original notices from split Qt wheels and transitive Python/native
# dependencies, plus the versioned official WebEngine license-page snapshot.
# This snapshot is not a binary-generated SBOM (see THIRD_PARTY_NOTICES.rst).
sys.path.insert(0,str(repo/'packaging'))
from collect_desktop_licenses import license_data
datas.extend(license_data(repo))
hidden=['measure_release','measure_determined','tof_phase_simulator','desktop_freeze_smoke','desktop_gui_smoke']
# Trusted joblib estimators unpickle private sklearn symbols dynamically.
hidden+=collect_submodules('sklearn.ensemble')+collect_submodules('sklearn.tree')
a=Analysis([str(repo/'scripts/desktop_launcher.py')],
    pathex=[str(repo/'src'),str(repo/'scripts')],
    binaries=[(str(repo/'src/liquid_depth/native/forest-win_amd64.dll'),'liquid_depth/native')],
    datas=datas,hiddenimports=hidden,hookspath=[],runtime_hooks=[],
    excludes=['matplotlib','pandas','tkinter','PyQt5','PyQt6','IPython','notebook','pytest','tensorboard'],
    noarchive=False)
# Fail rather than ship DLLs borrowed from unrelated development tools. Qt's
# unversioned ICU imports belong to Windows System32, not Poppler/Conda ICU.
for destination, source, kind in a.binaries:
    normalized=source.replace('\\','/').lower()
    if '/dependencies/native/' in normalized or 'anaconda' in normalized or 'miniconda' in normalized:
        raise RuntimeError('Foreign development DLL in portable bundle: '+source)
    if destination.replace('\\','/').lower()=='icuuc.dll':
        raise RuntimeError('Windows ICU must resolve from System32, not the portable root')
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='LiquidDepthSimulation',
        debug=False,bootloader_ignore_signals=False,strip=False,upx=False,
        console=False,disable_windowed_traceback=False)
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='LiquidDepthSimulation')
