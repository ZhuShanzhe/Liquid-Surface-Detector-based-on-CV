"""Build the offline Windows x64 forest DLL; never called by the application.

Use an x64 MSVC Developer shell (static CRT /MT), or a MinGW-w64 g++ compiler
(static libstdc++ and libgcc). The numerical ABI is identical to Linux: FP32
features, FP64 thresholds, tree-order double summation, no fast-math.
"""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def msvc_environment(compiler):
    """Use the installed MSVC x64 environment without persisting environment data."""
    if os.environ.get('INCLUDE') and os.environ.get('LIB'):
        return os.environ.copy()
    batch = next((p/'Auxiliary/Build/vcvars64.bat' for p in Path(compiler).parents
                  if (p/'Auxiliary/Build/vcvars64.bat').is_file()), None)
    if batch is None:
        raise RuntimeError('MSVC_developer_environment_required')
    result = subprocess.run(f'call "{batch}" >nul && set', shell=True,
                            capture_output=True, text=True, errors='replace',
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError('MSVC_x64_environment_initialization_failed')
    environment = os.environ.copy()
    for line in result.stdout.splitlines():
        if '=' in line and not line.startswith('='):
            key, value = line.split('=', 1)
            environment[key] = value
    if not environment.get('INCLUDE') or not environment.get('LIB'):
        raise RuntimeError('MSVC_x64_environment_incomplete')
    return environment


def build(output_dir=None, compiler=None):
    if sys.platform != 'win32' or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise RuntimeError('run_windows_x64_build_on_windows_x64')
    source = ROOT/'src/liquid_depth/native_forest.cpp'
    output = Path(output_dir) if output_dir else source.parent/'native'
    output.mkdir(parents=True, exist_ok=True)
    compiler = compiler or shutil.which('cl') or shutil.which('g++')
    if not compiler:
        raise RuntimeError('MSVC_x64_or_MinGW_w64_compiler_required_at_build_time_only')
    compiler = str(Path(compiler).resolve())
    kind = 'msvc' if Path(compiler).name.lower() in ('cl', 'cl.exe') else 'mingw'
    environment = msvc_environment(compiler) if kind == 'msvc' else None
    with tempfile.TemporaryDirectory(prefix='forest-build-', dir=output) as temporary:
        folder = Path(temporary)
        library = folder/'forest-win_amd64.dll'
        if kind == 'msvc':
            command = [compiler, '/nologo', '/O2', '/std:c++17', '/LD', '/MT', '/EHsc',
                       '/fp:precise', str(source), '/Fo'+str(folder/'native_forest.obj'),
                       '/link', '/OUT:'+str(library)]
        else:
            command = [compiler, '-O3', '-std=c++17', '-shared', '-static-libgcc',
                       '-static-libstdc++', '-ffp-contract=off', str(source), '-o', str(library)]
        process = subprocess.run(command, cwd=folder, capture_output=True, text=True,
                                 errors='replace', env=environment,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        if process.returncode:
            raise RuntimeError('forest_build_failed\n'+process.stdout+'\n'+process.stderr)
        loaded = ctypes.CDLL(str(library))
        getattr(loaded, 'forest_predict')
        # The validation handle locks its DLL on Windows; unload before cleanup.
        target = output/'forest-win_amd64.dll'
        shutil.copy2(library, target)
        import _ctypes
        _ctypes.FreeLibrary(loaded._handle)
        del loaded
    record = dict(schema=1, abi='cdecl_f32_f64_v1', architecture='win_amd64',
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  compiler_family=kind, compiler_name=Path(compiler).name,
                  floating_point='float32_input_float64_threshold_and_tree_order_sum_no_fast_math',
                  runtime='static_msvc_CRT' if kind == 'msvc' else 'static_libgcc_static_libstdc++')
    target.with_suffix('.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(dict(library=str(target), **record), indent=2))
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--compiler', help='Explicit MSVC cl.exe or MinGW-w64 g++.exe')
    args = parser.parse_args()
    build(args.output_dir, args.compiler)
