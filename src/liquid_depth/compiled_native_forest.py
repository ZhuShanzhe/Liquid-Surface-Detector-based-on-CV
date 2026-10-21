"""Exact FP32-input/FP64-threshold forest execution; trusted local assets only."""
import ctypes,hashlib,os,subprocess,threading
from pathlib import Path
import numpy as np
_lock=threading.Lock()
def build():
    source=Path(__file__).with_name('native_forest.cpp')
    key=hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    directory=Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'liquid-depth'
    directory.mkdir(parents=True,exist_ok=True);target=directory/('forest-'+key+'.so')
    with _lock:
        if not target.exists():
            temp=target.with_name(target.name+'.'+str(os.getpid())+'.tmp')
            subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(temp)],check=True)
            temp.replace(target)
    f=ctypes.CDLL(str(target)).forest_predict
    f.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*8;f.restype=None
    return f
class CompiledForest:

    def __init__(self, original):
        import sys
        pass
        from .packed_native_forest import PackedForest
        self.pack = PackedForest(original)
        self.function = build()
        self.n_features_in_ = original.n_features_in_
        self.n_jobs = 1

    def predict(self, x):
        x = np.ascontiguousarray(x, np.float32)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_ or (not np.isfinite(x).all()):
            raise ValueError('finite_aligned_forest_features_required')
        a = self.pack
        out = np.empty(len(x), np.float64)
        arrays = [x, a.feature, a.threshold, a.left, a.right, a.leaf, a.value, out]
        self.function(len(x), x.shape[1], a.feature.shape[0], a.feature.shape[1], *[ctypes.c_void_p(v.ctypes.data) for v in arrays])
        return out

    def predict_proba(self, x):
        p = self.predict(x)
        return np.column_stack([1 - p, p])
