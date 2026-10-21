"""Exact, bounded geometry cache; copies prevent shared mutable masks."""
import pickle,threading
from .ordinary_surface_v35 import interior_ray_roi as _original
_lock=threading.RLock();_values={}
def interior_ray_roi(*args,**kwargs):
    key=pickle.dumps((args,kwargs),protocol=5)
    with _lock:
        if key not in _values:
            if len(_values)>=64:_values.pop(next(iter(_values)))
            _values[key]=_original(*args,**kwargs)
        return _values[key].copy()
