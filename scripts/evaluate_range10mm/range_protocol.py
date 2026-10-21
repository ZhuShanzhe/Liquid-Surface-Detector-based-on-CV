"""Frozen, output-independent range and calibration protocol."""
import json, math, os
from pathlib import Path
import numpy as np

ROOT=Path(os.environ.get('LIQUID_EVAL_ROOT','/root/autodl-tmp/liquid-depth-artifacts/formal-v15-range01-9-fixed10mm-20261021'))
OLD=Path('/root/autodl-tmp/liquid-depth-artifacts/formal-v14-diverse-20261021')
BANDS=[(.1,1.)]+[(float(i),float(i+1)) for i in range(1,9)]
QUOTAS=[12]+[11]*8

def band_of(h):
    if not .1 <= float(h) <= 9.:return None
    return min(8,int(float(h)))

def installation(band,attempt):
    """Depth first; dimensions sampled within a declared feasible set, not shrunk."""
    gid=10000+band*1000+attempt
    r=np.random.default_rng(20261021170039+gid*104729)
    lo,hi=BANDS[band]
    h=float(r.uniform(lo+.01,hi-.01))
    if attempt==0 and band==0:h=.1
    if attempt==0 and band==8:h=9.
    aspect=1. if gid%3==0 else float(r.uniform(.75,1.35))
    cap=h+float(r.uniform(.08,.12) if band==8 else r.uniform(.15,.5))
    minimum_headroom=.3*1.4*max(1,aspect)*(480/270)/math.tan(math.radians(50))+.02
    b=float(r.uniform(max(.9,cap+minimum_headroom,6.0 if band==0 and attempt%2==0 else .9),10.))
    # Draw a realizable installation with a visible rim for the existing
    # independent-dimensions calibration route; never resize old inputs.
    maximum=min(1.5,(b-cap)*math.tan(math.radians(50))/(1.4*max(1,aspect)*(480/270)))
    if maximum<.3:raise ValueError('no_feasible_radius_at_declared_range')
    radius=float(r.uniform(.3,maximum)) if maximum>.3 else .3
    xy=r.uniform(-.15,.15,2)*radius
    fov=max(45.,math.degrees(2*math.atan(1.13*(radius*max(1,aspect)+np.linalg.norm(xy))*(480/270)/(b-cap))))
    if fov>100:raise ValueError('no_feasible_fov_at_declared_range')
    return dict(bottom_m=b,capacity_m=cap,radius_m=radius,aspect=aspect,shape=('cylindrical','elliptical','rectangular_tank')[gid%3],
        camera_xy_m=xy.tolist(),camera_mount='top_offset',horizontal_fov_deg=fov,installation_id=gid,
        noise_family=int(np.random.default_rng(80710001+gid*907).integers(4))),h

def fixed_noise(value,rng):
    while True:
        error=float(rng.normal(0,.01/3))
        if abs(error)<=.01:return float(value)+error
