"""Predeclared additional holdout and report-cell contract, not depth strata."""
import numpy as np

SEED=202610170835
SENSORS=('tof',)
SCENES=('ordinary_diffuse','homogeneous_transparent','homogeneous_translucent',
        'partial_depth_loss','low_light','glare','uneven_surface','floating_objects')
FAR_INSTALLATIONS=25
REPEATS=4
LEVELS=('mild','moderate','severe')
CONDITIONS=(('ordinary_diffuse','all'),('floating_objects','all'),
            *((s,l) for s in ('low_light','glare','uneven_surface') for l in LEVELS))


def far_design(gid):
    if not 0<=gid<FAR_INSTALLATIONS:raise ValueError('unknown_installation')
    r=np.random.default_rng(SEED+gid*104729)
    # New geometry/pose/appearance seed. No former case or RGB copied as a new frame.
    h=float(r.uniform(.01,1.));b=float(r.uniform(6.05,9.95));z=b-h
    rx=z*np.tan(np.deg2rad(27))*.38
    return dict(physical_id=gid,nominal_liquid_height_m=h,z=z,camera_to_bottom_m=b,
        radius_x=rx,radius_y=rx*float(r.uniform(.78,1.02)),offset_angle=float(r.uniform(0,2*np.pi)),
        shape=('cylindrical','elliptical','rectangular_tank')[gid%3],phase=float(r.uniform(0,2*np.pi)),
        frequency_x=float(r.uniform(3.5,7.5)),frequency_y=float(r.uniform(3.5,7.5)))


def condition_settings(scenario,level,d):
    i=LEVELS.index(level) if level in LEVELS else 1
    amp=min(d['radius_x']*(.012,.035,.075)[i],d['nominal_liquid_height_m']*(.035,.1,.22)[i],.08) if scenario=='uneven_surface' else 0.
    return dict(corruption_severity=(.18,.48,.78)[i] if scenario in ('low_light','glare','uneven_surface') else .22,
        wave_amplitude_m=amp,light_level=(.18,.08,.03)[i] if scenario=='low_light' else (.2,.45,1.)[i] if scenario=='glare' else 1.,
        exposure=(-1.4,-2.6,-3.8)[i] if scenario=='low_light' else (-1.3,-.5,.2)[i] if scenario=='glare' else -.3,
        liquid_roughness=(.12,.055,.018)[i] if scenario=='glare' else .28)


def protocol():
    return dict(version='tof_formal_v1_20261017',seed=SEED,sensors=list(SENSORS),
        reporting_cell='scene x sensor x severity (no actual-depth strata)',minimum_frames_per_report_cell=100,
        far_shallow='offline grouping B>=6m and0<H<=1m, not an inference input',
        far_new_physical_installations=25,noise_frames_per_condition=4,
        prior_budget_opaque='one empty RGB-D plus independent operator metric geometry',
        prior_budget_refraction='one empty + four distinct known liquid heights; current H never available to inference',
        no_zero_snap_or_prediction_clip=True,no_active_stereo_in_v2=True,
        reuse='unchanged opaque predictions retained and tagged historical; old stereo image proxy excluded, not relabeled',
        inclusion='old cohorts retain recorded protocol; new opaque draw requires raw spatial support and offline direct-liquid source audit; exclude input only, never exclude by prediction error',
        statistical_unit='physical installation clusters, not artificial duplicate frames',
        camera_scope='ToF parametric depth proxy for opaque scenes; raw4-phase multifrequency AMCW iToF simulation for uniform refraction; not a commercial device qualification')
