"""Predeclared balanced holdout; no bpy or learned-model dependencies."""
import math
import random

SEED = 202610167331
REPLICATES = 6
REPEATS = 4
SENSORS = ('tof', 'structured_light')
SCENES = ('low_light', 'glare', 'uneven_surface')
LEVELS = ('mild', 'moderate', 'severe')
BANDS = ((0., 0.), (.01, .1), (.1, 1.), (1., 2.), (2., 3.), (3., 4.),
         (4., 5.), (5., 6.), (6., 7.), (7., 8.), (8., 9.))
BAND_NAMES = ('dry', '0.01-0.1m', '0.1-1m', '1-2m', '2-3m', '3-4m',
              '4-5m', '5-6m', '6-7m', '7-8m', '8-9m')


def design(gid):
    if not 0 <= gid < len(BANDS)*REPLICATES:
        raise ValueError('unknown physical installation')
    band, rep = divmod(gid, REPLICATES)
    rng = random.Random(SEED + gid*100003)
    lo, hi = BANDS[band]
    h = rng.uniform(lo+.1*(hi-lo), hi-.1*(hi-lo)) if hi else 0.
    z = min((.55, 2.8, 8.8)[rep % 3]*rng.uniform(.92, 1.04), (10-h)*.91)
    angle = rng.uniform(0, 2*math.pi)
    radius = z*math.tan(math.radians(27))*.38
    shape = ('cylindrical', 'elliptical', 'rectangular_tank')[(rep+band) % 3]
    return dict(physical_id=gid, band=band, band_name=BAND_NAMES[band], replicate=rep,
                nominal_liquid_height_m=h, z=z, offset_angle=angle, radius_x=radius,
                radius_y=radius if shape == 'cylindrical' else radius*rng.uniform(.76, 1.04),
                shape=shape, frequency_x=rng.uniform(3.5, 7.5), frequency_y=rng.uniform(3.5, 7.5),
                phase=rng.uniform(0, 2*math.pi))


def settings(scenario, level, d):
    i = LEVELS.index(level)
    severity = (.18, .48, .78)[i]
    amp = min(d['radius_x']*(.012, .035, .075)[i],
              d['nominal_liquid_height_m']*(.035, .10, .22)[i],
              min(.12, d['z']*.15)*.7) if scenario == 'uneven_surface' else 0.
    return dict(corruption_severity=severity, wave_amplitude_m=amp,
                light_level=(.18, .08, .03)[i] if scenario == 'low_light' else
                            (.2, .45, 1.)[i] if scenario == 'glare' else 1.,
                exposure=(-1.4, -2.6, -3.8)[i] if scenario == 'low_light' else
                         (-1.3, -.5, .2)[i] if scenario == 'glare' else -.3,
                liquid_roughness=(.12, .055, .018)[i] if scenario == 'glare' else .28)


def tolerance(scene, truth, legacy=False):
    if scene == 'uneven_surface' and not legacy:
        return max(.05, .05*truth)
    return max(.02, .02*truth)


def protocol():
    return dict(version='special_scene_stratified_v1', seed=SEED, width=480, height=270,
        physical_installations=len(BANDS)*REPLICATES, sensor_installations=len(BANDS)*REPLICATES*2,
        scenes=list(SCENES), levels=list(LEVELS), sensors=list(SENSORS), bands=list(BANDS),
        band_names=list(BAND_NAMES), geometry_replicates_per_band=REPLICATES,
        noise_repeats=REPEATS, total_frames=4752, frames_per_scene=1584,
        frames_per_severity_per_scene=528, frames_per_sensor_scene_severity_depth=24,
        liquid_tolerance={'low_light':'max(20mm,2% actual bottom-relative liquid depth)',
                         'glare':'max(20mm,2% actual bottom-relative liquid depth)',
                         'uneven_surface':'max(50mm,5% actual bottom-relative area-mean liquid depth)'},
        uneven_peak_tolerance='max(50mm,5% actual highest height above bottom)',
        algorithms='unchanged DeterminedSceneMeter determined_all_scenes_v1_v8_peak_20261016',
        visual_enabled=False, optimization=False, training=False,
        pairing='same physical installation, RGB, exact shape, operator perturbations and dry reference for both sensors; common random numbers across severity',
        geometry_prior='one high-quality noisy empty RGB-D reference plus noisy operator outline/dimension/gravity/distance; shared across severities; annotation retry<=6 before scoring',
        dry_prior_noise='legacy reference sigma=1mm+0.5mm*z^2;4% holes; identical controlled dry reference across sensors, not a native single-frame camera noise comparison',
        uneven_scope='opaque diffuse wavy surface, no floating objects; dry controls separate; no cross-scenario combinations',
        severity_definition='predeclared simulator control triplets, NOT measured lux or calibrated sensor saturation; RGB+depth joint degradation proxy',
        sensor_limitation='ToF parametric noise+dropout proxy; structured light speckle/disparity proxy with35mm baseline,0.26px noise; not full ToF waveform or phase-shifting SL physics or commercial10m qualification',
        inclusion='User-requested input-evidence amendment: primary evaluation only after input_evidence_v1 qualification; all planned inputs and excluded reasons retained; no true/predicted depth or error used for qualification',
        input_evidence_amendment='requested after first unfiltered run; fixed64 interior returns,6of16 tiles,and25% span on both footprint axes; engineering rule not tuned to pass rate; original unfiltered evidence retained separately',
        balanced_qualification='first4 jointly input-qualified ToF/SL noise draws per physical installation and condition, at most64 attempts; no intensity/geometry changes; paired rejected draws retained and acquisition success reported separately',
        offline_capture_sufficiency='additional known synthetic return-origin audit:>=64 direct surface returns,>=6of16 spatial tiles,>=25% visible-surface span in both axes; never filters by magnitude of numeric error and never feeds labels to meter',
        no_zero_snap_or_output_clip=True, camera_to_bottom_max_m=10., raw_return_profile_m=10.5,
        truth='offline physical-area mean and geometric peak relative to interior bottom; never passed to inference',
        uncertainty='95% percentile bootstrap over66 physical installations, not4752 independent frames; no industry certification',
        relative_error='exclude actual zero liquid depth from relative denominators; zero-depth still in absolute/coverage/pass metrics',
        reference_only_legacy_uneven_tolerance='max(20mm,2% actual liquid depth)')


def validate():
    assert protocol()['total_frames'] == len(BANDS)*REPLICATES*2*3*3*REPEATS
    for gid in range(len(BANDS)*REPLICATES):
        d = design(gid)
        assert d['z'] + d['nominal_liquid_height_m'] < 10
        for level in LEVELS:
            s = settings('uneven_surface', level, d)
            assert s['wave_amplitude_m'] <= d['nominal_liquid_height_m']*.22 + 1e-12
    assert tolerance('uneven_surface', .1) == .05
    assert tolerance('uneven_surface', 9) == .45
    assert tolerance('low_light', 9) == .18


if __name__ == '__main__':
    import json
    validate()
    print(json.dumps(protocol(), indent=2))
