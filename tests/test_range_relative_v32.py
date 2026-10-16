import numpy as np
import pytest
from liquid_depth import finite_relative_v30 as base
from liquid_depth import range_relative_v32 as model
from experiment_focus_v32 import measurement_levels,specs,choices


CAL=dict(fx=450.,cx=80.,cy=64.,bottom_m=10.,container_max_height_m=9.4)
HS=[0.,1.4,3.7,6.1,8.4]


def pair(s,ch='left'):
    p=np.array([[-30.,-20.],[30.,-20.],[-30.,20.],[30.,20.],[0.,20.],[20.,0.]])+[80,64]
    return base.package(p,(p-[80,64])*np.exp(s)+[80,64],CAL,ch,dict(correlation=.98))


def calibration_edges(biased=False):
    return [(a,b,pair(.025*(HS[b]-HS[a])+(.0005*(b-a)**2 if biased else 0),ch))
            for a in range(5) for b in range(a+1,5) for ch in ('left','right')]


@pytest.mark.parametrize('shape',['linear','pchip'])
def test_signed_outputs_not_zeroed_or_capacity_clipped(shape):
    prior=model.fit_local(calibration_edges(),HS,CAL,'transparent',shape)
    for h in (-.02,.002,.01,.05,.5,2.5,9.,11.):
        result=model.infer([(0,pair(.025*h))],prior,CAL,'mean',None)
        assert result['height_m']==pytest.approx(h,abs=1e-8)
        assert result['height_m']!=0
        assert not result['zero_snapping'] and not result['capacity_clipping']
        assert not result['confidence_gate_enabled'] and not result['coverage_gate_enabled']
    assert 'outside_container_capacity' in result['reasons']


def test_local_reference_calibration_removes_nontransitive_zero_offsets():
    edges=calibration_edges(True);prior=model.fit_local(edges,HS,CAL,'translucent_mild')
    groups=[(0,pair(1e-5))]
    for a in range(1,5):groups.append((a,pair(-(.025*HS[a]+.0005*a*a)+1e-5)))
    result=model.infer(groups,prior,CAL,'mean',None)
    assert 0<result['height_m']<.001
    assert len(prior['curves'])==10


def test_pchip_monotonic_and_extrapolation_not_flat():
    x=[0.,.03,.09,.16,.25];y=[0.,1.4,3.7,6.1,8.4]
    out=np.array([model.inverse_curve(s,x,y) for s in np.linspace(-.1,.35,200)])
    assert np.all(np.diff(out)>0)
    assert out[0]<0 and out[-1]>8.4
    for s,h in zip(x,y):assert model.inverse_curve(s,x,y)==pytest.approx(h)


def test_all_soft_weights_positive_and_missing_observations_explicit():
    prior=model.fit_local(calibration_edges(),HS,CAL,'transparent')
    groups=[(a,pair(.025*(2.5-HS[a]))) for a in range(5)]
    result=model.infer(groups,prior,CAL,'robust',.025)
    assert result['track_groups']==5
    assert result['height_m']==pytest.approx(2.5,abs=1e-7)
    assert model.infer([],prior,CAL)['height_m'] is None
    with pytest.raises(ValueError):model.fit_local([],HS,CAL,'transparent')
    with pytest.raises(ValueError):model.fit_local(calibration_edges(),HS[:4],CAL,'transparent')


def test_new_split_and_repeated_zero_budget():
    dev=specs('development');test=specs('test')
    assert not {c['seed'] for c in dev}&{c['seed'] for c in test}
    for split,repeat in [('development',12),('test',24)]:
        levels=measurement_levels(split)
        assert sum(s['height']==0 for s in levels)==repeat
        assert max(s['height'] for s in levels)==9
        assert {.002,.005,.01,.02,.04,.06,.08,.1}<={s['height'] for s in levels}
    assert all(c['bottom']<=10 and c['capacity']>9 and c['width_px']>=96 for c in dev+test)
    assert len(choices())==37
