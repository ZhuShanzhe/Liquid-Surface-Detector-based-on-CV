from dataclasses import replace
import numpy as np
import pytest
from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.ordinary_surface_v40 import prepare,calibrate,estimate,disparity,smooth_residual


def setup(bottom=10.02,height=.01):
    k=np.array([[160.,0,63.5],[0,160.,47.5],[0,0,1.]])
    pose=np.eye(4);pose[:3,:3]=np.diag([1.,-1.,-1.]);pose[2,3]=bottom
    cal=HeightCalibration(k,pose,0.,(3.,3.),2.)
    ctx=prepare(cal,.8,(96,128),.1)
    return ctx,np.full(ctx.shape,bottom,np.float32),np.full(ctx.shape,bottom-height,np.float32)


@pytest.mark.parametrize('height',[-.025,0.,.015,.5,1.2])
def test_signed_no_capacity_or_zero_clipping(height):
    ctx,dry,wet=setup(height=height)
    out=estimate(wet,ctx,calibrate(dry,ctx))
    assert out['statistics']['mean_m']==pytest.approx(height,abs=1e-4)
    assert out['selected_points']>24


def test_over_ten_numeric_depth_retained():
    ctx,dry,wet=setup(height=.01)
    assert (wet>10).all()
    _,valid=disparity(wet,ctx)
    assert valid.sum()==ctx.roi.sum()
    assert estimate(wet,ctx,calibrate(dry,ctx))['statistics'] is not None


def test_holes_never_filled_or_boundary_expanded():
    ctx,dry,wet=setup()
    wet[40:50,60:70]=0
    d,v=disparity(wet,ctx);_,v2,_=smooth_residual(d,v,ctx,.18)
    assert np.array_equal(v,v2)
    assert not v2[40:50,60:70].any()
    assert not v2[~ctx.roi].any()


def test_reference_geometry_mismatch_rejected():
    ctx,dry,wet=setup();ref=calibrate(dry,ctx)
    with pytest.raises(ValueError):estimate(wet,replace(ctx,digest='different'),ref)


def test_no_data_is_no_output():
    ctx,dry,wet=setup()
    assert estimate(wet*0,ctx,calibrate(dry,ctx))['statistics'] is None


def test_common_disparity_bias_corrected():
    ctx,dry,wet=setup(height=.05)
    dry=ctx.fb/(ctx.fb/dry+.05);wet=ctx.fb/(ctx.fb/wet+.05)
    ref=calibrate(dry,ctx)
    assert abs(estimate(wet,ctx,ref,'paired')['statistics']['mean_m']-.05)<1e-4


def test_free_plane_orientation_exact():
    ctx,dry,wet=setup(height=.2)
    out=estimate(wet,ctx,calibrate(dry,ctx))
    assert out['free_plane_angle_deg']<1e-3
    assert out['free_plane_height_m']==pytest.approx(.2,abs=1e-4)


def test_baseline_and_shape_are_part_of_installation_identifier():
    ctx,dry,wet=setup()
    cal=HeightCalibration(np.array([[160.,0,63.5],[0,160.,47.5],[0,0,1.]]),ctx.pose,0.,(3.,3.),2.)
    assert prepare(cal,.8,(96,128),.2).digest!=ctx.digest
    assert prepare(cal,.8,(97,128),.1).digest!=ctx.digest
