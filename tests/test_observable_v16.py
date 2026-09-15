import inspect
import numpy as np
import torch
from torch import nn
from liquid_depth.models.observable_v16 import ObservableLiquidPipeline,encode_rgbd,gated_measurements,sanitize_raw

def test_invalid_depth_is_not_an_anchor():
    raw=torch.tensor([[[[float('nan'),float('inf'),-.1,0,.09,.1,2,10,11]]]])
    assert sanitize_raw(raw).tolist()==[[[[0,0,0,0,0,.10000000149011612,2,10,0]]]]
def test_prediction_gate_rejects_empty_and_sparse_support():
    raw=torch.ones(2,1,10,10);p=torch.zeros_like(raw);p[0,:,2:8,2:8]=1
    selected,count=gated_measurements(raw,p,radius=0)
    assert count.tolist()==[36,0]
    assert selected[0].sum()==36 and selected[1].sum()==0
    raw[0,:,2:8,2:8]=0
    selected,count=gated_measurements(raw,p,radius=0)
    assert selected.sum()==0
def test_dilation_uses_predicted_support_only():
    raw=torch.ones(1,1,9,9);p=torch.zeros_like(raw);p[:,:,4,4]=1
    selected,count=gated_measurements(raw,p,radius=1,min_points=1)
    assert selected.sum()==9 and count.item()==9
def test_encoding_does_not_treat_missing_as_near_depth():
    rgb=torch.zeros(1,3,8,8);raw=torch.zeros(1,1,8,8)
    x=encode_rgbd(rgb,raw)
    assert x.shape==(1,5,8,8) and x[:,3:].sum()==0
class Locator(nn.Module):
    def forward(self,x):
        return {'mask_logits':torch.full_like(x[:,3:4],-20),'depth_m':torch.ones_like(x[:,3:4])}
class Depth(nn.Module):
    def forward(self,rgb240,dn,low,span,rgb,raw,prior,mask,anchor):
        return torch.ones_like(raw)*anchor[:,None,None,None]
def test_pipeline_has_no_target_or_metadata_input_and_no_empty_gate_fallback():
    assert list(inspect.signature(ObservableLiquidPipeline.forward).parameters)==['self','rgb','raw']
    rgb=torch.rand(1,3,16,16);raw=torch.ones(1,1,16,16)
    p=ObservableLiquidPipeline(Locator(),Depth(),'gated')(rgb,raw)
    assert p['depth_m'].sum()==0 and not p['has_depth_anchor'].item()
    p=ObservableLiquidPipeline(Locator(),Depth(),'full')(rgb,raw)
    assert p['depth_m'].sum()==256
def test_invalid_alignment_is_rejected():
    pipe=ObservableLiquidPipeline(Locator(),Depth())
    import pytest
    with pytest.raises(ValueError):pipe(torch.rand(1,3,16,16),torch.ones(1,1,8,8))

def test_camera_units_are_explicit_and_alignment_is_checked():
    import pytest
    from liquid_depth.observation_v16 import prepare_observation
    rgb=np.zeros((4,4,3),np.uint8);raw=np.full((4,4),1000,np.uint16)
    _,depth=prepare_observation(rgb,raw,{'depth_alignment':'color','depth_scale_to_m':.001})
    assert np.all(depth==1)
    with pytest.raises(KeyError):prepare_observation(rgb,raw,{'depth_alignment':'color'})
    with pytest.raises(ValueError):prepare_observation(rgb,raw,{'depth_scale_to_m':.001})
def test_factory_reprojection_zbuffer_uses_nearest_color_depth():
    from liquid_depth.observation_v16 import prepare_observation
    rgb=np.zeros((3,3,3),np.uint8);raw=np.array([[2.,1.]],np.float32)
    camera={'depth_alignment':'native','depth_scale_to_m':1.,'depth_resolution':[2,1],'color_resolution':[3,3],
            'depth_camera_matrix':[[100,0,0],[0,100,0],[0,0,1]],'color_camera_matrix':[[1,0,0],[0,1,0],[0,0,1]],
            'depth_to_color_rotation':np.eye(3).tolist(),'depth_to_color_translation_m':[0,0,.5]}
    _,depth=prepare_observation(rgb,raw,camera)
    assert depth[0,0]==1.5 and np.count_nonzero(depth)==1
