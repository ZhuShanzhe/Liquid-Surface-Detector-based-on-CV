"""Optional bottom-referenced height residual. No ground-truth input contract."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from liquid_depth.liquid_height_metrics import FEATURE_NAMES, observable_features


class HeightResidualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(len(FEATURE_NAMES),24),nn.SiLU(),nn.Linear(24,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,features):
        return .05*torch.tanh(self.net(features).squeeze(-1))


class CalibratedHeightPredictor:
    """Experimental scalar correction; does not certify identity or uncertainty."""
    def __init__(self,checkpoint,base_checkpoint_sha256):
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['feature_names']!=list(FEATURE_NAMES):raise ValueError('Feature contract mismatch')
        if state['source_checkpoint_sha256']!=base_checkpoint_sha256:raise ValueError('Base model mismatch')
        self.model=HeightResidualHead().eval();self.model.load_state_dict(state['model'])
        self.mean,self.scale=state['mean'],state['scale']

    @torch.inference_mode()
    def correct(self,prediction,raw_prediction,raw,predicted_mask,calibration):
        result=deepcopy(prediction)
        result['research_only']=True
        if not result['accepted']:return result
        features,d=observable_features(result,raw_prediction,raw,predicted_mask,calibration)
        x=(torch.tensor([features],dtype=torch.float32)-self.mean)/self.scale
        delta=float(self.model(x)[0])*d
        result['height_head_delta_m']=delta
        result['statistics']={k:v+delta for k,v in result['statistics'].items()}
        if not np.isfinite(result['statistics']['mean_m']) or result['statistics']['mean_m']<=0:
            result.update(accepted=False,reasons=['nonpositive_calibrated_height'])
        return result
