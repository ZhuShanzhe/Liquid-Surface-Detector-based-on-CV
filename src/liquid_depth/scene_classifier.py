"""Small multi-label RGB routing head; its scores are not measurement confidence."""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn

TAGS = ('low_light', 'glare', 'uneven_surface', 'floating_objects')
MATERIALS = ('ordinary_diffuse', 'homogeneous_transparent', 'homogeneous_translucent')


class SceneHead(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = (3, 16, 24, 40, 64)
        for a, b in zip(channels, channels[1:]):
            layers += [nn.Conv2d(a, b, 3, 2, 1), nn.GroupNorm(4, b), nn.SiLU()]
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((3, 5))
        self.head = nn.Sequential(nn.Linear(64 * 16 + 16, 96), nn.SiLU(), nn.Dropout(.25), nn.Linear(96, 7))
        self.register_buffer('geometry_mean',torch.zeros(16))
        self.register_buffer('geometry_scale',torch.ones(16))

    def forward(self, x, geometry=None):
        z=self.encoder(x)
        if geometry is None:geometry=torch.zeros((len(x),16),device=x.device)
        geometry=((geometry-self.geometry_mean)/self.geometry_scale).clamp(-6,6)
        pooled=torch.cat([self.pool(z).flatten(1),z.amax((2,3)),geometry],1)
        return self.head(pooled)


class PretrainedSceneHead(nn.Module):
    """Transfer-learned RGB encoder with a small normalized RGB-D routing head."""
    def __init__(self,pretrained=False):
        super().__init__()
        from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
        self.encoder=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None).features
        for p in self.encoder.parameters():p.requires_grad=False
        for block in list(self.encoder.children())[-3:]:
            for p in block.parameters():p.requires_grad=True
        self.head=nn.Sequential(nn.Linear(576*2+16,128),nn.SiLU(),nn.Dropout(.35),nn.Linear(128,7))
        self.register_buffer('geometry_mean',torch.zeros(16))
        self.register_buffer('geometry_scale',torch.ones(16))
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('rgb_scale',torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self,x,geometry=None):
        # Keep pretraining BN statistics; small grouped synthetic batches are not
        # representative camera-domain recalibration batches.
        self.encoder.eval()
        z=self.encoder((x-self.rgb_mean)/self.rgb_scale)
        if geometry is None:geometry=torch.zeros((len(x),16),device=x.device)
        geometry=((geometry-self.geometry_mean)/self.geometry_scale).clamp(-6,6)
        return self.head(torch.cat([z.mean((2,3)),z.amax((2,3)),geometry],1))


def image_tensor(bgr,roi=None):
    a = np.asarray(bgr)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError('uint8_aligned_bgr_image_required')
    if roi is not None:
        mask=np.asarray(roi,bool)
        if mask.shape!=a.shape[:2]:raise ValueError('aligned_image_roi_required')
        y,x=np.nonzero(mask)
        if len(x):
            mx=max(4,int(np.ptp(x)*.08));my=max(4,int(np.ptp(y)*.08))
            a=a[max(0,y.min()-my):min(len(a),y.max()+my+1),max(0,x.min()-mx):min(a.shape[1],x.max()+mx+1)]
    rgb = cv2.cvtColor(cv2.resize(a, (320, 192)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255.)


def geometry_features(depth_m=None,dry_depth_m=None,roi=None):
    """Observable RGB-D routing features, no true current mask or height."""
    if depth_m is None or dry_depth_m is None or roi is None:return np.zeros(16,np.float32)
    depth=np.asarray(depth_m,float);dry=np.asarray(dry_depth_m,float);roi=np.asarray(roi,bool)
    if depth.shape!=dry.shape or roi.shape!=depth.shape:raise ValueError('aligned_routing_prior_required')
    valid=roi & np.isfinite(depth) & np.isfinite(dry) & (depth>0) & (dry>0)
    if valid.sum()<16:return np.zeros(16,np.float32)
    # Remove the dominant depth shift to separate local objects/waves from liquid level.
    shift=np.where(valid,dry-depth,0).astype(np.float32)
    center=float(np.median(shift[valid]));residual=shift-center
    z=float(np.median(depth[valid]));noise=.0015+.0007*z
    filled=np.where(valid,residual,0).astype(np.float32)
    smooth=cv2.medianBlur(filled,3)
    a=smooth[valid];scaled=a/max(noise,.001)
    q=np.percentile(scaled,[1,5,10,25,50,75,90,95,99])
    above=valid & (smooth>max(.015,3*noise))
    n,labels,stats,_=cv2.connectedComponentsWithStats(above.astype(np.uint8))
    sizes=stats[1:,cv2.CC_STAT_AREA] if n>1 else np.array([0])
    v=np.r_[q, np.mean(above[valid]), np.mean(sizes>=3), np.log1p(max(sizes))/8,
            np.log1p(max(n-1,0))/5, np.std(scaled), float(valid.sum()/max(roi.sum(),1)),z/10]
    return np.clip(v,-20,20).astype(np.float32)


class SceneClassifier:
    def __init__(self, checkpoint, device='cpu'):
        self.device = torch.device(device)
        state = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
        if state['schema'] not in (3,4):
            raise ValueError('unsupported_scene_head_schema')
        self.net = (PretrainedSceneHead() if state['schema']==4 else SceneHead()).to(self.device).eval()
        self.net.load_state_dict(state['model'])
        self.thresholds = state['thresholds']
        self.material_threshold = float(state['material_threshold'])
        self.model_id = state['model_id']
        self.activation_thresholds=state.get('activation_thresholds',{})

    def predict(self, rgb_bgr, *, depth_m=None,dry_depth_m=None,roi=None):
        with torch.inference_mode():
            features=torch.from_numpy(geometry_features(depth_m,dry_depth_m,roi))[None].to(self.device)
            scores = self.net(image_tensor(rgb_bgr,roi)[None].to(self.device),features)[0]
            material = scores[:3].softmax(0).cpu().numpy()
            tags = scores[3:].sigmoid().cpu().numpy()
        winner = int(material.argmax())
        return dict(material=MATERIALS[winner], material_score=float(material[winner]),
                    material_scores=dict(zip(MATERIALS, map(float, material))),
                    material_ambiguous=bool(material[winner] < self.material_threshold),
                    tag_scores=dict(zip(TAGS, map(float, tags))),
                    tags=[k for k, p in zip(TAGS, tags) if p >= self.thresholds[k]],
                    float_peak_recommended=bool(tags[TAGS.index('floating_objects')]>=self.activation_thresholds.get('float_highest',self.thresholds['floating_objects'])),
                    activation_thresholds=self.activation_thresholds,
                    model_id=self.model_id, probabilities_are_metrology_confidence=False)
