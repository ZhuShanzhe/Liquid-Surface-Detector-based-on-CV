"""Native-scale observation features promoted from the verified V13 runtime."""
from dataclasses import replace
import cv2,numpy as np,torch
from torch import nn
from .empty_geometry_v41 import height_calibration
from .geometry_execution import interior_ray_roi
from .liquid_height_metrics import world_points
@torch.inference_mode()
def global_observed_features(base, rgb, depth, dry, roi):
    from liquid_depth.scene_classifier import image_tensor, geometry_features
    model = base.net
    gx = geometry_features(depth, dry, roi)
    x = image_tensor(rgb, roi)[None].to(base.device)
    z = model.encoder((x - model.rgb_mean) / model.rgb_scale)
    embedding = torch.cat([z.mean((2, 3)), z.amax((2, 3))], 1)
    g = torch.from_numpy(gx)[None].to(base.device)
    normalized = ((g - model.geometry_mean) / model.geometry_scale).clamp(-6, 6)
    logits = model.head(torch.cat([embedding, normalized], 1))[0].cpu().numpy()
    return np.r_[embedding[0].cpu().numpy(), gx, logits].astype(np.float32)

def native_input(rgb, raw, dry, bundle):
    """No labels. Native crop, never rescale or require valid depth for aperture."""
    cal = height_calibration(bundle)
    rim = float(bundle['rim_z_m'])
    roi = interior_ray_roi(replace(cal, bottom_z_m=rim - 1e-05), rim, rgb.shape[:2], 0.98)
    roi = cv2.dilate(roi.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    yy, xx = np.nonzero(roi)
    if not len(xx):
        raise ValueError('empty calibrated aperture')
    y0, y1 = (max(0, yy.min() - 8), min(rgb.shape[0], yy.max() + 9))
    x0, x1 = (max(0, xx.min() - 8), min(rgb.shape[1], xx.max() + 9))
    ay, by = (max(0, y0 - 3), min(rgb.shape[0], y1 + 3))
    ax, bx = (max(0, x0 - 3), min(rgb.shape[1], x1 + 3))
    image = rgb[y0:y1, x0:x1].astype(np.float32) / 255
    median = cv2.medianBlur(rgb[ay:by, ax:bx], 7)[y0 - ay:y1 - ay, x0 - ax:x1 - ax].astype(np.float32) / 255
    contrast = image - median
    valid = (raw > 0) & np.isfinite(raw) & roi
    heights = world_points(raw, cal)[..., 2] - cal.bottom_z_m
    center = float(np.median(heights[valid])) if valid.any() else 0.0
    sigma = 0.0015 + 0.0007 * np.maximum(raw, 0)
    relative = np.where(valid, np.clip((heights - center) / sigma, -20, 20) / 20, 0)
    changes = np.where(valid & (dry > 0), np.clip((dry - raw) / sigma, -20, 20) / 20, 0)
    selection = np.s_[y0:y1, x0:x1]
    x = np.concatenate([image, contrast * 2, valid[selection][..., None], relative[selection][..., None], changes[selection][..., None], roi[selection][..., None]], axis=-1)
    return (x.transpose(2, 0, 1).astype(np.float32), (y0, y1, x0, x1))

class ChannelNorm(nn.Module):
    """Per-pixel normalization: no dependence on other images or padded area."""

    def __init__(self, n):
        super().__init__()
        self.norm = nn.LayerNorm(n)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

def conv(a, b, d=1):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=d, dilation=d), ChannelNorm(b), nn.SiLU())

class NativeMIL(nn.Module):

    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(conv(10, 24), conv(24, 24), conv(24, 24, 2))
        self.context = nn.Sequential(nn.AvgPool2d(2), conv(24, 32), conv(32, 32, 2), conv(32, 32, 3))
        self.fuse = nn.Sequential(conv(56, 32), conv(32, 24), nn.Conv2d(24, 1, 1))

    def forward(self, x):
        z = self.local(x)
        c = self.context(z)
        return self.fuse(torch.cat([z, torch.nn.functional.interpolate(c, size=z.shape[-2:], mode='bilinear', align_corners=False)], 1))[:, 0]

def summarize(logit, roi):
    p = 1 / (1 + np.exp(-np.clip(logit, -40, 40)))
    v = p[roi]
    if not len(v):
        return np.zeros(42, np.float32)
    ordered = np.sort(v)[::-1]
    out = [*np.percentile(v, [0, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]), *[float(ordered[:k].mean()) for k in (1, 2, 3, 5, 10, 20, 50, 100)]]
    for t in (0.1, 0.25, 0.5, 0.75, 0.9, 0.97):
        n, _, stats, _ = cv2.connectedComponentsWithStats((roi & (p >= t)).astype(np.uint8))
        areas = stats[1:, cv2.CC_STAT_AREA]
        out.extend([np.log1p(n - 1), np.log1p(areas.max()) if len(areas) else 0, np.mean(v >= t), np.log1p(np.sum(areas >= 2))])
    return np.array(out, np.float32)
