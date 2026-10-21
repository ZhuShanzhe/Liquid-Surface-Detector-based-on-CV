"""Calibrated RGB-D descriptors; no scene controls or truth input."""
from pathlib import Path
from dataclasses import replace
from functools import lru_cache
import cv2,json,numpy as np,torch
from .empty_geometry_v41 import height_calibration
from .geometry_execution import interior_ray_roi
from .liquid_height_metrics import world_points
from .new_view_native import native_input,summarize
REPO=Path(__file__).resolve().parents[2]
_GLOBAL_ENCODER=None
@lru_cache(maxsize=128)
def _aperture_cached(key, shape):
    bundle = json.loads(key)
    cal = height_calibration(bundle)
    rim = float(bundle['rim_z_m'])
    return cv2.dilate(interior_ray_roi(replace(cal, bottom_z_m=rim - 1e-05), rim, shape, 0.98).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

def aperture(bundle, shape):
    return _aperture_cached(json.dumps(bundle, sort_keys=True), tuple(shape)).copy()

@lru_cache(maxsize=128)
def _bottom_cached(key, shape):
    bundle = json.loads(key)
    return interior_ray_roi(height_calibration(bundle), float(bundle['rim_z_m']), shape, 0.9)

def bottom_roi(bundle, shape):
    return _bottom_cached(json.dumps(bundle, sort_keys=True), tuple(shape)).copy()

def stats(a):
    a = np.asarray(a)
    return np.r_[np.percentile(a, [0, 10, 25, 50, 75, 90, 95, 99, 100]), np.mean(a), np.std(a)] if a.size else np.zeros(11)

def features(rgb, raw, dry, bundle, net):
    roi = aperture(bundle, raw.shape)
    ys, xs = np.nonzero(roi)
    y0, y1, x0, x1 = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = gray[y0:y1, x0:x1]
    s = hsv[y0:y1, x0:x1, 1] / 255
    v = hsv[y0:y1, x0:x1, 2] / 255
    r = roi[y0:y1, x0:x1]
    z = raw[y0:y1, x0:x1]
    d = dry[y0:y1, x0:x1]
    valid = (z >= 0.08) & np.isfinite(z) & r
    cal = height_calibration(bundle)
    height = (world_points(raw, cal)[..., 2] - cal.bottom_z_m)[y0:y1, x0:x1]
    center = float(np.median(height[valid])) if valid.any() else 0.0
    sigma = 0.0015 + 0.0007 * np.maximum(z, 0)
    residual = np.where(valid, (height - center) / sigma, 0)
    change = np.where(valid & (d > 0), (d - z) / sigma, 0)
    grad = np.sqrt(cv2.Sobel(gray, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(gray, cv2.CV_32F, 0, 1) ** 2)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    local = gray - cv2.GaussianBlur(gray, (0, 0), 3)
    f = [r.mean(), r.sum() / raw.size, (x1 - x0) / rgb.shape[1], (y1 - y0) / rgb.shape[0], np.mean(valid[r])]
    for a in (gray, s, v, grad, lap, local, residual[valid], change[valid]):
        f.extend(stats(a[r] if a.ndim == 2 else a))
    for th in (0.25, 0.5, 0.7, 0.85, 0.95, 0.99):
        mask = (v >= th) & (s < 0.3) & r
        n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        f.extend([mask.sum() / max(1, r.sum()), np.log1p(n - 1), st[1:, 4].max() / max(1, r.sum()) if n > 1 else 0])
    for yy in np.array_split(np.arange(gray.shape[0]), 4):
        for xx in np.array_split(np.arange(gray.shape[1]), 4):
            ix = np.ix_(yy, xx)
            rr = r[ix]
            vv = valid[ix]
            for a in (gray, s, v, grad, lap):
                x = a[ix][rr]
                f.extend([np.mean(x), np.std(x), np.max(x)] if x.size else [0, 0, 0])
            x = residual[ix][vv]
            f.extend(stats(x))
            f.append(np.mean(vv[rr]) if rr.any() else 0)
    ni, _ = native_input(rgb, raw, dry, bundle)
    h, w = ni.shape[1:]
    with torch.inference_mode():
        z = net(torch.nn.functional.pad(torch.from_numpy(ni)[None].cuda(), (0, w % 2, 0, h % 2)))[0, :h, :w].cpu().numpy()
    f.extend(summarize(z, ni[-1] > 0.5))
    bottom = bottom_roi(bundle, raw.shape)
    sy, sx = np.nonzero(bottom)
    gray_full = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    hsv_full = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
    for image in (gray_full, hsv_full[..., 1] / 255, hsv_full[..., 2] / 255, *[rgb[..., k] / 255 for k in range(3)]):
        f.extend(stats(image[bottom]))
    if len(sx):
        box = np.s_[sy.min():sy.max() + 1, sx.min():sx.max() + 1]
        mask = bottom[box]
        image = gray_full[box]
        sat = hsv_full[box][..., 1] / 255
        contrast = image - cv2.GaussianBlur(image, (0, 0), 2)
        for a in (contrast, cv2.Sobel(image, cv2.CV_32F, 1, 0), cv2.Sobel(image, cv2.CV_32F, 0, 1)):
            f.extend(stats(a[mask]))
        for th in (0.4, 0.6, 0.75, 0.85, 0.95, 0.99):
            bright = (image >= th) & (sat < 0.3) & mask
            n, _, st, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8))
            f.extend([bright.sum() / max(1, mask.sum()), np.log1p(n - 1), st[1:, 4].max() / max(1, mask.sum()) if n > 1 else 0])
        for yy in np.array_split(np.arange(image.shape[0]), 4):
            for xx in np.array_split(np.arange(image.shape[1]), 4):
                ix = np.ix_(yy, xx)
                rr = mask[ix]
                for a in (image, sat, contrast):
                    x = a[ix][rr]
                    f.extend([np.mean(x), np.std(x), np.max(x)] if x.size else [0, 0, 0])
    else:
        f.extend(np.zeros(33 + 18 + 16 * 9))
    global _GLOBAL_ENCODER
    if _GLOBAL_ENCODER is None:
        from liquid_depth.scene_classifier import SceneClassifier
        _GLOBAL_ENCODER = SceneClassifier(REPO / 'assets/models/scene_router/scene_head.pt', device='cuda')
    from .new_view_native import global_observed_features
    f.extend(global_observed_features(_GLOBAL_ENCODER, rgb, raw, dry, bottom))
    return np.nan_to_num(np.array(f, np.float32), nan=0.0, posinf=10000.0, neginf=-10000.0)
