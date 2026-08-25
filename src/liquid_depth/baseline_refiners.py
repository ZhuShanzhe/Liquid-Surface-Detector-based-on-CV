from __future__ import annotations

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np

from .geometry import depth_to_meters
from .refinement import RefinedDepth


class DREDSSwinDRNetRefiner:
    """Adapter for the official DREDS SwinDRNet validation preprocessing."""

    def __init__(
        self,
        source_path: Path,
        checkpoint_path: Path,
        input_size: list[int],
        max_depth_m: float,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install the train and research extras to use SwinDRNet") from exc

        source_path = source_path.resolve()
        if not (source_path / "networks" / "SwinDRNet.py").is_file():
            raise FileNotFoundError(f"SwinDRNet source is incomplete: {source_path}")
        source_text = str(source_path)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        config_module = importlib.import_module("config")
        config = config_module._C.clone()
        config_module._update_config_from_file(
            config,
            str(source_path / "configs" / "swin_tiny_patch4_window7_224_lite.yaml"),
        )
        model_class = importlib.import_module("networks.SwinDRNet").SwinDRNet
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        self.model = model_class(
            config,
            img_size=int(input_size[0]),
            num_classes=9,
        )
        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch = torch
        self.model.eval().to(self.device)
        self.input_size = tuple(map(int, input_size))
        self.max_depth_m = float(max_depth_m)

    def predict(self, rgb_bgr: np.ndarray, raw_depth: np.ndarray) -> RefinedDepth:
        height, width = rgb_bgr.shape[:2]
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb,
            self.input_size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        rgb /= 255.0
        depth = depth_to_meters(raw_depth).astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0) & (depth <= self.max_depth_m)
        depth = np.where(valid, depth, 0.0)
        depth = cv2.resize(
            depth,
            self.input_size,
            interpolation=cv2.INTER_NEAREST,
        )
        rgb_tensor = self.torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        depth_tensor = self.torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        with self.torch.inference_mode():
            prediction, _, confidence_raw, confidence_restored = self.model(
                rgb_tensor.to(self.device),
                depth_tensor.to(self.device),
            )
        restored = prediction[0, 0].float().cpu().numpy()
        confidence = self.torch.maximum(confidence_raw, confidence_restored)[0, 0].float().cpu().numpy()
        restored = cv2.resize(
            restored,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        confidence = cv2.resize(
            confidence,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        output_valid = np.isfinite(restored) & (restored > 0) & (restored <= self.max_depth_m)
        restored = np.where(output_valid, restored, 0.0).astype(np.float32)
        confidence = np.where(output_valid, confidence, 0.0).astype(np.float32)
        return RefinedDepth(
            restored,
            np.clip(confidence, 0.0, 1.0),
            "dreds_swindrnet",
        )
