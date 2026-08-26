from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .geometry import depth_to_meters
from .refinement import RefinedDepth


class ClearGraspRefiner:
    """Modern adapter for the released ClearGrasp DRN models and depth2depth solver."""

    def __init__(
        self,
        source_path: Path,
        checkpoint_root: Path,
        executable_path: Path,
        inference_size: list[int],
        output_size: list[int],
        intrinsics: list[float],
        min_depth_m: float,
        max_depth_m: float,
        inertia_weight: float,
        smoothness_weight: float,
        tangent_weight: float,
    ) -> None:
        try:
            import h5py
            import torch
        except ImportError as exc:
            raise RuntimeError("Install the train and research extras to use ClearGrasp") from exc

        source_path = source_path.resolve()
        checkpoint_root = checkpoint_root.resolve()
        executable_path = executable_path.resolve()
        if not (source_path / "api" / "modeling" / "deeplab.py").is_file():
            raise FileNotFoundError(f"ClearGrasp source is incomplete: {source_path}")
        if not executable_path.is_file():
            raise FileNotFoundError(f"ClearGrasp depth2depth executable is missing: {executable_path}")
        source_text = str(source_path)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        deeplab = importlib.import_module("api.modeling.deeplab")
        deeplab_masks = importlib.import_module("api.modeling.deeplab_masks")
        if source_path not in Path(deeplab.__file__).resolve().parents:
            raise RuntimeError(f"A conflicting Python package named 'api' shadowed {source_path / 'api'}")

        self.torch = torch
        self.h5py = h5py
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        specs = {
            "mask": (
                deeplab_masks.DeepLab,
                2,
                checkpoint_root / "mask_segmentation" / "checkpoint_mask.pth",
            ),
            "outlines": (
                deeplab_masks.DeepLab,
                3,
                checkpoint_root / "outlines" / "checkpoint_outlines.pth",
            ),
            "normals": (
                deeplab.DeepLab,
                3,
                checkpoint_root / "surface_normals" / "checkpoint_normals.pth",
            ),
        }
        self.models = {}
        for name, (model_class, classes, checkpoint_path) in specs.items():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model = model_class(
                num_classes=classes,
                backbone="drn",
                sync_bn=True,
                freeze_bn=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.models[name] = model.eval().to(self.device)

        self.inference_size = tuple(map(int, inference_size))
        self.output_size = tuple(map(int, output_size))
        if len(intrinsics) != 4:
            raise ValueError("ClearGrasp intrinsics must be [fx, fy, cx, cy]")
        self.fx, self.fy, self.cx, self.cy = map(float, intrinsics)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.inertia_weight = float(inertia_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.tangent_weight = float(tangent_weight)
        self.executable_path = executable_path
        self._temporary = tempfile.TemporaryDirectory(prefix="liquid-depth-cleargrasp-")
        self.temporary_path = Path(self._temporary.name)

    @staticmethod
    def _rotate_normals(normals: np.ndarray) -> np.ndarray:
        """Rotate ClearGrasp Y-up normals +90 degrees around X for depth2depth."""
        return np.stack((normals[0], -normals[2], normals[1]), axis=0)

    @staticmethod
    def _occlusion_weight(logits) -> np.ndarray:
        probability = logits.softmax(dim=1)[0, 1].float().cpu().numpy()
        weight = np.power(1.0 - probability, 3.0) * 1000.0
        return np.clip(weight, 1.0, 999.0).astype(np.uint16)

    def _tensor(self, rgb_bgr: np.ndarray):
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.inference_size, interpolation=cv2.INTER_NEAREST)
        value = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        return self.torch.from_numpy(value).unsqueeze(0).to(self.device)

    def predict(self, rgb_bgr: np.ndarray, raw_depth: np.ndarray) -> RefinedDepth:
        height, width = rgb_bgr.shape[:2]
        tensor = self._tensor(rgb_bgr)
        with self.torch.inference_mode():
            mask_logits = self.models["mask"](tensor)
            outline_logits = self.models["outlines"](tensor)
            normal_logits = self.models["normals"](tensor)
            normal_logits = self.torch.nn.functional.normalize(normal_logits, p=2, dim=1)

        mask = mask_logits.argmax(dim=1)[0].byte().cpu().numpy() * 255
        mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
        mask = cv2.resize(mask, self.output_size, interpolation=cv2.INTER_NEAREST)
        occlusion = self._occlusion_weight(outline_logits)
        occlusion = cv2.resize(occlusion, self.output_size, interpolation=cv2.INTER_NEAREST)
        normals = normal_logits[0].float().cpu().numpy()
        normals = self._rotate_normals(normals).transpose(1, 2, 0)
        normals = cv2.resize(normals, self.output_size, interpolation=cv2.INTER_NEAREST)
        normals = normals.transpose(2, 0, 1).astype(np.float32)

        depth = depth_to_meters(raw_depth).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth[~np.isfinite(depth)] = 0.0
        depth = cv2.resize(depth, self.output_size, interpolation=cv2.INTER_NEAREST)
        depth[depth < self.min_depth_m] = 0.0
        depth[depth > self.max_depth_m] = self.max_depth_m
        depth[mask > 0] = 0.0

        input_path = self.temporary_path / "input-depth.png"
        output_path = self.temporary_path / "output-depth.png"
        normals_path = self.temporary_path / "predicted-surface-normals.h5"
        occlusion_path = self.temporary_path / "predicted-occlusion-weight.png"
        scaled_depth = np.clip(depth * 4000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        if not cv2.imwrite(str(input_path), scaled_depth):
            raise RuntimeError(f"Failed to write ClearGrasp input: {input_path}")
        if not cv2.imwrite(str(occlusion_path), occlusion):
            raise RuntimeError(f"Failed to write ClearGrasp boundary weights: {occlusion_path}")
        with self.h5py.File(normals_path, "w") as stream:
            stream.create_dataset("/result", data=normals)

        command = [
            str(self.executable_path),
            str(input_path),
            str(output_path),
            "-xres",
            str(self.output_size[0]),
            "-yres",
            str(self.output_size[1]),
            "-fx",
            str(self.fx),
            "-fy",
            str(self.fy),
            "-cx",
            str(self.cx),
            "-cy",
            str(self.cy),
            "-inertia_weight",
            str(self.inertia_weight),
            "-smoothness_weight",
            str(self.smoothness_weight),
            "-tangent_weight",
            str(self.tangent_weight),
            "-input_normals",
            str(normals_path),
            "-input_tangent_weight",
            str(occlusion_path),
        ]
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode != 255:
            message = (result.stdout + result.stderr).decode(errors="replace")
            raise RuntimeError(f"ClearGrasp depth2depth failed ({result.returncode}): {message}")
        restored = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        if restored is None:
            raise RuntimeError(f"ClearGrasp did not produce output: {output_path}")
        restored = restored.astype(np.float32) / 4000.0
        restored = cv2.resize(restored, (width, height), interpolation=cv2.INTER_NEAREST)
        valid = np.isfinite(restored) & (restored >= self.min_depth_m) & (restored <= self.max_depth_m)
        restored = np.where(valid, restored, 0.0).astype(np.float32)
        confidence = np.where(valid, 0.5, 0.0).astype(np.float32)
        return RefinedDepth(restored, confidence, "cleargrasp")
