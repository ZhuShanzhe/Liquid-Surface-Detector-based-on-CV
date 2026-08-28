from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .config import resolve_config_path
from .geometry import depth_to_meters


@dataclass(frozen=True)
class RefinedDepth:
    depth_m: np.ndarray
    confidence: np.ndarray
    backend: str


class DepthRefiner(Protocol):
    def predict(self, rgb_bgr: np.ndarray, raw_depth: np.ndarray) -> RefinedDepth:
        """Return metric depth and per-pixel confidence in [0, 1]."""


class IdentityDepthRefiner:
    """Pass raw sensor depth through and expose validity as confidence."""

    def predict(self, rgb_bgr: np.ndarray, raw_depth: np.ndarray) -> RefinedDepth:
        del rgb_bgr
        depth_m = depth_to_meters(raw_depth)
        valid = np.isfinite(depth_m) & (depth_m > 0)
        confidence = valid.astype(np.float32)
        depth_m = np.where(valid, depth_m, 0.0).astype(np.float32)
        return RefinedDepth(depth_m, confidence, "identity")


class TorchScriptDepthRefiner:
    """Run a portable RGB-D refinement model with a stable five-channel contract."""

    def __init__(
        self,
        model_path: Path,
        input_size: list[int],
        max_depth_m: float,
        preserve_valid_raw: bool = False,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install the 'train' extra to use TorchScript depth refinement") from exc
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(
            str(model_path),
            map_location=self.device,
        ).eval()
        self.input_size = tuple(map(int, input_size))
        self.max_depth_m = float(max_depth_m)
        self.preserve_valid_raw = bool(preserve_valid_raw)

    def _metric_dict_output(
        self,
        output: dict,
        width: int,
        height: int,
    ) -> RefinedDepth | None:
        if "depth_m" not in output:
            return None
        depth_tensor = output["depth_m"]
        confidence_tensor = output.get("confidence")
        if confidence_tensor is None and "log_variance" in output:
            confidence_tensor = self.torch.sigmoid(-output["log_variance"])
        if confidence_tensor is None:
            confidence_tensor = self.torch.ones_like(depth_tensor)
        depth = depth_tensor[0, 0].float().cpu().numpy()
        confidence = confidence_tensor[0, 0].float().cpu().numpy()
        depth = cv2.resize(
            depth,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        confidence = cv2.resize(
            confidence,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0) & (depth <= self.max_depth_m)
        depth = np.where(valid, depth, 0.0).astype(np.float32)
        confidence = np.where(valid, confidence, 0.0).astype(np.float32)
        return RefinedDepth(
            depth,
            np.clip(confidence, 0.0, 1.0),
            "torchscript_multitask",
        )

    def predict(self, rgb_bgr: np.ndarray, raw_depth: np.ndarray) -> RefinedDepth:
        torch = self.torch
        height, width = rgb_bgr.shape[:2]
        depth_m = depth_to_meters(raw_depth).astype(np.float32)
        valid = np.isfinite(depth_m) & (depth_m > 0) & (depth_m <= self.max_depth_m)
        depth_m = np.where(valid, depth_m, 0.0)

        target_width, target_height = self.input_size
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb /= 255.0
        rgb = cv2.resize(
            rgb,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        normalized_depth = cv2.resize(
            depth_m / self.max_depth_m,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        validity = cv2.resize(
            valid.astype(np.float32),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        model_input = np.concatenate(
            (rgb, normalized_depth[..., None], validity[..., None]),
            axis=2,
        )
        tensor = torch.from_numpy(model_input.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(tensor)
        if isinstance(output, dict):
            metric = self._metric_dict_output(output, width, height)
            if metric is not None:
                if not self.preserve_valid_raw:
                    return metric
                return RefinedDepth(
                    np.where(valid, depth_m, metric.depth_m).astype(np.float32),
                    np.where(valid, 1.0, metric.confidence).astype(np.float32),
                    f"{metric.backend}_preserve_valid",
                )
            output = output.get("out", next(iter(output.values())))
        if isinstance(output, (tuple, list)):
            output = output[0]
        output = output[0].float().cpu().numpy()
        if output.ndim == 2:
            output = output[None]
        refined = np.clip(output[0], 0.0, 1.0) * self.max_depth_m
        confidence = 1.0 / (1.0 + np.exp(-output[1])) if output.shape[0] > 1 else np.ones_like(refined)
        refined = cv2.resize(
            refined,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        confidence = cv2.resize(
            confidence,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        return RefinedDepth(
            refined,
            np.clip(confidence, 0.0, 1.0),
            "torchscript",
        )


class TransCGDFNetRefiner:
    """Adapt the official corrected TransCG DFNet and preprocessing contract."""

    def __init__(
        self,
        source_path: Path,
        checkpoint_path: Path,
        input_size: list[int],
        depth_min_m: float,
        depth_max_m: float,
        depth_coefficient: float,
        inpainting: bool,
    ) -> None:
        import importlib
        import sys

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install the 'train' extra to use TransCG DFNet") from exc

        source_path = source_path.resolve()
        if not (source_path / "models" / "DFNet.py").is_file():
            raise FileNotFoundError(f"TransCG source is incomplete: {source_path}")
        source_text = str(source_path)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        model_class = importlib.import_module("models.DFNet").DFNet

        # The official checkpoint stores a NumPy scalar in metric metadata.
        scalar_type = np._core.multiarray.scalar
        torch.serialization.add_safe_globals(
            [
                (scalar_type, "numpy.core.multiarray.scalar"),
                np.dtype,
                type(np.dtype(np.float64)),
            ]
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch = torch
        self.model = model_class(
            in_channels=4,
            hidden_channels=64,
            L=5,
            k=12,
        )
        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        self.model.eval().to(self.device)
        self.input_size = tuple(map(int, input_size))
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.depth_coefficient = float(depth_coefficient)
        self.inpainting = bool(inpainting)

    def predict(self, rgb_bgr: np.ndarray, raw_depth: np.ndarray) -> RefinedDepth:
        from scipy.interpolate import NearestNDInterpolator

        height, width = rgb_bgr.shape[:2]
        depth = depth_to_meters(raw_depth).astype(np.float32)
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = cv2.resize(
            rgb,
            self.input_size,
            interpolation=cv2.INTER_NEAREST,
        )
        depth = cv2.resize(
            depth,
            self.input_size,
            interpolation=cv2.INTER_NEAREST,
        )
        valid = np.isfinite(depth) & (depth >= self.depth_min_m) & (depth <= self.depth_max_m)
        depth = np.where(valid, depth, 0.0)
        available = depth[depth > 0]
        if available.size:
            mean = float(available.mean())
            radius = self.depth_coefficient * float(available.std())
            depth = np.where(
                (depth >= mean - radius) & (depth <= mean + radius),
                depth,
                0.0,
            )
        if self.inpainting:
            coordinates = np.where(depth > 0)
            if coordinates[0].size:
                interpolator = NearestNDInterpolator(
                    np.transpose(coordinates),
                    depth[coordinates],
                )
                depth = interpolator(*np.indices(depth.shape)).astype(np.float32)

        low = float(depth.min() - 0.5 * depth.std() - 1e-6)
        high = float(depth.max() + 0.5 * depth.std() + 1e-6)
        normalized_depth = (depth - low) / (high - low)
        rgb_tensor = self.torch.from_numpy((rgb / 255.0).transpose(2, 0, 1)).float()
        depth_tensor = self.torch.from_numpy(normalized_depth).float()
        with self.torch.inference_mode():
            result = self.model(
                rgb_tensor.unsqueeze(0).to(self.device),
                depth_tensor.unsqueeze(0).to(self.device),
            )
        result = result[0].float().cpu().numpy() * (high - low) + low
        result = cv2.resize(
            result,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        output_valid = np.isfinite(result) & (result >= self.depth_min_m) & (result <= self.depth_max_m)
        result = np.where(output_valid, result, 0.0).astype(np.float32)
        confidence = np.where(output_valid, 0.5, 0.0).astype(np.float32)
        return RefinedDepth(result, confidence, "transcg_dfnet")


def make_depth_refiner(
    config: dict,
    options_override: dict | None = None,
) -> DepthRefiner:
    options = options_override or config.get("depth_refinement", {"backend": "identity"})
    backend = str(options.get("backend", "identity")).lower()
    if backend in {"none", "identity"}:
        return IdentityDepthRefiner()
    if backend == "torchscript":
        model = options["torchscript"]
        path = resolve_config_path(config, model["model_path"])
        return TorchScriptDepthRefiner(
            path,
            model["input_size"],
            model["max_depth_m"],
            model.get("preserve_valid_raw", False),
        )
    if backend == "transcg_dfnet":
        model = options["transcg_dfnet"]
        return TransCGDFNetRefiner(
            resolve_config_path(config, model["source_path"]),
            resolve_config_path(config, model["checkpoint_path"]),
            model.get("input_size", [320, 240]),
            model.get("depth_min_m", 0.3),
            model.get("depth_max_m", 1.0),
            model.get("depth_coefficient", 3.0),
            model.get("inpainting", True),
        )
    if backend == "dreds_swindrnet":
        from .baseline_refiners import DREDSSwinDRNetRefiner

        model = options["dreds_swindrnet"]
        return DREDSSwinDRNetRefiner(
            resolve_config_path(config, model["source_path"]),
            resolve_config_path(config, model["checkpoint_path"]),
            model.get("input_size", [224, 224]),
            model.get("max_depth_m", 3.0),
        )
    if backend == "cleargrasp":
        from .cleargrasp_refiner import ClearGraspRefiner

        model = options["cleargrasp"]
        return ClearGraspRefiner(
            resolve_config_path(config, model["source_path"]),
            resolve_config_path(config, model["checkpoint_root"]),
            resolve_config_path(config, model["executable_path"]),
            model.get("inference_size", [256, 256]),
            model.get("output_size", [256, 144]),
            model.get("intrinsics", [185.0, 185.0, 128.0, 72.0]),
            model.get("min_depth_m", 0.1),
            model.get("max_depth_m", 1.5),
            model.get("inertia_weight", 1000.0),
            model.get("smoothness_weight", 0.001),
            model.get("tangent_weight", 1.0),
        )
    raise ValueError(f"Unknown depth refinement backend: {backend}")


def make_complex_depth_refiners(config: dict) -> dict[str, DepthRefiner]:
    """Build configured specialist refiners once for stream/batch reuse."""

    specialists: dict[str, DepthRefiner] = {}
    cache: dict[str, DepthRefiner] = {}
    model_options = config.get("complex_scene", {}).get("models", {})
    if not isinstance(model_options, dict):
        raise TypeError("complex_scene.models must be a mapping")
    for variant, options in model_options.items():
        if not isinstance(options, dict):
            raise TypeError(f"complex_scene.models.{variant} must be a mapping")
        if not bool(options.get("enabled", True)):
            continue
        clean_options = deepcopy(options)
        clean_options.pop("enabled", None)
        cache_key = repr(clean_options)
        if cache_key not in cache:
            cache[cache_key] = make_depth_refiner(config, clean_options)
        specialists[str(variant)] = cache[cache_key]
    return specialists
