from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SCENARIOS = (
    "ordinary",
    "transparent",
    "translucent",
    "glare",
    "low_light",
    "depth_failure",
    "multilayer",
    "uneven_surface",
    "floating_objects",
    "compound",
)


@dataclass(frozen=True)
class SyntheticScene:
    index: int
    seed: int
    split: str
    sequence_id: str
    scenario: str
    width: int
    height: int
    horizontal_fov_deg: float
    camera_position_m: tuple[float, float, float]
    camera_target_m: tuple[float, float, float]
    surface_radius_x_m: float
    surface_radius_y_m: float
    container_bottom_z_m: float
    container_rim_z_m: float
    wall_thickness_m: float
    tilt_x: float
    tilt_y: float
    wave_amplitude_m: float
    wave_frequency_x: float
    wave_frequency_y: float
    wave_phase: float
    liquid_ior: float
    liquid_roughness: float
    liquid_turbidity: float
    liquid_color_rgb: tuple[float, float, float]
    glass_ior: float
    glass_roughness: float
    light_level: float
    exposure: float
    floating_object_count: int

    @property
    def difficulty_tags(self) -> tuple[str, ...]:
        tags = {"synthetic", "container_edge"}
        if self.scenario != "ordinary":
            tags.add(self.scenario)
        if self.scenario in {"transparent", "multilayer", "compound"}:
            tags.add("transparent")
        if self.scenario == "translucent":
            tags.add("translucent")
        if self.scenario in {"glare", "compound"}:
            tags.update(("glare", "saturated_highlight"))
        if self.scenario in {"uneven_surface", "floating_objects", "compound"}:
            tags.add("nonplanar_surface")
        if self.scenario == "compound":
            tags.update(("low_light", "multilayer", "depth_failure"))
        return tuple(sorted(tags))


def _stable_split(index: int, seed: int) -> str:
    value = (index // 4 * 2654435761 + seed * 2246822519) & 0xFFFFFFFF
    bucket = value % 10
    return "test" if bucket == 0 else "val" if bucket == 1 else "train"


def sample_scene(
    index: int,
    *,
    seed: int = 2026,
    width: int = 640,
    height: int = 360,
    min_distance_m: float = 0.1,
    max_distance_m: float = 10.0,
) -> SyntheticScene:
    if not 0 < min_distance_m < max_distance_m:
        raise ValueError("Expected 0 < min_distance_m < max_distance_m")
    rng = np.random.default_rng(seed + index * 104729)
    scenario = SCENARIOS[index % len(SCENARIOS)]
    distance = float(np.exp(rng.uniform(np.log(min_distance_m), np.log(max_distance_m))))
    elevation = math.radians(float(rng.uniform(12.0, 58.0)))
    azimuth = float(rng.uniform(-math.pi, math.pi))
    camera_position = (
        distance * math.cos(elevation) * math.cos(azimuth),
        distance * math.cos(elevation) * math.sin(azimuth),
        distance * math.sin(elevation),
    )
    fov = float(rng.uniform(45.0, 62.0))
    radius_x = float(
        np.clip(
            distance * math.tan(math.radians(fov / 2.0)) * rng.uniform(0.18, 0.38),
            0.025,
            2.5,
        )
    )
    radius_y = float(radius_x * rng.uniform(0.65, 1.25))
    fill_depth = float(np.clip(radius_x * rng.uniform(0.8, 2.8), 0.04, 4.0))
    rim_height = float(np.clip(radius_x * rng.uniform(0.12, 0.45), 0.008, 0.6))
    wall_thickness = float(np.clip(radius_x * rng.uniform(0.015, 0.05), 0.0015, 0.025))
    nonplanar = scenario in {"uneven_surface", "floating_objects", "compound"}
    wave_amplitude = float(
        radius_x * rng.uniform(0.012, 0.07)
        if nonplanar
        else radius_x * rng.uniform(0.0, 0.004)
    )
    tilt_limit = 0.12 if scenario in {"uneven_surface", "compound"} else 0.025
    tilt_x, tilt_y = rng.uniform(-tilt_limit, tilt_limit, size=2)
    transparent = scenario != "translucent"
    turbidity = float(rng.uniform(0.0, 0.12) if transparent else rng.uniform(0.25, 0.65))
    color = np.clip(
        rng.uniform(0.55, 1.0, size=3) * (1.0 - 0.35 * turbidity), 0.05, 1.0
    )
    dark = scenario in {"low_light", "compound"}
    light_level = float(rng.uniform(0.05, 0.22) if dark else rng.uniform(0.7, 1.4))
    exposure = float(rng.uniform(-3.0, -1.2) if dark else rng.uniform(-0.3, 0.7))
    floating_count = int(rng.integers(4, 18)) if scenario in {"floating_objects", "compound"} else 0
    return SyntheticScene(
        index=index,
        seed=seed,
        split=_stable_split(index, seed),
        sequence_id=f"synthetic_{seed}_{index // 4:07d}",
        scenario=scenario,
        width=width,
        height=height,
        horizontal_fov_deg=fov,
        camera_position_m=tuple(map(float, camera_position)),
        camera_target_m=(0.0, 0.0, 0.0),
        surface_radius_x_m=radius_x,
        surface_radius_y_m=radius_y,
        container_bottom_z_m=-fill_depth,
        container_rim_z_m=rim_height,
        wall_thickness_m=wall_thickness,
        tilt_x=float(tilt_x),
        tilt_y=float(tilt_y),
        wave_amplitude_m=wave_amplitude,
        wave_frequency_x=float(rng.uniform(1.0, 4.5)),
        wave_frequency_y=float(rng.uniform(1.0, 4.5)),
        wave_phase=float(rng.uniform(-math.pi, math.pi)),
        liquid_ior=float(rng.uniform(1.30, 1.48)),
        liquid_roughness=float(rng.uniform(0.0, 0.16)),
        liquid_turbidity=turbidity,
        liquid_color_rgb=tuple(map(float, color)),
        glass_ior=float(rng.uniform(1.45, 1.55)),
        glass_roughness=float(rng.uniform(0.0, 0.12)),
        light_level=light_level,
        exposure=exposure,
        floating_object_count=floating_count,
    )


def camera_to_world(scene: SyntheticScene) -> np.ndarray:
    position = np.asarray(scene.camera_position_m, dtype=np.float64)
    forward = np.asarray(scene.camera_target_m, dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, np.asarray((0.0, 1.0, 0.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2] = right, up, -forward
    matrix[:3, 3] = position
    return matrix


def camera_intrinsics(scene: SyntheticScene) -> np.ndarray:
    fx = scene.width / (2.0 * math.tan(math.radians(scene.horizontal_fov_deg / 2.0)))
    return np.asarray(
        (
            (fx, 0.0, (scene.width - 1) / 2.0),
            (0.0, fx, (scene.height - 1) / 2.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def surface_height_and_gradient(
    scene: SyntheticScene, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ax = scene.wave_frequency_x / max(scene.surface_radius_x_m, 1e-6)
    ay = scene.wave_frequency_y / max(scene.surface_radius_y_m, 1e-6)
    phase_x = ax * x + scene.wave_phase
    phase_y = ay * y - 0.73 * scene.wave_phase
    wave = scene.wave_amplitude_m * np.sin(phase_x) * np.cos(phase_y)
    height = scene.tilt_x * x + scene.tilt_y * y + wave
    dx = scene.tilt_x + scene.wave_amplitude_m * ax * np.cos(phase_x) * np.cos(phase_y)
    dy = scene.tilt_y - scene.wave_amplitude_m * ay * np.sin(phase_x) * np.sin(phase_y)
    return height, dx, dy


def _camera_rays(scene: SyntheticScene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intrinsics = camera_intrinsics(scene)
    u, v = np.meshgrid(np.arange(scene.width), np.arange(scene.height))
    camera_dirs = np.stack(
        (
            (u - intrinsics[0, 2]) / intrinsics[0, 0],
            -(v - intrinsics[1, 2]) / intrinsics[1, 1],
            -np.ones_like(u),
        ),
        axis=-1,
    ).astype(np.float64)
    camera_dirs /= np.linalg.norm(camera_dirs, axis=-1, keepdims=True)
    transform = camera_to_world(scene)
    return transform[:3, 3], camera_dirs @ transform[:3, :3].T, transform


def _elliptical_wall_layers(
    scene: SyntheticScene,
    origin: np.ndarray,
    directions: np.ndarray,
    world_to_camera: np.ndarray,
) -> np.ndarray:
    rx = scene.surface_radius_x_m + scene.wall_thickness_m
    ry = scene.surface_radius_y_m + scene.wall_thickness_m
    dx, dy = directions[..., 0], directions[..., 1]
    ox, oy = float(origin[0]), float(origin[1])
    a = (dx / rx) ** 2 + (dy / ry) ** 2
    b = 2.0 * (ox * dx / (rx * rx) + oy * dy / (ry * ry))
    c = (ox / rx) ** 2 + (oy / ry) ** 2 - 1.0
    discriminant = b * b - 4.0 * a * c
    sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
    safe_a = np.where(np.abs(a) > 1e-12, a, np.nan)
    roots = np.stack(
        ((-b - sqrt_disc) / (2.0 * safe_a), (-b + sqrt_disc) / (2.0 * safe_a))
    )
    layers = np.full_like(roots, np.nan, dtype=np.float32)
    for layer_index, t in enumerate(roots):
        points = origin + t[..., None] * directions
        inside_height = (
            (points[..., 2] >= scene.container_bottom_z_m)
            & (points[..., 2] <= scene.container_rim_z_m)
        )
        camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        depth = -camera_points[..., 2]
        valid = (discriminant >= 0.0) & (t > 0.0) & inside_height & (depth > 0.0)
        layers[layer_index] = np.where(valid, depth, np.nan)
    return layers


def render_geometric_labels(scene: SyntheticScene) -> dict[str, np.ndarray]:
    origin, directions, transform = _camera_rays(scene)
    world_to_camera = np.linalg.inv(transform)
    dz = directions[..., 2]
    t = np.where(np.abs(dz) > 1e-9, -origin[2] / dz, np.nan)
    for _ in range(8):
        points = origin + t[..., None] * directions
        height, grad_x, grad_y = surface_height_and_gradient(scene, points[..., 0], points[..., 1])
        derivative = dz - grad_x * directions[..., 0] - grad_y * directions[..., 1]
        t -= (points[..., 2] - height) / np.where(np.abs(derivative) > 1e-8, derivative, np.nan)
    points = origin + t[..., None] * directions
    height, grad_x, grad_y = surface_height_and_gradient(scene, points[..., 0], points[..., 1])
    ellipse = (
        (points[..., 0] / scene.surface_radius_x_m) ** 2
        + (points[..., 1] / scene.surface_radius_y_m) ** 2
    )
    mask = np.isfinite(t) & (t > 0.0) & (ellipse <= 1.0) & (np.abs(points[..., 2] - height) < 1e-4)
    camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    target_depth = np.where(mask, -camera_points[..., 2], 0.0).astype(np.float32)
    normals_world = np.stack((-grad_x, -grad_y, np.ones_like(grad_x)), axis=-1)
    normals_world /= np.maximum(np.linalg.norm(normals_world, axis=-1, keepdims=True), 1e-8)
    normals_camera = normals_world @ world_to_camera[:3, :3].T
    normals_camera = np.where(mask[..., None], normals_camera, 0.0).astype(np.float32)
    wall_layers = _elliptical_wall_layers(scene, origin, directions, world_to_camera)
    candidates = np.concatenate((wall_layers, np.where(mask, target_depth, np.nan)[None]), axis=0)
    candidates.sort(axis=0)
    layer_depths = np.zeros((4, scene.height, scene.width), dtype=np.float32)
    layer_valid = np.zeros_like(layer_depths, dtype=np.uint8)
    count = min(candidates.shape[0], 4)
    layer_depths[:count] = np.nan_to_num(candidates[:count], nan=0.0)
    layer_valid[:count] = np.isfinite(candidates[:count]).astype(np.uint8)
    return {
        "target_depth_m": target_depth,
        "mask": mask.astype(np.uint8),
        "normal_camera": normals_camera,
        "layer_depths_m": layer_depths,
        "layer_valid": layer_valid,
    }


def _coarse_noise(rng: np.random.Generator, height: int, width: int, scale: int = 24) -> np.ndarray:
    coarse = rng.random((max(2, math.ceil(height / scale)), max(2, math.ceil(width / scale))))
    return np.repeat(np.repeat(coarse, scale, axis=0), scale, axis=1)[:height, :width]


def simulate_raw_depth(scene: SyntheticScene, labels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(scene.seed + scene.index * 130363 + 17)
    target = labels["target_depth_m"]
    mask = labels["mask"] > 0
    raw = np.where(mask, target, 0.0).astype(np.float32)
    sigma = 0.0007 + 0.0012 * target * target
    raw[mask] += rng.normal(0.0, sigma[mask]).astype(np.float32)
    fx, baseline_m = camera_intrinsics(scene)[0, 0], 0.055
    valid = raw > 0.0
    disparity = np.zeros_like(raw)
    disparity[valid] = fx * baseline_m / raw[valid]
    disparity[valid] += rng.normal(0.0, 0.18, size=int(valid.sum())).astype(np.float32)
    raw[valid] = fx * baseline_m / np.maximum(np.round(disparity[valid] * 16.0) / 16.0, 1e-3)
    invalid_ranges = {
        "ordinary": (0.02, 0.10),
        "transparent": (0.25, 0.60),
        "translucent": (0.15, 0.42),
        "glare": (0.38, 0.75),
        "low_light": (0.28, 0.68),
        "depth_failure": (0.65, 0.96),
        "multilayer": (0.35, 0.72),
        "uneven_surface": (0.18, 0.45),
        "floating_objects": (0.20, 0.48),
        "compound": (0.72, 0.98),
    }
    low, high = invalid_ranges[scene.scenario]
    probability = float(rng.uniform(low, high))
    structure = 0.65 * _coarse_noise(rng, scene.height, scene.width) + 0.35 * rng.random(target.shape)
    dropout = mask & (structure < probability)
    highlight = np.zeros_like(mask)
    if scene.scenario in {"glare", "compound"}:
        u, v = np.arange(scene.width)[None, :], np.arange(scene.height)[:, None]
        cx, cy = rng.uniform(0.3, 0.7) * scene.width, rng.uniform(0.25, 0.75) * scene.height
        rx, ry = rng.uniform(0.05, 0.18) * scene.width, rng.uniform(0.03, 0.15) * scene.height
        highlight = (((u - cx) / rx) ** 2 + ((v - cy) / ry) ** 2 <= 1.0) & mask
        dropout |= highlight
    wrong_rate = 0.22 if scene.scenario in {"transparent", "multilayer", "compound"} else 0.03
    wrong_return = mask & ~dropout & (rng.random(target.shape) < wrong_rate)
    far_layer = np.max(labels["layer_depths_m"], axis=0)
    wrong_return &= far_layer > 0
    raw[wrong_return] = far_layer[wrong_return]
    if scene.scenario in {"translucent", "multilayer", "compound"}:
        biased = mask & ~dropout & ~wrong_return
        raw[biased] += rng.uniform(0.002, 0.025) * np.maximum(raw[biased], 1.0)
    raw[dropout] = 0.0
    raw = np.where(np.isfinite(raw) & (raw > 0.0) & (raw <= 10.5), raw, 0.0).astype(np.float32)
    tolerance = np.maximum(0.003, 0.01 * target)
    uncertainty = np.ones_like(target, dtype=np.float32)
    measured = mask & (raw > 0)
    uncertainty[measured] = np.clip(
        np.abs(raw[measured] - target[measured]) / np.maximum(tolerance[measured], 1e-6),
        0.0,
        10.0,
    ) / 10.0
    return {
        "raw_depth_m": raw,
        "uncertainty": uncertainty,
        "simulated_dropout_mask": dropout.astype(np.uint8),
        "simulated_highlight_mask": highlight.astype(np.uint8),
    }


def scene_metadata(scene: SyntheticScene, sample_dir: Path) -> dict[str, object]:
    metadata = asdict(scene)
    metadata.update(
        {
            "difficulty_tags": ";".join(scene.difficulty_tags),
            "camera_to_world": camera_to_world(scene).tolist(),
            "camera_intrinsics": camera_intrinsics(scene).tolist(),
            "paths": {
                "rgb": "rgb.png",
                "raw_depth": "raw_depth_m.npy",
                "target_depth": "target_depth_m.npy",
                "mask": "liquid_surface_mask.npy",
                "normal": "normal_camera.npy",
                "uncertainty": "uncertainty.npy",
                "layer_depths": "layer_depths_m.npy",
                "layer_valid": "layer_valid.npy",
                "dropout_mask": "simulated_dropout_mask.npy",
                "highlight_mask": "simulated_highlight_mask.npy",
            },
            "sample_dir": sample_dir.name,
            "generator_version": "liquid_sim_v1",
            "sensor_model": "active_stereo_proxy_v1",
        }
    )
    return metadata


def write_sample_arrays(
    sample_dir: Path, labels: dict[str, np.ndarray], sensor: dict[str, np.ndarray]
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "target_depth_m.npy": labels["target_depth_m"],
        "liquid_surface_mask.npy": labels["mask"],
        "normal_camera.npy": labels["normal_camera"],
        "layer_depths_m.npy": labels["layer_depths_m"],
        "layer_valid.npy": labels["layer_valid"],
        "raw_depth_m.npy": sensor["raw_depth_m"],
        "uncertainty.npy": sensor["uncertainty"],
        "simulated_dropout_mask.npy": sensor["simulated_dropout_mask"],
        "simulated_highlight_mask.npy": sensor["simulated_highlight_mask"],
    }
    for name, value in files.items():
        np.save(sample_dir / name, value)


MANIFEST_FIELDS = (
    "rgb_path",
    "raw_depth_path",
    "target_depth_path",
    "mask_path",
    "normal_path",
    "split",
    "sequence_id",
    "difficulty_tags",
    "depth_scale_to_m",
    "scenario",
    "uncertainty_path",
    "layer_depths_path",
    "layer_valid_path",
    "metadata_path",
    "sensor_model",
)


def build_manifest(root: str | Path, output: str | Path) -> int:
    root, output = Path(root).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for metadata_path in sorted(root.glob("samples/*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sample_dir, paths = metadata_path.parent, metadata["paths"]

        def relative(name: str) -> str:
            return os.path.relpath(sample_dir / paths[name], output.parent)

        rows.append(
            {
                "rgb_path": relative("rgb"),
                "raw_depth_path": relative("raw_depth"),
                "target_depth_path": relative("target_depth"),
                "mask_path": relative("mask"),
                "normal_path": relative("normal"),
                "split": str(metadata["split"]),
                "sequence_id": str(metadata["sequence_id"]),
                "difficulty_tags": str(metadata["difficulty_tags"]),
                "depth_scale_to_m": "1.0",
                "scenario": str(metadata["scenario"]),
                "uncertainty_path": relative("uncertainty"),
                "layer_depths_path": relative("layer_depths"),
                "layer_valid_path": relative("layer_valid"),
                "metadata_path": os.path.relpath(metadata_path, output.parent),
                "sensor_model": str(metadata["sensor_model"]),
            }
        )
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
