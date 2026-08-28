#!/usr/bin/env python3
"""Blender background entry point for project-specific RGB-D liquid simulation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from liquid_depth.simulation import (  # noqa: E402
    build_manifest,
    camera_to_world,
    render_geometric_labels,
    sample_scene,
    scene_metadata,
    simulate_raw_depth,
    surface_height_and_gradient,
    write_sample_arrays,
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--min-distance-m", type=float, default=0.1)
    parser.add_argument("--max-distance-m", type=float, default=10.0)
    parser.add_argument("--engine", choices=("eevee", "cycles"), default="eevee")
    parser.add_argument("--render-samples", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def _set_input(node, name: str, value) -> bool:
    socket = node.inputs.get(name)
    if socket is None:
        return False
    socket.default_value = value
    return True


def principled_material(
    name: str,
    color: tuple[float, float, float],
    *,
    roughness: float,
    transmission: float = 0.0,
    ior: float = 1.45,
    metallic: float = 0.0,
    alpha: float = 1.0,
):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    _set_input(bsdf, "Base Color", (*color, 1.0))
    _set_input(bsdf, "Roughness", roughness)
    _set_input(bsdf, "Metallic", metallic)
    _set_input(bsdf, "IOR", ior)
    _set_input(bsdf, "Alpha", alpha)
    if not _set_input(bsdf, "Transmission Weight", transmission):
        _set_input(bsdf, "Transmission", transmission)
    if alpha < 1.0:
        material.surface_render_method = "DITHERED"
    return material


def create_container(scene):
    segments = 128
    rx, ry = scene.surface_radius_x_m, scene.surface_radius_y_m
    thickness = scene.wall_thickness_m
    outer_rx, outer_ry = rx + thickness, ry + thickness
    z_bottom, z_top = scene.container_bottom_z_m, scene.container_rim_z_m
    vertices: list[tuple[float, float, float]] = []
    for ring_rx, ring_ry, z in (
        (outer_rx, outer_ry, z_bottom),
        (outer_rx, outer_ry, z_top),
        (rx, ry, z_bottom + thickness),
        (rx, ry, z_top),
    ):
        vertices.extend(
            (ring_rx * math.cos(2 * math.pi * i / segments), ring_ry * math.sin(2 * math.pi * i / segments), z)
            for i in range(segments)
        )
    faces = []
    ob, ot, ib, it = (0, segments, 2 * segments, 3 * segments)
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend(
            (
                (ob + i, ob + j, ot + j, ot + i),
                (ib + i, it + i, it + j, ib + j),
                (ob + i, ib + i, ib + j, ob + j),
                (ot + i, ot + j, it + j, it + i),
            )
        )
    mesh = bpy.data.meshes.new("container_shell_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("transparent_container", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(
        principled_material(
            "container_glass",
            (0.92, 0.97, 1.0),
            roughness=scene.glass_roughness,
            transmission=1.0,
            ior=scene.glass_ior,
        )
    )
    bevel = obj.modifiers.new("edge_bevel", "BEVEL")
    bevel.width = max(thickness * 0.3, 0.0003)
    bevel.segments = 2
    return obj


def create_liquid(scene):
    segments, rings = 128, 20
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, float(surface_height_and_gradient(scene, np.array(0.0), np.array(0.0))[0]))]
    for ring in range(1, rings + 1):
        radius = ring / rings
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = scene.surface_radius_x_m * radius * math.cos(angle)
            y = scene.surface_radius_y_m * radius * math.sin(angle)
            z = float(surface_height_and_gradient(scene, np.array(x), np.array(y))[0])
            vertices.append((x, y, z))
    faces = []
    for i in range(segments):
        faces.append((0, 1 + i, 1 + (i + 1) % segments))
    for ring in range(1, rings):
        inner = 1 + (ring - 1) * segments
        outer = 1 + ring * segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.extend(((inner + i, outer + i, outer + j), (inner + i, outer + j, inner + j)))
    top_outer = 1 + (rings - 1) * segments
    bottom_start = len(vertices)
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        vertices.append(
            (
                scene.surface_radius_x_m * math.cos(angle),
                scene.surface_radius_y_m * math.sin(angle),
                scene.container_bottom_z_m + scene.wall_thickness_m,
            )
        )
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, scene.container_bottom_z_m + scene.wall_thickness_m))
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((top_outer + i, bottom_start + i, bottom_start + j, top_outer + j))
        faces.append((bottom_center, bottom_start + j, bottom_start + i))
    mesh = bpy.data.meshes.new("liquid_volume_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("liquid_volume", mesh)
    bpy.context.collection.objects.link(obj)
    base = tuple(float(value) for value in scene.liquid_color_rgb)
    obj.data.materials.append(
        principled_material(
            "liquid_optical_material",
            base,
            roughness=scene.liquid_roughness,
            transmission=max(0.15, 1.0 - scene.liquid_turbidity),
            ior=scene.liquid_ior,
            alpha=max(0.25, 0.95 - 0.7 * scene.liquid_turbidity),
        )
    )
    return obj


def create_floating_objects(scene, rng: random.Random) -> None:
    material = principled_material(
        "floating_material",
        (rng.uniform(0.05, 0.8), rng.uniform(0.05, 0.8), rng.uniform(0.05, 0.8)),
        roughness=rng.uniform(0.35, 0.9),
    )
    for index in range(scene.floating_object_count):
        angle, radius = rng.uniform(0, 2 * math.pi), math.sqrt(rng.uniform(0.0, 0.78))
        x = scene.surface_radius_x_m * radius * math.cos(angle)
        y = scene.surface_radius_y_m * radius * math.sin(angle)
        z = float(surface_height_and_gradient(scene, np.array(x), np.array(y))[0])
        size = scene.surface_radius_x_m * rng.uniform(0.018, 0.065)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=size, location=(x, y, z + size * 0.45))
        obj = bpy.context.object
        obj.name = f"floating_object_{index:02d}"
        obj.scale.z = rng.uniform(0.15, 0.55)
        obj.data.materials.append(material)


def create_environment(scene, rng: random.Random) -> None:
    table_material = principled_material(
        "industrial_table",
        (rng.uniform(0.08, 0.35),) * 3,
        roughness=rng.uniform(0.25, 0.8),
        metallic=rng.choice((0.0, 0.0, 0.65)),
    )
    radius = max(scene.surface_radius_x_m, scene.surface_radius_y_m)
    bpy.ops.mesh.primitive_cube_add(
        location=(0.0, 0.0, scene.container_bottom_z_m - radius * 0.08),
        scale=(radius * 4.0, radius * 4.0, radius * 0.08),
    )
    bpy.context.object.data.materials.append(table_material)
    for index in range(5):
        angle = rng.uniform(0, 2 * math.pi)
        distance = radius * rng.uniform(2.0, 4.0)
        bpy.ops.mesh.primitive_cube_add(
            location=(distance * math.cos(angle), distance * math.sin(angle), scene.container_bottom_z_m + radius * rng.uniform(-0.2, 1.2)),
            scale=(radius * rng.uniform(0.15, 0.7),) * 3,
        )
        obj = bpy.context.object
        obj.name = f"background_distractor_{index:02d}"
        obj.rotation_euler = tuple(rng.uniform(-1.0, 1.0) for _ in range(3))
        obj.data.materials.append(
            principled_material(
                f"distractor_material_{index}",
                tuple(rng.uniform(0.04, 0.95) for _ in range(3)),
                roughness=rng.uniform(0.1, 0.95),
                metallic=rng.uniform(0.0, 0.8),
            )
        )


def create_camera(scene):
    data = bpy.data.cameras.new("rgbd_camera")
    camera = bpy.data.objects.new("rgbd_camera", data)
    bpy.context.collection.objects.link(camera)
    camera.matrix_world = Matrix(camera_to_world(scene).tolist())
    data.sensor_width = 36.0
    data.lens = 36.0 / (2.0 * math.tan(math.radians(scene.horizontal_fov_deg / 2.0)))
    data.clip_start = max(0.005, 0.02 * min(np.linalg.norm(scene.camera_position_m), 1.0))
    data.clip_end = 20.0
    bpy.context.scene.camera = camera
    return camera


def create_lighting(scene, rng: random.Random) -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        rng.uniform(0.5, 1.0), rng.uniform(0.5, 1.0), rng.uniform(0.5, 1.0), 1.0
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.08 * scene.light_level
    radius = max(scene.surface_radius_x_m, scene.surface_radius_y_m)
    count = 3 if scene.scenario in {"glare", "compound"} else 2
    for index in range(count):
        data = bpy.data.lights.new(f"area_light_{index}", "AREA")
        data.energy = radius * radius * scene.light_level * (1800.0 if index == 0 and scene.scenario in {"glare", "compound"} else 500.0)
        data.shape = "DISK"
        data.size = radius * (0.25 if index == 0 and scene.scenario in {"glare", "compound"} else rng.uniform(1.0, 3.0))
        obj = bpy.data.objects.new(f"area_light_{index}", data)
        bpy.context.collection.objects.link(obj)
        angle = rng.uniform(0, 2 * math.pi)
        obj.location = (radius * rng.uniform(1.0, 4.0) * math.cos(angle), radius * rng.uniform(1.0, 4.0) * math.sin(angle), radius * rng.uniform(2.0, 6.0))
        direction = np.asarray((0.0, 0.0, 0.0)) - np.asarray(obj.location)
        obj.rotation_euler = tuple(np.asarray(direction.tolist()))
        obj.rotation_euler = direction_to_euler(direction)


def direction_to_euler(direction: np.ndarray):
    from mathutils import Vector

    return Vector(direction.tolist()).to_track_quat("-Z", "Y").to_euler()


def configure_render(scene, args: argparse.Namespace) -> str:
    render = bpy.context.scene
    render.render.resolution_x = scene.width
    render.render.resolution_y = scene.height
    render.render.resolution_percentage = 100
    render.render.image_settings.file_format = "PNG"
    render.render.image_settings.color_mode = "RGB"
    render.render.film_transparent = False
    render.view_settings.look = "AgX - Medium High Contrast"
    render.view_settings.exposure = scene.exposure
    if args.engine == "cycles":
        render.render.engine = "BLENDER_EEVEE_NEXT"
        try:
            preferences = bpy.context.preferences.addons["cycles"].preferences
            preferences.compute_device_type = "OPTIX"
            preferences.get_devices()
            for device in preferences.devices:
                device.use = True
            render.render.engine = "CYCLES"
            render.cycles.device = "GPU"
            render.cycles.samples = args.render_samples
            render.cycles.use_denoising = True
            return "cycles_optix"
        except Exception as exc:
            print(f"Cycles GPU unavailable, falling back to Eevee: {exc}", flush=True)
    render.render.engine = "BLENDER_EEVEE_NEXT"
    render.render.image_settings.color_depth = "8"
    return "eevee"


def generate_one(args: argparse.Namespace, index: int) -> None:
    sample_dir = args.output_root / "samples" / f"{index:08d}"
    metadata_path = sample_dir / "metadata.json"
    rgb_path = sample_dir / "rgb.png"
    if metadata_path.is_file() and rgb_path.is_file() and not args.overwrite:
        print(json.dumps({"index": index, "status": "skipped"}), flush=True)
        return
    synthetic = sample_scene(
        index,
        seed=args.seed,
        width=args.width,
        height=args.height,
        min_distance_m=args.min_distance_m,
        max_distance_m=args.max_distance_m,
    )
    rng = random.Random(args.seed + index * 104729)
    clear_scene()
    create_environment(synthetic, rng)
    create_container(synthetic)
    create_liquid(synthetic)
    create_floating_objects(synthetic, rng)
    create_camera(synthetic)
    create_lighting(synthetic, rng)
    backend = configure_render(synthetic, args)
    sample_dir.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(rgb_path)
    bpy.ops.render.render(write_still=True)
    labels = render_geometric_labels(synthetic)
    sensor = simulate_raw_depth(synthetic, labels)
    write_sample_arrays(sample_dir, labels, sensor)
    metadata = scene_metadata(synthetic, sample_dir)
    metadata["render_backend"] = backend
    metadata["render_samples"] = args.render_samples
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    mask_pixels = int(labels["mask"].sum())
    valid_raw = int(((sensor["raw_depth_m"] > 0) & (labels["mask"] > 0)).sum())
    print(
        json.dumps(
            {
                "index": index,
                "scenario": synthetic.scenario,
                "split": synthetic.split,
                "mask_pixels": mask_pixels,
                "raw_coverage": valid_raw / max(mask_pixels, 1),
                "backend": backend,
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.count <= 0:
        raise ValueError("--count must be positive")
    for index in range(args.start_index, args.start_index + args.count):
        generate_one(args, index)
    rows = build_manifest(args.output_root, args.output_root / "manifest.csv")
    print(json.dumps({"manifest": str(args.output_root / "manifest.csv"), "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
