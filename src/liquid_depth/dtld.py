from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

MANIFEST_FIELDS = (
    "frame_id",
    "split",
    "sequence_id",
    "scenario",
    "rgb_path",
    "raw_depth_path",
    "mask_path",
    "visible_mask_path",
    "object_id",
    "instance_index",
    "liquid_height_mm",
    "depth_cm",
    "liquid_label_json",
    "camera_json",
    "pose_json",
    "object_info_json",
    "difficulty_tags",
    "depth_scale_to_m",
)
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tiff", ".tif")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _load_json(path: Path, required: bool = True) -> dict:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        if required:
            raise
        # Some official DTLD releases contain a truncated optional metadata file.
        return {}


def _frame_item(payload: dict, frame_key: str):
    return payload.get(frame_key, payload.get(str(int(frame_key)), {}))


def _has_image_signature(path: Path) -> bool:
    with path.open("rb") as stream:
        header = stream.read(8)
    suffix = path.suffix.lower()
    if suffix == ".png":
        return header == _PNG_SIGNATURE
    if suffix in {".jpg", ".jpeg"}:
        return header.startswith(b"\xff\xd8")
    return header.startswith((b"II*\x00", b"MM\x00*"))


def _image_path(directory: Path, stem: str, required: bool) -> Path | None:
    for suffix in _IMAGE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file() and _has_image_signature(candidate):
            return candidate.resolve()
    if required:
        raise FileNotFoundError(f"No image for {directory / stem}")
    return None


def _instance_mask(
    directory: Path,
    frame_stem: str,
    instance_index: int,
    required: bool,
) -> Path | None:
    return _image_path(
        directory,
        f"{frame_stem}_{instance_index:06d}",
        required,
    )


def _split_name(
    root: Path,
    scene_dir: Path,
    split_map: dict[str, str] | None = None,
) -> str:
    relative = scene_dir.relative_to(root)
    if split_map is not None and relative.as_posix() in split_map:
        split = split_map[relative.as_posix()]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {split!r} for DTLD scene {relative}")
        return split
    parts = relative.parts
    for name in ("train", "val", "validation", "test"):
        if name in parts:
            return "val" if name == "validation" else name
    return "unspecified"


def _translation_distance(annotation: dict, pose: dict) -> float:
    annotation_t = annotation.get("cam_t_m2c")
    pose_t = pose.get("cam_t_m2c")
    if not isinstance(annotation_t, list) or not isinstance(pose_t, list):
        return float("inf")
    if len(annotation_t) != 3 or len(pose_t) != 3:
        return float("inf")
    return sum((float(left) - float(right)) ** 2 for left, right in zip(annotation_t, pose_t))


def _align_liquid_annotations(
    annotations: list[dict],
    poses: list[dict],
) -> list[tuple[int, dict, dict]]:
    if not poses:
        return [(index, annotation, {}) for index, annotation in enumerate(annotations)]
    unused = set(range(len(annotations)))
    aligned = []
    for instance_index, pose in enumerate(poses):
        candidates = [index for index in unused if annotations[index].get("obj_id") == pose.get("obj_id")]
        if not candidates:
            continue
        annotation_index = min(
            candidates,
            key=lambda index: _translation_distance(annotations[index], pose),
        )
        unused.remove(annotation_index)
        aligned.append((instance_index, annotations[annotation_index], pose))
    return aligned


def build_dtld_rows(
    root: str | Path,
    *,
    allow_missing: bool = False,
    split_map: dict[str, str] | None = None,
) -> list[dict[str, str | int | float]]:
    root = Path(root).resolve()
    scene_dirs = sorted(
        path.parent for path in root.rglob("rgb") if path.is_dir() and (path.parent / "depth").is_dir()
    )
    liquid_files = []
    for scene_dir in scene_dirs:
        liquid_path = scene_dir / "scene_gt_liquid.json"
        if not liquid_path.is_file():
            liquid_path = scene_dir / "liquid_label" / "scene_gt_liquid.json"
        if liquid_path.is_file():
            liquid_files.append((scene_dir, liquid_path))
    if not liquid_files:
        raise FileNotFoundError(f"No scene_gt_liquid.json found under {root}")

    rows: list[dict[str, str | int | float]] = []
    for scene_dir, liquid_path in liquid_files:
        split = _split_name(root, scene_dir, split_map)
        sequence_id = scene_dir.relative_to(root).as_posix()
        liquid_gt = _load_json(liquid_path)
        camera_gt = _load_json(scene_dir / "scene_camera.json", required=False)
        pose_gt = _load_json(scene_dir / "scene_gt.json", required=False)
        info_gt = _load_json(scene_dir / "scene_gt_info.json", required=False)

        for frame_key, liquid_annotations in sorted(
            liquid_gt.items(),
            key=lambda item: int(item[0]),
        ):
            frame_stem = f"{int(frame_key):06d}"
            rgb_path = _image_path(
                scene_dir / "rgb",
                frame_stem,
                False,
            )
            depth_path = _image_path(
                scene_dir / "depth",
                frame_stem,
                False,
            )
            if not allow_missing and (rgb_path is None or depth_path is None):
                continue
            camera = _frame_item(camera_gt, frame_key)
            poses = _frame_item(pose_gt, frame_key) or []
            infos = _frame_item(info_gt, frame_key) or []
            if not isinstance(liquid_annotations, list):
                liquid_annotations = [liquid_annotations]

            for instance_index, annotation, pose in _align_liquid_annotations(liquid_annotations, poses):
                info = infos[instance_index] if instance_index < len(infos) else {}
                object_id = annotation.get("obj_id", pose.get("obj_id", ""))
                liquid_height_mm = float(annotation["liquid_h"])
                mask_path = _instance_mask(
                    scene_dir / "mask",
                    frame_stem,
                    instance_index,
                    False,
                )
                visible_mask_path = _instance_mask(
                    scene_dir / "mask_visib",
                    frame_stem,
                    instance_index,
                    False,
                )
                declared_scale = float(camera.get("depth_scale", 0.001))
                depth_scale = declared_scale if declared_scale < 0.1 else declared_scale / 1000.0
                rows.append(
                    {
                        "frame_id": (
                            f"dtld_{sequence_id.replace('/', '_')}_{frame_stem}_{instance_index:06d}"
                        ),
                        "split": split,
                        "sequence_id": sequence_id,
                        "scenario": f"dtld_{split}",
                        "rgb_path": str(rgb_path or ""),
                        "raw_depth_path": str(depth_path or ""),
                        "mask_path": str(mask_path or ""),
                        "visible_mask_path": str(visible_mask_path or ""),
                        "object_id": object_id,
                        "instance_index": instance_index,
                        "liquid_height_mm": liquid_height_mm,
                        "depth_cm": liquid_height_mm / 10.0,
                        "liquid_label_json": json.dumps(
                            annotation.get("liquid_label", {}),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "camera_json": json.dumps(
                            camera,
                            separators=(",", ":"),
                        ),
                        "pose_json": json.dumps(
                            pose,
                            separators=(",", ":"),
                        ),
                        "object_info_json": json.dumps(
                            info,
                            separators=(",", ":"),
                        ),
                        "difficulty_tags": ("transparent;container_edge;non_lambertian"),
                        "depth_scale_to_m": depth_scale,
                    }
                )
    if not rows:
        raise ValueError(f"No DTLD annotations found under {root}")
    return rows


def write_dtld_manifest(
    root: str | Path,
    output: str | Path,
    *,
    allow_missing: bool = False,
    split_map: dict[str, str] | None = None,
) -> dict[str, int]:
    rows = build_dtld_rows(
        root,
        allow_missing=allow_missing,
        split_map=split_map,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return dict(Counter(str(row["split"]) for row in rows))
