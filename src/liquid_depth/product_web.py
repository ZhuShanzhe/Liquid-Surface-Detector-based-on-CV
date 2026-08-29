from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2

from .calibration import import_factory_calibration
from .io import load_frame
from .rail_calibration import fit_rail_calibration
from .rail_runtime import make_product_system
from .system_runtime import save_system_profile


class PanelState:
    def __init__(self, capture_dir: Path, output_dir: Path) -> None:
        self.capture_dir = capture_dir.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.frame_dir: Path | None = None
        self.frame_image: bytes | None = None
        self.overlay_image: bytes | None = None
        self.last_result: dict[str, Any] | None = None
        self.system_cache: dict[tuple[str, str | None, bool], Any] = {}
        self.lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.measurement_lock = threading.Lock()

    def system_for(
        self,
        profile_path: Path,
        device: str | None,
        temporal: bool,
    ) -> Any:
        key = (str(profile_path), device, temporal)
        with self.cache_lock:
            system = self.system_cache.get(key)
            if system is None:
                system = make_product_system(
                    profile_path,
                    device=device,
                    temporal=temporal,
                )
                self.system_cache[key] = system
            return system

    def invalidate_profile(self, profile_path: Path) -> None:
        target = str(profile_path)
        with self.cache_lock:
            self.system_cache = {key: value for key, value in self.system_cache.items() if key[0] != target}


def _complete_frame(path: Path) -> bool:
    return all((path / name).is_file() for name in ("rgb.png", "depth.npy", "depth_info.json"))


def _latest_frame(root: Path) -> Path:
    candidates = sorted(
        (item for item in root.iterdir() if item.is_dir() and _complete_frame(item)),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No complete RGB-D frame under {root}")
    return candidates[0]


def _source_from_payload(payload: dict[str, Any], default_root: Path) -> Path:
    requested = payload.get("frame_dir") or payload.get("path")
    if not requested:
        return _latest_frame(default_root)
    source = Path(str(requested)).expanduser().resolve()
    if _complete_frame(source):
        return source
    if source.is_dir():
        return _latest_frame(source)
    raise FileNotFoundError(f"RGB-D source is neither a complete frame nor a frame directory: {source}")


def _safe_profile_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Profile path must end in .yaml or .yml")
    return path


def create_rail_profile(payload: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    frame_dir = Path(payload["frame_dir"]).expanduser().resolve()
    frame = load_frame(frame_dir)
    camera = import_factory_calibration(frame_dir)
    crop = [int(value) for value in payload.get("crop_xyxy", payload.get("roi", []))]
    if len(crop) != 4:
        raise ValueError("crop_xyxy must contain x0,y0,x1,y1")
    x0, y0, x1, y1 = crop
    height, width = frame.rgb_bgr.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Selected region is outside the image")
    points = [
        {
            "x_px": float(item.get("x_px", item.get("u"))),
            "y_px": float(item.get("y_px", item.get("v"))),
            "depth_m": (
                float(item["depth_mm"]) / 1000.0 if "depth_mm" in item else float(item["known_depth_m"])
            ),
        }
        for item in payload["points"]
    ]
    rail = fit_rail_calibration(points, minimum_points=3)
    profile_path = _safe_profile_path(payload["profile_path"])
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path = profile_path.with_name(profile_path.stem + "_reference.png")
    if not cv2.imwrite(str(reference_path), frame.rgb_bgr):
        raise OSError(f"Could not write reference image {reference_path}")
    checkpoint_value = payload.get("checkpoint_path") or payload.get("checkpoint")
    if not checkpoint_value:
        raise ValueError("A model checkpoint is required")
    checkpoint = Path(str(checkpoint_value)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    complex_mode = str(payload.get("complex_scene_mode", "auto")).lower()
    if complex_mode not in {"off", "auto", "always"}:
        raise ValueError("complex_scene_mode must be off, auto, or always")
    profile = {
        "schema_version": 1,
        "name": profile_path.stem,
        "measurement": {
            "mode": "fixed_rail",
            "rail_calibration": rail,
            "reference_image_path": str(reference_path),
            "max_reference_motion_px": float(payload.get("max_reference_motion_px", 4.0)),
            "min_intersection_confidence": float(payload.get("min_intersection_confidence", 0.5)),
            "extrapolation_margin_px": float(payload.get("extrapolation_margin_px", 2.0)),
        },
        "camera": {
            **camera.to_dict(),
            "depth_scale_to_m": float(payload.get("depth_scale_to_m", payload.get("depth_scale", 0.001))),
            "depth_registered_to_color": True,
            "depth_correction": {"scale": 1.0, "offset_m": 0.0, "status": "not_verified"},
        },
        "perception": {
            "checkpoint_path": str(checkpoint),
            "object_index": int(payload["object_index"]),
            "device": payload.get("device", "cuda"),
            "crop_xyxy": crop,
        },
        "output_calibration": {
            "scale": 1.0,
            "offset_m": 0.0,
            "status": "rail_calibrated",
        },
        "temporal": {
            "process_variance_m2": 1e-6,
            "measurement_variance_m2": 2.5e-5,
            "gate_sigma": 3.5,
            "max_jump_m": 0.02,
            "min_confidence": 0.2,
        },
        "complex_scene": {
            "mode": complex_mode,
            "latency_budget_ms": 500.0,
            "enforce_latency_budget": True,
            "hold_frames": 8,
            "missing_model_policy": "reject",
            "auto": {
                "raw_depth_valid_ratio_below": 0.45,
                "saturated_pixel_ratio_above": 0.10,
                "luma_p50_below": 0.18,
                "dark_pixel_ratio_above": 0.70,
                "dynamic_range_below": 0.06,
            },
            "scene_context": {},
            "models": {},
        },
        "deployment": {
            "camera_motion_policy": "recalibrate_after_move",
            "continuous_motion_requires": ["container_cad", "fixed_pose_marker"],
        },
    }
    save_system_profile(profile_path, profile)
    return profile_path, rail


class PanelHandler(BaseHTTPRequestHandler):
    server_version = "LiquidDepthPanel/1"
    state: PanelState
    ui_root = Path(__file__).with_name("ui")

    def _json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("Invalid request size")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise TypeError("JSON body must be an object")
        return value

    def _image(self, data: bytes | None) -> None:
        if data is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            with self.state.lock:
                self._json(
                    {
                        "capture_dir": str(self.state.capture_dir),
                        "output_dir": str(self.state.output_dir),
                        "frame_dir": (str(self.state.frame_dir) if self.state.frame_dir else None),
                        "frame_id": (self.state.frame_dir.name if self.state.frame_dir else None),
                        "overlay_available": self.state.overlay_image is not None,
                        "last_result": self.state.last_result,
                    }
                )
            return
        if path == "/api/frame-image":
            with self.state.lock:
                self._image(self.state.frame_image)
            return
        if path == "/api/overlay-image":
            with self.state.lock:
                self._image(self.state.overlay_image)
            return
        relative = "index.html" if path == "/" else path.lstrip("/")
        candidate = (self.ui_root / relative).resolve()
        if self.ui_root.resolve() not in candidate.parents and candidate != self.ui_root.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        try:
            payload = self._body()
            path = urlparse(self.path).path
            if path == "/api/load-frame":
                source = _source_from_payload(payload, self.state.capture_dir)
                frame = load_frame(source)
                ok, encoded = cv2.imencode(".jpg", frame.rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise OSError("Could not encode frame")
                with self.state.lock:
                    self.state.frame_dir = source
                    self.state.frame_image = encoded.tobytes()
                self._json(
                    {
                        "ok": True,
                        "frame_dir": str(source),
                        "frame_id": source.name,
                        "width": frame.rgb_bgr.shape[1],
                        "height": frame.rgb_bgr.shape[0],
                    }
                )
                return
            if path == "/api/calibrate-rail":
                source = _source_from_payload(payload, self.state.capture_dir)
                payload["frame_dir"] = str(source)
                profile_path, rail = create_rail_profile(payload)
                self.state.invalidate_profile(profile_path)
                self._json(
                    {
                        "ok": True,
                        "profile_path": str(profile_path),
                        "calibration": rail,
                        "cross_validation_mae_mm": (float(rail.get("cross_validation_mae_m", 0.0)) * 1000.0),
                    }
                )
                return
            if path == "/api/measure":
                source = _source_from_payload(payload, self.state.capture_dir)
                profile_path = _safe_profile_path(payload["profile_path"])
                target = self.state.output_dir / source.name
                system = self.state.system_for(
                    profile_path,
                    payload.get("device"),
                    bool(payload.get("temporal", False)),
                )
                with self.state.measurement_lock:
                    result = system.measure(source, target)
                    overlay = cv2.imread(
                        str(target / "measurement_overlay.png"),
                        cv2.IMREAD_COLOR,
                    )
                    encoded_data = None
                    if overlay is not None:
                        ok, encoded = cv2.imencode(
                            ".jpg",
                            overlay,
                            [cv2.IMWRITE_JPEG_QUALITY, 92],
                        )
                        if ok:
                            encoded_data = encoded.tobytes()
                with self.state.lock:
                    self.state.frame_dir = source
                    self.state.overlay_image = encoded_data
                    self.state.last_result = result
                self._json({"ok": True, "output_dir": str(target), "result": result})
                return
            self._json({"ok": False, "error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - boundary returns errors to panel
            self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        print("[panel] " + format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Industrial RGB-D liquid-depth control panel")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--capture-dir", type=Path, default=Path("data/live"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/live"))
    args = parser.parse_args()
    state = PanelState(args.capture_dir, args.output_dir)
    handler = type("ConfiguredPanelHandler", (PanelHandler,), {"state": state})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Liquid depth panel: http://{args.host}:{args.port}")
    print("Keep this terminal open. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
