from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger


class RGBDFrameSaver(Node):
    def __init__(self) -> None:
        super().__init__("rgbd_frame_saver")
        defaults = {
            "output_dir": "./data/captures",
            "color_topic": "/camera/color/image_raw",
            "depth_topic": "/camera/depth/image_raw",
            "color_info_topic": "/camera/color/camera_info",
            "depth_info_topic": "/camera/depth/camera_info",
            "queue_size": 10,
            "sync_slop_seconds": 0.1,
            "auto_save_interval_seconds": 0.0,
            "preview": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.output_dir = Path(self.get_parameter("output_dir").value).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        self.lock = Lock()
        self.latest = None
        self.color_info = None
        self.depth_info = None
        self.last_save_time = 0.0

        color_info_topic = self.get_parameter("color_info_topic").value
        depth_info_topic = self.get_parameter("depth_info_topic").value
        self.create_subscription(CameraInfo, color_info_topic, self._on_color_info, 10)
        self.create_subscription(CameraInfo, depth_info_topic, self._on_depth_info, 10)
        color = Subscriber(self, Image, self.get_parameter("color_topic").value)
        depth = Subscriber(self, Image, self.get_parameter("depth_topic").value)
        self.sync = ApproximateTimeSynchronizer(
            [color, depth],
            queue_size=int(self.get_parameter("queue_size").value),
            slop=float(self.get_parameter("sync_slop_seconds").value),
        )
        self.sync.registerCallback(self._on_rgbd)
        self.create_service(Trigger, "~/save", self._on_save)
        self.get_logger().info(f"Writing synchronized RGB-D frames to {self.output_dir}")
        self.get_logger().info("Call ~/save (std_srvs/srv/Trigger) to capture one frame")

    def _on_color_info(self, message: CameraInfo) -> None:
        self.color_info = message

    def _on_depth_info(self, message: CameraInfo) -> None:
        self.depth_info = message

    def _on_rgbd(self, color_message: Image, depth_message: Image) -> None:
        try:
            rgb = self.bridge.imgmsg_to_cv2(color_message, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_message, desired_encoding="passthrough")
            with self.lock:
                self.latest = (rgb.copy(), depth.copy())

            interval = float(self.get_parameter("auto_save_interval_seconds").value)
            now = time.monotonic()
            if interval > 0 and now - self.last_save_time >= interval:
                self._save(rgb, depth)
                self.last_save_time = now

            if bool(self.get_parameter("preview").value):
                cv2.imshow("RGB", rgb)
                cv2.imshow("Depth", self._depth_preview(depth))
                cv2.waitKey(1)
        except Exception as exc:  # noqa: BLE001 - keep the ROS callback alive
            self.get_logger().error(f"RGB-D callback failed: {exc}")

    def _on_save(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        with self.lock:
            latest = None if self.latest is None else (self.latest[0].copy(), self.latest[1].copy())
        if latest is None:
            response.success = False
            response.message = "No synchronized RGB-D frame has arrived"
            return response
        try:
            frame_dir = self._save(*latest)
            response.success = True
            response.message = str(frame_dir)
        except Exception as exc:  # noqa: BLE001 - report failure through the service
            response.success = False
            response.message = str(exc)
        return response

    @staticmethod
    def _camera_info(message: CameraInfo | None) -> dict | None:
        if message is None:
            return None
        return {
            "width": message.width,
            "height": message.height,
            "distortion_model": message.distortion_model,
            "D": list(message.d),
            "K": list(message.k),
            "R": list(message.r),
            "P": list(message.p),
        }

    def _save(self, rgb: np.ndarray, depth: np.ndarray) -> Path:
        frame_id = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
        target = self.output_dir / frame_id
        target.mkdir(parents=True, exist_ok=False)
        if not cv2.imwrite(str(target / "rgb.png"), rgb):
            raise OSError("Could not write rgb.png")
        np.save(target / "depth.npy", depth, allow_pickle=False)
        if not cv2.imwrite(str(target / "depth_vis.png"), self._depth_preview(depth)):
            raise OSError("Could not write depth_vis.png")
        for name, info in (
            ("color_info.json", self._camera_info(self.color_info)),
            ("depth_info.json", self._camera_info(self.depth_info)),
        ):
            if info is not None:
                (target / name).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        self.get_logger().info(f"Saved {target}")
        return target

    @staticmethod
    def _depth_preview(depth: np.ndarray) -> np.ndarray:
        values = depth.astype(np.float32)
        valid = np.isfinite(values) & (values > 0)
        if not np.any(valid):
            return np.zeros((*depth.shape, 3), np.uint8)
        low, high = np.percentile(values[valid], (2, 98))
        normalized = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
        image = (normalized * 255).astype(np.uint8)
        image[~valid] = 0
        return cv2.applyColorMap(image, cv2.COLORMAP_JET)


def main() -> None:
    rclpy.init()
    node = RGBDFrameSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
