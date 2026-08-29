# Orbbec Gemini 2 acquisition

The acquisition component targets Ubuntu 22.04, ROS 2 Humble, and the Orbbec Gemini 2. The upstream driver is pinned
as the `third_party/OrbbecSDK_ROS2` Git submodule instead of copying its build products into this repository.

## Build

```bash
sudo apt install ros-humble-desktop python3-colcon-common-extensions python3-rosdep python3-vcstool
./scripts/build_camera_ws.sh
source ros2_ws/install/setup.bash
```

Install the upstream udev rules on the physical camera machine, reconnect the camera, then launch:

```bash
ros2 launch orbbec_camera gemini2.launch.py
ros2 run liquid_depth_camera capture_node --ros-args --params-file \
  ros2_ws/src/liquid_depth_camera/config/camera.yaml
```

Capture a synchronized frame without a GUI:

```bash
ros2 service call /rgbd_frame_saver/save std_srvs/srv/Trigger
```

The cloud GPU server cannot directly see a USB camera attached to another computer. Upload recorded frame directories,
or run the inference component on the camera machine after exporting a trained TorchScript model.

## Per-camera depth qualification

Before deployment, run the five-distance diffuse-plane protocol described in
[camera_plane_qualification.md](camera_plane_qualification.md). It generates an
independent holdout report and an optional verified depth correction.

