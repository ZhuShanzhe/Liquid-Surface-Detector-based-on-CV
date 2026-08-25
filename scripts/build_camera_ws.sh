#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-humble}"

if [[ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  echo "ROS 2 ${ROS_DISTRO} is not installed. See docs/camera.md." >&2
  exit 1
fi

git -C "${REPO_ROOT}" submodule update --init --recursive
mkdir -p "${REPO_ROOT}/ros2_ws/src"
ln -sfn "${REPO_ROOT}/third_party/OrbbecSDK_ROS2" "${REPO_ROOT}/ros2_ws/src/OrbbecSDK_ROS2"

source "/opt/ros/${ROS_DISTRO}/setup.bash"
rosdep install --from-paths "${REPO_ROOT}/ros2_ws/src" --ignore-src --rosdistro "${ROS_DISTRO}" -r -y
cd "${REPO_ROOT}/ros2_ws"
colcon build --symlink-install --cmake-args -DBUILD_USB_PAL=ON -DPython3_EXECUTABLE=/usr/bin/python3

