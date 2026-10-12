#!/usr/bin/env bash
# Nav2 for AdaptMap's autonomous-navigation phase. Run: sudo bash scripts/install_nav2.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-nav2-smac-planner \
  ros-humble-nav2-regulated-pure-pursuit-controller ros-humble-launch-testing-ament-cmake
echo "DONE: Nav2 installed."
