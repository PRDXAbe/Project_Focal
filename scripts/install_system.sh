#!/usr/bin/env bash
# AdaptMap system deps: ROS 2 Humble + Gazebo Harmonic + grid_map. Run: sudo bash scripts/install_system.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y software-properties-common curl gnupg lsb-release ffmpeg x11-utils python3-venv python3-pip
add-apt-repository -y universe

# ROS 2 apt source
VER=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F tag_name | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${VER}/ros2-apt-source_${VER}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
apt-get install -y /tmp/ros2-apt-source.deb

# Gazebo (OSRF) apt source for Harmonic
curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" > /etc/apt/sources.list.d/gazebo-stable.list

apt-get update
apt-get install -y ros-humble-desktop ros-dev-tools python3-colcon-common-extensions \
  ros-humble-grid-map ros-humble-teleop-twist-keyboard ros-humble-tf-transformations \
  ros-humble-rosbag2-storage-mcap
apt-get install -y ros-humble-ros-gzharmonic || { echo "Harmonic failed -> Fortress"; apt-get install -y ros-humble-ros-gz; }
echo "DONE. Add to ~/.bashrc: source /opt/ros/humble/setup.bash"
