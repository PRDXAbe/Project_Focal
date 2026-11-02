"""Gazebo urban world + robot + ros_gz bridge + static TFs.

world:=<name or path>  'urban.sdf' (default demo world), 'urban_s11.sdf', or an absolute .sdf path.
robot:=<name>          robot profile (configs/robots/<name>.yaml): 'rover' (default) or 'ugv'. The worlds include
                       model://rover; for another robot, robots/<name>/rover/ goes first on the model path.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROOT = os.environ.get("ADAPTMAP_ROOT", os.path.expanduser("~/Projects/Project_Focal"))


def _gazebo(context):
    share = get_package_share_directory("adaptmap_sim")
    world = LaunchConfiguration("world").perform(context)
    if not os.path.isabs(world):
        # worlds generated after the last colcon build aren't installed yet: fall back to the source tree
        installed = os.path.join(share, "worlds", world)
        src = os.path.join(ROOT, "ros2_ws", "src", "adaptmap_sim", "worlds", world)
        world = installed if os.path.exists(installed) else src
    robot = LaunchConfiguration("robot").perform(context)
    with open(os.path.join(ROOT, "configs", "robots", f"{robot}.yaml")) as f:
        x, y, z = yaml.safe_load(f)["lidar_offset"]
    path = [os.path.join(share, "models")]
    if robot != "rover":
        path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "adaptmap_sim", "robots", robot))
    res = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    return [
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", ":".join(path) + (":" + res if res else "")),
        ExecuteProcess(cmd=["gz", "sim", "-r", world], output="screen"),
        # gz odometry reports the model origin (ground level) as base_link; the LiDAR sits at lidar_offset
        Node(package="tf2_ros", executable="static_transform_publisher", parameters=[{"use_sim_time": True}],
             arguments=["--x", str(x), "--y", str(y), "--z", str(z), "--frame-id", "base_link",
                        "--child-frame-id", "lidar_link"]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="urban.sdf"),
        DeclareLaunchArgument("robot", default_value="rover"),
        OpaqueFunction(function=_gazebo),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen", arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ], parameters=[{"use_sim_time": True}]),
    ])
