"""Gazebo urban world + rover + ros_gz bridge + static TFs.

world:=<name or path>  'urban.sdf' (default demo world), 'urban_s11.sdf', or an absolute .sdf path.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _gazebo(context):
    share = get_package_share_directory("adaptmap_sim")
    world = LaunchConfiguration("world").perform(context)
    if not os.path.isabs(world):
        # worlds generated after the last colcon build aren't installed yet: fall back to the source tree
        installed = os.path.join(share, "worlds", world)
        src = os.path.join(os.environ.get("ADAPTMAP_ROOT", os.path.expanduser("~/Projects/Project_Focal")),
                           "ros2_ws", "src", "adaptmap_sim", "worlds", world)
        world = installed if os.path.exists(installed) else src
    return [ExecuteProcess(cmd=["gz", "sim", "-r", world], output="screen")]


def generate_launch_description():
    share = get_package_share_directory("adaptmap_sim")
    models = os.path.join(share, "models")
    res = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="urban.sdf"),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", models + (":" + res if res else "")),
        OpaqueFunction(function=_gazebo),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen", arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ], parameters=[{"use_sim_time": True}]),
        # gz odometry reports the model origin (ground level) as base_link; lidar is 1.73 m above it
        Node(package="tf2_ros", executable="static_transform_publisher", parameters=[{"use_sim_time": True}],
             arguments=["--z", "1.73", "--frame-id", "base_link", "--child-frame-id", "lidar_link"]),
    ])
