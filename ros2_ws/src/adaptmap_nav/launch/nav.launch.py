"""Autonomous navigation demo: Gazebo + AdaptMap + Nav2 + RViz.

Click "2D Goal Pose" in RViz (or run tools/nav_scenarios.py); Nav2 plans on /adaptmap/costmap.
Args: world:=urban.sdf  rviz:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    adaptmap = os.path.join(get_package_share_directory("adaptmap_ros"), "launch", "adaptmap.launch.py")
    nav2 = os.path.join(get_package_share_directory("nav2_bringup"), "launch", "navigation_launch.py")
    params = os.path.join(get_package_share_directory("adaptmap_nav"), "params", "nav2.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="urban.sdf"),
        DeclareLaunchArgument("rviz", default_value="true"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(adaptmap), launch_arguments={
            "drive": "false", "world": LaunchConfiguration("world"), "rviz": LaunchConfiguration("rviz"),
        }.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(nav2), launch_arguments={
            "use_sim_time": "true", "params_file": params, "autostart": "true",
        }.items()),
    ])
