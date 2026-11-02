"""Autonomous navigation demo: Gazebo + AdaptMap + Nav2 + RViz.

Click "2D Goal Pose" in RViz (or run tools/nav_scenarios.py); Nav2 plans on AdaptMap's costmaps.
Args: world:=urban.sdf  rviz:=true  robot:=rover|ugv (the profile's `nav2:` footprint and planner radius are
written over params/nav2.yaml)
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from nav2_common.launch import RewrittenYaml

ROOT = os.environ.get("ADAPTMAP_ROOT", os.path.expanduser("~/Projects/Project_Focal"))


def _nav2(context):
    nav2 = os.path.join(get_package_share_directory("nav2_bringup"), "launch", "navigation_launch.py")
    params = os.path.join(get_package_share_directory("adaptmap_nav"), "params", "nav2.yaml")
    robot = LaunchConfiguration("robot").perform(context)
    with open(os.path.join(ROOT, "configs", "robots", f"{robot}.yaml")) as f:
        prof = yaml.safe_load(f).get("nav2", {})
    # footprint only exists in the local costmap, robot_radius only in the global one (nav2.yaml)
    params = RewrittenYaml(source_file=params, param_rewrites={k: str(v) for k, v in prof.items()}, convert_types=True)
    return [IncludeLaunchDescription(PythonLaunchDescriptionSource(nav2), launch_arguments={
        "use_sim_time": "true", "params_file": params, "autostart": "true",
    }.items())]


def generate_launch_description():
    adaptmap = os.path.join(get_package_share_directory("adaptmap_ros"), "launch", "adaptmap.launch.py")
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="urban.sdf"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("robot", default_value="rover"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(adaptmap), launch_arguments={
            "drive": "false", "world": LaunchConfiguration("world"), "rviz": LaunchConfiguration("rviz"),
            "robot": LaunchConfiguration("robot"),
        }.items()),
        OpaqueFunction(function=_nav2),
    ])
