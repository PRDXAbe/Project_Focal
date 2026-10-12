"""Full demo: sim + AdaptMap node + RViz (+ optional scripted drive)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ROOT = os.environ.get("ADAPTMAP_ROOT", os.path.expanduser("~/Projects/Project_Focal"))


def generate_launch_description():
    sim = os.path.join(get_package_share_directory("adaptmap_sim"), "launch", "sim.launch.py")
    rviz = os.path.join(get_package_share_directory("adaptmap_ros"), "rviz", "adaptmap.rviz")
    st = {"use_sim_time": True}
    return LaunchDescription([
        DeclareLaunchArgument("drive", default_value="true"),
        DeclareLaunchArgument("stats_log", default_value=""),
        DeclareLaunchArgument("viz_every", default_value="3"),
        DeclareLaunchArgument("rviz", default_value="true"),
        SetEnvironmentVariable("ADAPTMAP_ROOT", ROOT),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(sim)),
        Node(package="adaptmap_ros", executable="adaptmap_node", output="screen",
             prefix=f"{ROOT}/.venv/bin/python",
             parameters=[st, {"stats_log": LaunchConfiguration("stats_log"),
                              "viz_every": ParameterValue(LaunchConfiguration("viz_every"), value_type=int)}]),
        Node(package="adaptmap_ros", executable="waypoint_driver", output="screen", parameters=[st],
             condition=IfCondition(LaunchConfiguration("drive"))),
        Node(package="rviz2", executable="rviz2", arguments=["-d", rviz], parameters=[st],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
