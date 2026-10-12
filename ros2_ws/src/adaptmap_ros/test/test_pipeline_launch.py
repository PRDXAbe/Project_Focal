"""End-to-end smoke test: Gazebo + AdaptMap node (scripted lap, no RViz) must publish a sensible
costmap and map cells within two minutes.

Run (ROS + workspace sourced):  launch_test ros2_ws/src/adaptmap_ros/test/test_pipeline_launch.py
"""
import os
import time
import unittest

import launch
import launch_testing
import numpy as np
import pytest
import rclpy
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


@pytest.mark.launch_test
def generate_test_description():
    launch_file = os.path.join(get_package_share_directory("adaptmap_ros"), "launch", "adaptmap.launch.py")
    return launch.LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(launch_file),
                                 launch_arguments={"rviz": "false", "drive": "true"}.items()),
        launch_testing.actions.ReadyToTest(),
    ])


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node("adaptmap_smoke_test")

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def _wait_for(self, msg_type, topic, timeout=150.0, accept=lambda m: True):
        got = []
        sub = self.node.create_subscription(msg_type, topic, lambda m: got.append(m) if accept(m) else None, 1)
        end = time.time() + timeout
        while not got and time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=0.2)
        self.node.destroy_subscription(sub)
        self.assertTrue(got, f"nothing on {topic} within {timeout} s")
        return got[0]

    def test_costmap_has_free_and_blocked_cells(self):
        m = self._wait_for(OccupancyGrid, "/adaptmap/costmap",
                           accept=lambda m: (np.frombuffer(bytes(m.data), np.int8) >= 0).sum() > 10000)
        a = np.frombuffer(bytes(m.data), np.int8)
        self.assertAlmostEqual(m.info.resolution, 0.2, places=3)
        self.assertGreater(((a >= 0) & (a < 20)).sum(), 1000, "no low-cost (drivable) cells")
        self.assertGreater((a == 100).sum(), 100, "no lethal cells (walls, curbs, obstacles)")

    def test_near_zone_cells_published(self):
        m = self._wait_for(PointCloud2, "/adaptmap/cells/near", accept=lambda m: m.width > 10000)
        self.assertGreater(m.width, 10000)

    def test_stats_report_realtime_pipeline(self):
        import json
        s = json.loads(self._wait_for(String, "/adaptmap/stats", accept=lambda m: json.loads(m.data)["frame"] > 20).data)
        self.assertEqual(s["total_cells"], 702400)
        self.assertLess(s["timing_ms"]["total"], 500.0)
