"""Record auto-labelled training scans straight from a running Gazebo sim (no rosbag needed).

Every Nth LiDAR scan is preprocessed exactly like the live pipeline, labelled from the world's
ground-truth manifest at the scan's sim time (moving pedestrians included), and written as
  <out>/sequences/<world>/velodyne/NNNNNN.bin    float32 x, y, z, intensity (sensor frame)
  <out>/sequences/<world>/labels/NNNNNN.label    uint32 AdaptMap class id (0 = unlabelled)

Usage (sim running with that world, ROS sourced):
  .venv/bin/python tools/record_labeled.py --world urban_s11 --out data/sim_adapt --every 2 --stop-x 114
"""
import argparse
import collections
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "ros2_ws" / "src" / "adaptmap_ros")]
from adaptmap.config import load_config  # noqa: E402
from adaptmap.preprocess import preprocess  # noqa: E402
from adaptmap_ros.adaptmap_node import cloud_to_numpy, quat_to_rot  # noqa: E402
from autolabel import label_points  # noqa: E402
from synth_scan import manifest_for  # noqa: E402

LIDAR_Z = 1.73


class Recorder(Node):
    def __init__(self, a):
        super().__init__("adaptmap_record_labeled")
        self.a, self.cfg = a, load_config()
        self.manifest = manifest_for(a.world)
        self.dir = Path(a.out) / "sequences" / Path(a.world).stem
        (self.dir / "velodyne").mkdir(parents=True, exist_ok=True)
        (self.dir / "labels").mkdir(parents=True, exist_ok=True)
        self.n = len(list((self.dir / "velodyne").glob("*.bin")))  # append to an existing sequence
        self.seen, self.done = 0, False
        self.poses = collections.deque(maxlen=200)
        self.lock = threading.Lock()
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(PointCloud2, "/lidar/points", self.on_cloud,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))

    def on_odom(self, m):
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        with self.lock:
            self.poses.append((st, m.pose.pose))
        if m.pose.pose.position.x > self.a.stop_x:
            self.done = True

    def on_cloud(self, msg):
        self.seen += 1
        if self.done or self.seen % self.a.every:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            if not self.poses:
                return
            ts, pose = min(self.poses, key=lambda p: abs(p[0] - t))
        if abs(ts - t) > 0.05:
            return
        pts = torch.from_numpy(cloud_to_numpy(msg)).cuda()
        clean, _ = preprocess(pts, self.cfg["preprocess"], self.cfg["zones"])
        R = torch.from_numpy(quat_to_rot(pose.orientation)).cuda()
        p = pose.position
        world = (clean[:, :3] + torch.tensor([0, 0, LIDAR_Z], device="cuda")) @ R.T + torch.tensor([p.x, p.y, p.z], device="cuda")
        lab = label_points(world, t, manifest=self.manifest)
        clean.cpu().numpy().astype(np.float32).tofile(self.dir / "velodyne" / f"{self.n:06d}.bin")
        lab.cpu().numpy().astype(np.uint32).tofile(self.dir / "labels" / f"{self.n:06d}.label")
        self.n += 1
        if self.n % 50 == 0:
            self.get_logger().info(f"{self.n} scans written to {self.dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, help="world name, e.g. urban_s11 (selects its manifest)")
    ap.add_argument("--out", default=str(ROOT / "data" / "sim_adapt"))
    ap.add_argument("--every", type=int, default=2, help="keep every Nth scan")
    ap.add_argument("--stop-x", type=float, default=114.0, help="stop once the robot passes this x (end of lap)")
    ap.add_argument("--timeout", type=float, default=300.0)
    a = ap.parse_args()
    rclpy.init()
    node = Recorder(a)
    t0 = time.time()
    while rclpy.ok() and not node.done and time.time() - t0 < a.timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
    print(f"{node.n} labelled scans in {node.dir}", flush=True)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
