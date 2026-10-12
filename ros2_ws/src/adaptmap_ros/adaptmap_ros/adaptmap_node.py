"""AdaptMap ROS 2 node: /lidar/points -> preprocess -> segment -> foveated 2.5D semantic map.

All stages run in-process on the GPU (no point-cloud re-serialisation between stages).
Messages are filled from raw buffers (array.array / bytes): building per-element Python objects
holds the GIL for seconds per frame and starves the TF listener thread.

Publishes
  /adaptmap/grid/<zone>     grid_map_msgs/GridMap  (elevation, ground, std, class, confidence, color)
  /adaptmap/cells/<zone>    sensor_msgs/PointCloud2  class-coloured 2.5D cell centres at their height
  /adaptmap/labeled_cloud   sensor_msgs/PointCloud2  segmented points (odom frame, rgb)
  /adaptmap/nav_grid        nav_msgs/OccupancyGrid  fused obstacle grid (Nav2-ready hook)
  /adaptmap/stats           std_msgs/String  JSON per-frame timings + cell statistics
"""
import array
import json
import queue
import os
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
import collections
import threading

ROOT = Path(os.environ.get("ADAPTMAP_ROOT", Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(ROOT))
from adaptmap import classes  # noqa: E402
from adaptmap.config import load_config  # noqa: E402
from adaptmap.foveated_map import FoveatedMap  # noqa: E402
from adaptmap.preprocess import preprocess  # noqa: E402
from adaptmap.segmentation import build_segmenter  # noqa: E402

_PACKED_RGB = (classes.COLORS[:, 0].int() << 16 | classes.COLORS[:, 1].int() << 8 | classes.COLORS[:, 2].int())
CLASS_RGB_FLOAT = _PACKED_RGB.numpy().astype(np.uint32).view(np.float32)


def cloud_to_numpy(msg):
    """PointCloud2 with float32 x,y,z[,intensity] -> (N, 4) float32."""
    off = {f.name: f.offset for f in msg.fields}
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
    cols = [raw[:, off[k]:off[k] + 4].copy().view(np.float32)[:, 0] for k in ("x", "y", "z")]
    inten = raw[:, off["intensity"]:off["intensity"] + 4].copy().view(np.float32)[:, 0] if "intensity" in off else np.zeros(len(raw), np.float32)
    return np.stack(cols + [inten], 1)


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float32)


def grid_map_msg(view, name, header):
    """Our layer arrays are indexed [ix, iy] (x, y increasing). grid_map stores a column-major
    matrix whose (0, 0) is the (+x, +y) corner, so flip both axes then flatten column-major."""
    m = GridMap()
    m.header = header
    n, res = view["size"], view["resolution"]
    m.info.resolution = float(res)
    m.info.length_x = m.info.length_y = float(n * res)
    ox, oy = view["origin"]
    m.info.pose.position.x = ox + n * res / 2
    m.info.pose.position.y = oy + n * res / 2
    m.info.pose.orientation.w = 1.0
    cls = view["class"].cpu().numpy()
    valid = view["valid"].cpu().numpy()
    layers = {
        "elevation": view["elevation"].cpu().numpy(),
        "ground": view["ground"].cpu().numpy(),
        "std": view["std"].cpu().numpy(),
        "class": np.where(valid, cls.astype(np.float32), np.nan),
        "confidence": np.where(valid, view["confidence"].cpu().numpy(), np.nan),
        "color": np.where(valid, CLASS_RGB_FLOAT[cls], np.nan),
    }
    for k, a in layers.items():
        arr = Float32MultiArray()
        arr.layout.dim = [MultiArrayDimension(label="column_index", size=n, stride=n * n),
                          MultiArrayDimension(label="row_index", size=n, stride=n)]
        arr.data = array.array("f", np.ascontiguousarray(a[::-1, ::-1].T, dtype=np.float32).tobytes())
        m.layers.append(k)
        m.data.append(arr)
    m.basic_layers = ["elevation"]
    return m


def xyzrgb_cloud(xyz, labels, header):
    a = xyz.float().cpu().numpy()
    data = np.concatenate([a, CLASS_RGB_FLOAT[labels.cpu().numpy()][:, None]], 1).astype(np.float32)
    m = PointCloud2(header=header, height=1, width=len(data), is_bigendian=False, is_dense=True,
                    point_step=16, row_step=16 * len(data))
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(("x", "y", "z", "rgb"))]
    m.data = array.array("B", data.tobytes())  # array.array skips per-element validation in rclpy
    return m


class AdaptMapNode(Node):
    def __init__(self):
        super().__init__("adaptmap")
        self.declare_parameter("config", str(ROOT / "configs" / "adaptmap.yaml"))
        self.declare_parameter("map_frame", "odom")
        self.declare_parameter("cloud_topic", "/lidar/points")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("viz_every", 3)
        self.declare_parameter("stats_log", "")
        # sensor pose in base_link (the rover's fixed mast); avoids a TF listener, see on_odom
        self.declare_parameter("lidar_offset", [0.0, 0.0, 1.73])
        self.cfg = load_config(self.get_parameter("config").value)
        self.frame = self.get_parameter("map_frame").value
        self.viz_every = self.get_parameter("viz_every").value
        log = self.get_parameter("stats_log").value
        self.log = open(log, "w") if log else None

        self.dev = self.cfg["device"]
        self.seg = build_segmenter(self.cfg)
        self.map = FoveatedMap(self.cfg["zones"], self.cfg["fusion"], device=self.dev)
        self.get_logger().info(f"AdaptMap ready: {self.map.stats()['total_cells']} cells, model={self.cfg['model']['backend']}")

        # Robot pose from /odom (depth 1, own thread) into a short history; the Python TF listener
        # cannot keep up with 50 Hz /tf while the GPU thread runs, so its buffer lags ~2 s.
        self.lidar_offset = np.array(self.get_parameter("lidar_offset").value, np.float32)
        self.poses = collections.deque(maxlen=200)
        self.pose_lock = threading.Lock()
        latest = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self.on_odom, QoSProfile(depth=1),
                                 callback_group=MutuallyExclusiveCallbackGroup())
        self.sub = self.create_subscription(PointCloud2, self.get_parameter("cloud_topic").value, self.on_cloud, latest,
                                            callback_group=MutuallyExclusiveCallbackGroup())
        self.pub_grid = {z["name"]: self.create_publisher(GridMap, f"/adaptmap/grid/{z['name']}", 1) for z in self.cfg["zones"]}
        self.pub_cells = {z["name"]: self.create_publisher(PointCloud2, f"/adaptmap/cells/{z['name']}", 1) for z in self.cfg["zones"]}
        self.pub_cloud = self.create_publisher(PointCloud2, "/adaptmap/labeled_cloud", 1)
        self.pub_nav = self.create_publisher(OccupancyGrid, "/adaptmap/nav_grid", 1)
        self.pub_stats = self.create_publisher(String, "/adaptmap/stats", 10)
        self.frames = 0
        self.busy = False
        # Message building + publishing runs on its own thread so the mapping loop never waits on it.
        # 1-deep queue: if the publisher is still busy, that snapshot is dropped (the map keeps updating).
        self.pub_queue = queue.Queue(maxsize=1)
        self.pub_dropped = 0
        self.pub_ms = None
        threading.Thread(target=self.publish_worker, daemon=True).start()

    def on_odom(self, msg):
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        with self.pose_lock:
            self.poses.append((st, quat_to_rot(msg.pose.pose.orientation), np.array([p.x, p.y, p.z], np.float32)))

    def lookup(self, msg):
        """Sensor pose in the map frame at the cloud stamp (nearest odom sample)."""
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        for _ in range(20):  # wait up to 0.1 s for odometry to catch up with the cloud
            with self.pose_lock:
                if self.poses and self.poses[-1][0] >= st - 0.005:
                    break
            time.sleep(0.005)
        with self.pose_lock:
            if not self.poses:
                return None
            ps, R, t = min(self.poses, key=lambda q: abs(q[0] - st))
        if abs(ps - st) > 0.1:
            self.get_logger().warn(f"no odometry within 0.1 s of cloud (dt={ps - st:.2f} s)", throttle_duration_sec=2.0)
            return None
        return R, R @ self.lidar_offset + t

    def on_cloud(self, msg):
        if self.busy:
            return
        T = self.lookup(msg)
        if T is None:
            return
        self.busy = True
        try:
            self.process(msg, *T)
        finally:
            self.busy = False

    def process(self, msg, R, t):
        sync = torch.cuda.synchronize if self.dev.startswith("cuda") else (lambda: None)
        tm = {}
        t0 = time.perf_counter()
        pts = torch.from_numpy(cloud_to_numpy(msg)).to(self.dev)
        clean, pstats = preprocess(pts, self.cfg["preprocess"], self.cfg["zones"])
        sync(); tm["preprocess"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        labels, probs = self.seg.segment(clean)
        sync(); tm["segment"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        Rt, tt = torch.from_numpy(R).to(self.dev), torch.from_numpy(t).to(self.dev)
        xyz = clean[:, :3] @ Rt.T + tt
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.map.integrate(xyz, probs, tt[:2], stamp)
        sync(); tm["fuse"] = time.perf_counter() - t2
        tm["total"] = time.perf_counter() - t0

        self.frames += 1
        if self.frames % self.viz_every == 0:
            t3 = time.perf_counter()
            self.queue_publish(msg, xyz, labels)
            tm["snapshot"] = time.perf_counter() - t3

        s = self.map.stats()
        rec = {"stamp": stamp, "frame": self.frames, "timing_ms": {k: round(v * 1e3, 2) for k, v in tm.items()},
               "points": pstats, "occupied": {k: v["occupied"] for k, v in s["layers"].items()},
               "total_cells": s["total_cells"], "uniform_cells": s["uniform_cells"], "map_bytes": s["map_bytes"],
               "gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if self.dev.startswith("cuda") else 0,
               "publish_ms": self.pub_ms, "publish_dropped": self.pub_dropped}
        self.pub_stats.publish(String(data=json.dumps(rec)))
        if self.log:
            self.log.write(json.dumps(rec) + "\n")
            self.log.flush()
        if self.frames % 20 == 0:
            self.get_logger().info(f"frame {self.frames}: {rec['timing_ms']} ms, pts {pstats['out']}")

    # ---------------------------------------------------------------- publishing
    def queue_publish(self, msg, xyz, labels):
        """Snapshot the map on the mapping thread (fresh tensors, cheap) and hand it to the publisher."""
        header = msg.header
        header.frame_id = self.frame
        snap = (header, self.map.export(), self.map.nav_view(cell=0.2, half=50.0), xyz, labels)
        try:
            self.pub_queue.put_nowait(snap)
        except queue.Full:
            self.pub_dropped += 1

    def publish_worker(self):
        while rclpy.ok():
            try:
                header, export, nv, xyz, labels = self.pub_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            t = time.perf_counter()
            for name, view in export.items():
                self.pub_grid[name].publish(grid_map_msg(view, name, header))
                self.pub_cells[name].publish(self.cell_cloud(view, header))
            self.pub_cloud.publish(xyzrgb_cloud(xyz, labels, header))
            self.pub_nav.publish(self.nav_grid(nv, header))
            self.pub_ms = round((time.perf_counter() - t) * 1e3, 1)

    def cell_cloud(self, v, header):
        """Observed cells of one zone as points (cell centre, elevation, class colour)."""
        idx = v["valid"].nonzero()
        r = v["resolution"]
        ox, oy = v["origin"]
        xyz = torch.stack([ox + (idx[:, 0].float() + 0.5) * r, oy + (idx[:, 1].float() + 0.5) * r,
                           v["elevation"][idx[:, 0], idx[:, 1]]], 1)
        return xyzrgb_cloud(xyz, v["class"][idx[:, 0], idx[:, 1]], header)

    def nav_grid(self, nv, header):
        cls, el = nv["class"], nv["elevation"]
        known = torch.isfinite(el)
        ground = classes.GROUND.to(cls.device)[cls]
        occ = torch.full(cls.shape, -1, dtype=torch.int8, device=cls.device)
        occ[known & ground] = 0
        occ[known & ~ground & (cls != 0)] = 100
        g = OccupancyGrid(header=header)
        n = cls.shape[0]
        g.info.resolution = nv["resolution"]
        g.info.width = g.info.height = n
        g.info.origin.position.x, g.info.origin.position.y = nv["origin"]
        g.info.origin.orientation.w = 1.0
        g.data = array.array("b", occ.T.contiguous().cpu().numpy().tobytes())  # row-major, x fastest
        return g


def main():
    rclpy.init()
    node = AdaptMapNode()
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
