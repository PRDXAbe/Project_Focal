"""Ground-truth labels for simulated LiDAR from the world manifest, exported in SemanticKITTI format.

Each point gets the class of the smallest manifest box (oriented, yaw-only, +margin) containing it;
points outside all boxes near z=0 are terrain, others unlabeled. Actor boxes follow their trajectory
at the scan time.

Sources:
  --synth N          N scans from the analytic ray-caster along the demo drive line (no ROS needed)
  --bag PATH         a rosbag2 with /lidar/points + /odom recorded from Gazebo (needs ROS sourced)

Output: <out>/sequences/00/{velodyne/*.bin, labels/*.label}  (SemanticKITTI raw label ids)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from adaptmap import classes  # noqa: E402
from synth_scan import load_boxes, scan  # noqa: E402

# AdaptMap class -> SemanticKITTI raw label id (0 = unlabeled / ignored)
ADAPT_TO_RAW = {"unknown": 0, "road": 40, "sidewalk_curb": 48, "terrain": 72, "vegetation": 70,
                "building_wall": 50, "pole": 80, "vehicle": 10, "pedestrian": 30, "other_obstacle": 0}
RAW_LUT = torch.tensor([ADAPT_TO_RAW[n] for n in classes.NAMES], dtype=torch.int32)


def label_points(xyz_world, t, margin=0.05):
    """(N,3) world-frame points -> AdaptMap labels (N,)."""
    c, h, yaw, lab = load_boxes(t, device=xyz_world.device)
    vol = (h + margin).prod(1)
    best = torch.full((xyz_world.shape[0],), float("inf"), device=xyz_world.device)
    out = torch.where(xyz_world[:, 2] < 0.1, classes.NAMES.index("terrain"), 0).long()
    for i in range(0, c.shape[0], 16):
        cc, hh, yy, ll, vv = c[i:i + 16], h[i:i + 16] + margin, yaw[i:i + 16], lab[i:i + 16], vol[i:i + 16]
        d = xyz_world[None] - cc[:, None]
        cy, sy = yy.cos()[:, None], yy.sin()[:, None]
        lx, ly = cy * d[..., 0] + sy * d[..., 1], -sy * d[..., 0] + cy * d[..., 1]
        inside = (lx.abs() <= hh[:, None, 0]) & (ly.abs() <= hh[:, None, 1]) & (d[..., 2].abs() <= hh[:, None, 2])
        v = torch.where(inside, vv[:, None], torch.full_like(lx, float("inf")))
        vmin, bi = v.min(0)
        better = vmin < best
        best[better] = vmin[better]
        out[better] = ll[bi[better]]
    return out


def write(out, idx, pts, labels):
    seq = Path(out) / "sequences" / "00"
    (seq / "velodyne").mkdir(parents=True, exist_ok=True)
    (seq / "labels").mkdir(parents=True, exist_ok=True)
    pts.cpu().numpy().astype(np.float32).tofile(seq / "velodyne" / f"{idx:06d}.bin")
    RAW_LUT[labels.cpu()].numpy().astype(np.uint32).tofile(seq / "labels" / f"{idx:06d}.label")


def from_synth(n, out, speed=2.0):
    for i in range(n):
        t = i / 10.0
        x = speed * t
        o = [x, -1.0, 1.75]
        pts, lab = scan(o, load_boxes(t))
        write(out, i, pts, lab)
    print(f"wrote {n} synthetic scans to {out}")


def from_bag(bag, out, spawn):
    """Gazebo bag: points are in lidar_link (1.73 m above the model origin); gz odom = model pose in world."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "adaptmap_ros"))
    from adaptmap_ros.adaptmap_node import cloud_to_numpy, quat_to_rot

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag), rosbag2_py.ConverterOptions("", ""))
    odom, i = None, 0
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic == "/odom":
            odom = deserialize_message(data, Odometry)
        elif topic == "/lidar/points" and odom is not None:
            msg = deserialize_message(data, PointCloud2)
            p = cloud_to_numpy(msg)
            p = p[np.isfinite(p[:, :3]).all(1)]
            R = quat_to_rot(odom.pose.pose.orientation)
            tr = odom.pose.pose.position
            base = np.array([tr.x, tr.y, tr.z]) + np.array(spawn)
            world = (p[:, :3] + [0, 0, 1.73]) @ R.T + base
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            lab = label_points(torch.from_numpy(world.astype(np.float32)).cuda(), t)
            write(out, i, torch.from_numpy(p), lab)
            i += 1
    print(f"wrote {i} labelled Gazebo scans to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", type=int, default=0)
    ap.add_argument("--bag")
    ap.add_argument("--spawn", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="odom->world offset (gz odom is already world)")
    ap.add_argument("--out", default=str(ROOT / "data" / "sim_kitti"))
    a = ap.parse_args()
    if a.synth:
        from_synth(a.synth, a.out)
    if a.bag:
        from_bag(a.bag, a.out, a.spawn)


if __name__ == "__main__":
    main()
