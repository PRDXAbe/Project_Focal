"""Segmentation accuracy on real Gazebo scans: auto-label a recorded bag from the world manifest,
run preprocess + segmentation, and report per-class IoU. Appends a section to docs/RESULTS.md.

Usage (ROS sourced): .venv/bin/python tools/eval_bag.py data/demo_*/bag [--every 3] [--no-write]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "ros2_ws" / "src" / "adaptmap_ros")]
from adaptmap import classes  # noqa: E402
from adaptmap.config import load_config  # noqa: E402
from adaptmap.preprocess import preprocess  # noqa: E402
from adaptmap.segmentation import build_segmenter  # noqa: E402
from autolabel import label_points  # noqa: E402

LIDAR_Z = 1.73


def iter_bag(bag, every):
    import rosbag2_py
    from nav_msgs.msg import Odometry
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    from adaptmap_ros.adaptmap_node import cloud_to_numpy, quat_to_rot

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    odom, k = [], 0
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic == "/odom":
            m = deserialize_message(data, Odometry)
            odom.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.pose.pose))
            odom = odom[-100:]
        elif topic == "/lidar/points" and odom:
            k += 1
            if k % every:
                continue
            msg = deserialize_message(data, PointCloud2)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            ts, pose = min(odom, key=lambda o: abs(o[0] - t))
            if abs(ts - t) > 0.05:
                continue
            p = cloud_to_numpy(msg)
            p = p[np.isfinite(p[:, :3]).all(1)]
            pos = pose.position
            yield t, p, quat_to_rot(pose.orientation), np.array([pos.x, pos.y, pos.z], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    seg = build_segmenter(cfg)
    C = classes.NUM_CLASSES
    conf = torch.zeros(C, C, dtype=torch.long, device="cuda")
    n = 0
    for t, p, R, base in iter_bag(a.bag, a.every):
        pts = torch.from_numpy(p).cuda()
        clean, _ = preprocess(pts, cfg["preprocess"], cfg["zones"])
        Rt = torch.from_numpy(R).cuda()
        world = (clean[:, :3] + torch.tensor([0, 0, LIDAR_Z], device="cuda")) @ Rt.T + torch.from_numpy(base).cuda()
        gt = label_points(world, t)
        pred, _ = seg.segment(clean)
        keep = gt > 0  # unlabeled points don't count
        conf += torch.bincount(gt[keep] * C + pred[keep], minlength=C * C).view(C, C)
        n += 1
    tp = conf.diag().float()
    iou = (tp / (conf.sum(0) + conf.sum(1) - tp).clamp(min=1).float()).cpu().numpy()
    present = (conf.sum(1) > 0).cpu().numpy()
    miou = float(iou[1:][present[1:]].mean())
    acc = float(tp.sum() / conf.sum())
    print(f"{n} Gazebo scans: point accuracy {100*acc:.1f}%, mIoU {100*miou:.1f}%")
    rows = []
    for i in range(1, C):
        if present[i]:
            top = conf[i].argsort(descending=True)[:2].tolist()
            share = conf[i, top].float() / conf[i].sum()
            confused = classes.NAMES[top[0]] if top[0] != i else classes.NAMES[top[1]]
            rows.append(f"| {classes.NAMES[i]} | {100*iou[i]:.1f}% | {int(conf[i].sum()):,} | {confused} |")
            print(rows[-1])
    if not a.no_write:
        md = ["", "## Segmentation accuracy on Gazebo scans", "",
              f"{n} scans from a recorded demo lap (`tools/eval_bag.py`), every {a.every}rd scan. Gazebo renders real car "
              "and walking-person meshes; ground truth comes from the world manifest (`tools/autolabel.py`). "
              f"Pretrained SemanticKITTI weights, zero-shot. Point accuracy **{100*acc:.1f}%**, mIoU **{100*miou:.1f}%**.", "",
              "| Class | IoU | GT points | Most confused with |", "|---|---|---|---|"] + rows + [""]
        res = ROOT / "docs" / "RESULTS.md"
        txt = res.read_text()
        cut = txt.find("\n## Segmentation accuracy on Gazebo scans")
        res.write_text((txt if cut < 0 else txt[:cut]) + "\n".join(md))


if __name__ == "__main__":
    main()
