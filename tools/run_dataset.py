"""Run the full AdaptMap pipeline on a real KITTI raw drive (HDL-64 + OXTS GPS/IMU poses).

Same code path as the ROS node: preprocess -> MinkUNet (fp16) -> foveated map (ray clearing) ->
terrain. Real reflectance is kept (the model was trained on it). Produces a side-by-side video
(raw LiDAR | 2.5D map), a timing summary and a final map snapshot.

Usage:
  .venv/bin/python tools/run_dataset.py --drive data/real/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync \\
      --calib data/real/kitti_raw/2011_09_26 --out data/real/kitti_0005 [--color class|zone|cost]
"""
import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from adaptmap import classes, terrain  # noqa: E402
from adaptmap.config import load_config  # noqa: E402
from adaptmap.foveated_map import FoveatedMap  # noqa: E402
from adaptmap.preprocess import preprocess  # noqa: E402
from adaptmap.segmentation import build_segmenter  # noqa: E402

EARTH_R = 6378137.0


def oxts_poses(drive):
    """KITTI devkit convention: Mercator-projected GPS + roll/pitch/yaw -> 4x4 IMU poses, first = origin."""
    files = sorted((drive / "oxts" / "data").glob("*.txt"))
    rows = [np.array(f.read_text().split(), float) for f in files]
    scale = math.cos(rows[0][0] * math.pi / 180.0)
    poses = []
    for r in rows:
        lat, lon, alt, roll, pitch, yaw = r[:6]
        tx = scale * lon * math.pi * EARTH_R / 180.0
        ty = scale * EARTH_R * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
        Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rz @ Ry @ Rx, [tx, ty, alt]
        poses.append(T)
    T0 = np.linalg.inv(poses[0])
    return [T0 @ T for T in poses]


def calib_imu_to_velo(calib_dir):
    vals = {}
    for line in (Path(calib_dir) / "calib_imu_to_velo.txt").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                vals[k] = np.array(v.split(), float)
            except ValueError:
                pass
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = vals["R"].reshape(3, 3), vals["T"]
    return T


def timestamps(drive):
    from datetime import datetime
    out = []
    for line in (drive / "velodyne_points" / "timestamps.txt").read_text().splitlines():
        d, t = line.strip().split(" ")
        hh, mm, ss = t.split(":")
        out.append(int(hh) * 3600 + int(mm) * 60 + float(ss))
    return np.array(out) - out[0]


def export_cells(fmap):
    """Map export -> the {zone: (xyz, rgb, cell, zone_idx)} format the 2.5D renderer draws."""
    cells = {}
    for zi, (name, v) in enumerate(fmap.export().items()):
        idx = v["known"].nonzero()
        r = v["resolution"]
        ox, oy = v["origin"]
        xyz = torch.stack([ox + (idx[:, 0].float() + 0.5) * r, oy + (idx[:, 1].float() + 0.5) * r,
                           v["elevation"][idx[:, 0], idx[:, 1]]], 1).cpu().numpy()
        col = classes.COLORS[v["class"][idx[:, 0], idx[:, 1]].cpu()].numpy() / 255.0
        col = np.where(v["filled"][idx[:, 0], idx[:, 1]].cpu().numpy()[:, None], col * 0.55, col)
        cells[name] = (xyz, col, r, zi)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--color", default="class", choices=["class", "zone", "cost"])
    ap.add_argument("--top", action="store_true")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    drive, out = Path(a.drive), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    cfg["preprocess"]["constant_intensity"] = None              # keep real reflectance
    cfg["preprocess"]["self_box"] = [-2.7, 1.3, -1.0, 1.0, -2.0, 0.3]  # the KITTI car around the roof LiDAR
    seg = build_segmenter(cfg)
    fmap = FoveatedMap(cfg["zones"], cfg["fusion"])
    poses, ts = oxts_poses(drive), timestamps(drive)
    T_velo_imu = calib_imu_to_velo(a.calib)
    T_imu_velo = np.linalg.inv(T_velo_imu)
    bins = sorted((drive / "velodyne_points" / "data").glob("*.bin"))
    if a.max:
        bins = bins[: a.max]

    from view25d import Recorder
    rec = Recorder(cfg["zones"], a.color, "top" if a.top else "chase")
    frames = Path(tempfile.mkdtemp(prefix="adaptmap_kitti_"))
    tim = []
    sync = torch.cuda.synchronize
    for i, b in enumerate(bins):
        raw = np.fromfile(b, np.float32).reshape(-1, 4)
        T = poses[i] @ T_imu_velo                       # LiDAR pose in the drive's local frame
        R = torch.tensor(T[:3, :3], dtype=torch.float32, device="cuda")
        t = torch.tensor(T[:3, 3], dtype=torch.float32, device="cuda")
        sync(); t0 = time.perf_counter()
        clean, ps = preprocess(torch.from_numpy(raw).cuda(), cfg["preprocess"], cfg["zones"])
        sync(); t1 = time.perf_counter()
        _, probs = seg.segment(clean)
        sync(); t2 = time.perf_counter()
        xyz = clean[:, :3] @ R.T + t
        fmap.integrate(xyz, probs, t[:2], float(ts[i]), origin=t)
        sync(); t3 = time.perf_counter()
        if i >= 2:
            tim.append({"pre": t1 - t0, "seg": t2 - t1, "fuse": t3 - t2, "pts": ps["out"], "raw": ps["raw"]})
        # render (not timed)
        tc = cfg["terrain"]
        cells = export_cells(fmap)
        if a.color == "cost":
            nv = fmap.nav_view(cell=tc["costmap_cell"], half=tc["costmap_half"])
            cost = terrain.analyze(nv, tc)["cost"].cpu().numpy()
            rec.cost = (cost, tuple(nv["origin"]), nv["resolution"])
        raw_world = (torch.from_numpy(raw[:, :3]).cuda() @ R.T + t).cpu().numpy()
        rec.frame(cells, raw_world, (T[:3, 3].astype(np.float32), T[:3, :3].astype(np.float32))).save(frames / f"{i:05d}.png")
        if i % 25 == 0:
            print(f"frame {i}/{len(bins)}: seg {1e3 * (t2 - t1):.0f} ms, fuse {1e3 * (t3 - t2):.1f} ms", flush=True)

    video = out / f"kitti_{a.color}{'_top' if a.top else ''}.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "10", "-i", str(frames / "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    str(video)], check=True)
    last = sorted(frames.glob("*.png"))[-1]
    from PIL import Image
    Image.open(last).save(video.with_suffix(".png"))
    m = {k: round(1e3 * float(np.mean([x[k] for x in tim])), 1) for k in ("pre", "seg", "fuse")}
    summary = {"drive": drive.name, "scans": len(bins), "latency_ms": m, "total_ms": round(sum(m.values()), 1),
               "hz": round(1000 / sum(m.values()), 1), "points_raw": int(np.mean([x["raw"] for x in tim])),
               "points_after_preprocess": int(np.mean([x["pts"] for x in tim])), "map": fmap.stats() | {"layers": None}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, default=str))
    print(f"wrote {video}")


if __name__ == "__main__":
    main()
