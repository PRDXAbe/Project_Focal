"""Render the live AdaptMap output to a demo video, independent of the desktop session.

Subscribes to /adaptmap/cells/<zone> (published by adaptmap_node), /adaptmap/stats and /odom, and
draws each map update as a frame: left = top-down semantic map (each zone drawn at its own cell
size), right = elevation. Frames are encoded with ffmpeg on exit (Ctrl-C) or after --seconds.

Usage (ROS sourced): .venv/bin/python tools/render_live.py --out data/live/demo.mp4 --seconds 90
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from PIL import Image, ImageDraw, ImageFont
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adaptmap import classes  # noqa: E402
from adaptmap.config import load_config  # noqa: E402

HALF, PX = 60.0, 0.1          # rendered window +-60 m at 10 cm / pixel
N = int(2 * HALF / PX)
BG = np.array([22, 22, 26], np.uint8)


def cloud_xyzc(msg):
    a = np.frombuffer(msg.data, np.float32).reshape(-1, 4)
    rgb = a[:, 3].copy().view(np.uint32)
    return a[:, :3], np.stack([(rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255], 1).astype(np.uint8)


def elev_color(z):
    """Single-hue sequential ramp (light -> dark blue) for elevation 0..3 m above ground."""
    t = np.clip(z / 3.0, 0, 1)[:, None]
    lo, hi = np.array([214, 230, 250.]), np.array([12, 52, 122.])
    return (lo + (hi - lo) * t).astype(np.uint8)


class Renderer(Node):
    def __init__(self, frames_dir, zones):
        super().__init__("adaptmap_render")
        self.dir, self.zones = Path(frames_dir), zones
        self.cells, self.stats, self.pose, self.n = {}, None, (0.0, 0.0, 0.0), 0
        for z in zones:
            self.create_subscription(PointCloud2, f"/adaptmap/cells/{z['name']}", lambda m, k=z["name"]: self.on_cells(k, m), 1)
        self.create_subscription(String, "/adaptmap/stats", lambda m: setattr(self, "stats", json.loads(m.data)), 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 1)
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 22)
            self.small = ImageFont.truetype("DejaVuSans.ttf", 17)
        except OSError:
            self.font = self.small = ImageFont.load_default()

    def on_odom(self, m):
        q = m.pose.pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw)

    def on_cells(self, zone, msg):
        self.cells[zone] = cloud_xyzc(msg)
        if zone == self.zones[-1]["name"] and len(self.cells) == len(self.zones):
            self.render()

    def raster(self, colour_fn):
        img = np.tile(BG, (N, N, 1))
        cx, cy, _ = self.pose
        for z in reversed(self.zones):  # coarse first so finer zones draw on top
            xyz, rgb = self.cells[z["name"]]
            if len(xyz) == 0:
                continue
            col = colour_fn(xyz, rgb)
            k = int(np.ceil(z["cell"] / PX - 1e-6))  # cover the whole cell even when cell/PX is fractional
            i = np.floor((xyz[:, 0] - z["cell"] / 2 - (cx - HALF)) / PX).astype(int)
            j = np.floor((xyz[:, 1] - z["cell"] / 2 - (cy - HALF)) / PX).astype(int)
            for di in range(k):
                for dj in range(k):
                    ii, jj = i + di, j + dj
                    ok = (ii >= 0) & (ii < N) & (jj >= 0) & (jj < N)
                    img[N - 1 - jj[ok], ii[ok]] = col[ok]
        return img

    def render(self):
        sem = self.raster(lambda xyz, rgb: rgb)
        ground = np.percentile(np.concatenate([c[0][:, 2] for c in self.cells.values() if len(c[0])]), 5)
        ele = self.raster(lambda xyz, rgb: elev_color(xyz[:, 2] - ground))
        gap = np.tile(np.array([250, 250, 248], np.uint8), (N, 16, 1))
        canvas = Image.fromarray(np.concatenate([sem, gap, ele], 1))
        d = ImageDraw.Draw(canvas)
        cx, cy, yaw = self.pose
        for off in (0, N + 16):
            for z, dash in zip(self.zones, (1, 2, 3, 4)):
                h = z["half_extent"]
                if h > HALF:
                    continue
                a, b = (HALF - h) / PX, (HALF + h) / PX
                d.rectangle([off + a, a, off + b, b], outline=(255, 255, 255), width=2)
                d.text((off + a + 6, a + 4), f"{z['name']} {int(z['cell'] * 100)} cm", fill=(255, 255, 255), font=self.small)
            c = (off + N / 2, N / 2)
            tip = (c[0] + 18 * np.cos(yaw), c[1] - 18 * np.sin(yaw))
            d.line([c, tip], fill=(255, 255, 255), width=4)
            d.ellipse([c[0] - 6, c[1] - 6, c[0] + 6, c[1] + 6], fill=(255, 255, 255))
        d.text((12, N - 34), "semantic classes", fill=(255, 255, 255), font=self.font)
        d.text((N + 28, N - 34), "elevation: light = ground, dark = 3 m+", fill=(255, 255, 255), font=self.font)
        if self.stats:
            s, t = self.stats, self.stats["timing_ms"]
            occ = sum(s["occupied"].values())
            txt = (f"t={s['stamp']:.1f}s  pipeline {t['total']:.0f} ms  |  map cells {s['total_cells']:,} "
                   f"(uniform 5 cm: {s['uniform_cells']:,})  |  observed {occ:,}")
            d.rectangle([0, 0, 2 * N + 16, 34], fill=(0, 0, 0))
            d.text((12, 6), txt, fill=(255, 255, 255), font=self.font)
        # legend
        y = 44
        d.rectangle([N - 200, 36, N - 4, 44 + 22 * (classes.NUM_CLASSES - 1)], fill=(0, 0, 0))
        for i in range(1, classes.NUM_CLASSES):
            col = tuple(int(v) for v in classes.COLORS[i])
            d.rectangle([N - 190, y, N - 172, y + 16], fill=col)
            d.text((N - 164, y - 2), classes.NAMES[i], fill=(255, 255, 255), font=self.small)
            y += 22
        canvas.save(self.dir / f"{self.n:05d}.png")
        self.n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "live" / "demo.mp4"))
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--fps", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config()
    rclpy.init()
    frames = Path(tempfile.mkdtemp(prefix="adaptmap_frames_"))
    node = Renderer(frames, cfg["zones"])
    t0 = time.time()
    try:
        while rclpy.ok() and (not a.seconds or time.time() - t0 < a.seconds):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    print(f"{node.n} frames in {frames}")
    if node.n:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(a.fps), "-i", str(frames / "%05d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", a.out], check=True)
        last = sorted(frames.glob("*.png"))[-1]
        Image.open(last).save(Path(a.out).with_suffix(".png"))
        print(f"wrote {a.out}")
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
