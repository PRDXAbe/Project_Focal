"""Teaching figure: the same 6 m x 6 m patch of ground, zoomed in, in each distance zone.

Grabs one snapshot of the live map (/adaptmap/cells/*), finds the busiest 6 m patch in each zone and
draws every tile at its true size, so you can count the tiles: many tiny ones near the robot, a few
big ones far away. Also draws the overview with zones coloured.

Usage (sim + adaptmap_node running, ROS sourced):
  .venv/bin/python tools/resolution_figure.py [--out docs/figures/resolution_zoom.png]
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from adaptmap.config import load_config  # noqa: E402
from view25d import ZONE_RGB, Feed  # noqa: E402

PATCH = 6.0
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"


def tiles(ax, xyz, col, size, x0, y0, gap=0.12):
    m = (xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + PATCH) & (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + PATCH)
    h = size * (1 - gap) / 2
    c = xyz[m, :2]
    polys = np.stack([c + [-h, -h], c + [h, -h], c + [h, h], c + [-h, h]], 1)
    ax.add_collection(PolyCollection(polys, facecolors=col[m], edgecolors="none"))
    return int(m.sum())


def busiest_patch(xyz, center, lo, hi):
    """Lower-left corner of the 6 m patch (fully inside the zone ring) holding the most tiles."""
    d = np.abs(xyz[:, :2] - center[:2]).max(1)
    p = xyz[(d >= lo + PATCH) & (d < hi - PATCH) | ((lo == 0) & (d < hi - PATCH))]
    if not len(p):
        p = xyz
    k = np.floor(p[:, :2] / PATCH).astype(int)
    keys, cnt = np.unique(k, axis=0, return_counts=True)
    return keys[cnt.argmax()] * PATCH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures" / "resolution_zoom.png"))
    a = ap.parse_args()
    zones = load_config()["zones"]
    rclpy.init()
    feed = Feed(zones)
    threading.Thread(target=rclpy.spin, args=(feed,), daemon=True).start()
    for _ in range(300):
        cells, _, pose, _, _ = feed.snapshot()
        if pose is not None and len(cells) == len(zones):
            break
        time.sleep(0.1)
    else:
        sys.exit("no map received - is the sim + adaptmap_node running?")
    center = pose[0]

    fig = plt.figure(figsize=(17, 4.9), facecolor=SURF)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.35, 1, 1, 1, 1], wspace=0.18)
    ov = fig.add_subplot(gs[0])
    ov.set_facecolor("#16161a")
    lo = 0.0
    for i, z in enumerate(zones):
        xyz, _, size, _ = cells[z["name"]]
        ov.scatter(xyz[:, 0], xyz[:, 1], s=0.15, color=ZONE_RGB[i], linewidths=0, rasterized=True)
        h = z["half_extent"]
        ov.add_patch(Rectangle((center[0] - h, center[1] - h), 2 * h, 2 * h, fill=False, edgecolor=ZONE_RGB[i], linewidth=1.4))
    ov.plot(center[0], center[1], marker="^", color="white", markersize=8)
    ov.set_aspect("equal")
    ov.set_xlim(center[0] - 100, center[0] + 100); ov.set_ylim(center[1] - 100, center[1] + 100)
    ov.set_title("Whole map, coloured by tile size", loc="left", color=INK, fontsize=11)
    ov.tick_params(colors=INK2, labelsize=8)

    for i, z in enumerate(zones):
        xyz, col, size, _ = cells[z["name"]]
        x0, y0 = busiest_patch(xyz, center, lo, z["half_extent"])
        ax = fig.add_subplot(gs[i + 1])
        ax.set_facecolor("#16161a")
        n = tiles(ax, xyz, col, size, x0, y0)
        ov.add_patch(Rectangle((x0, y0), PATCH, PATCH, fill=False, edgecolor="white", linewidth=1.2))
        ov.text(x0 + PATCH + 1, y0 + PATCH, str(i + 1), color="white", fontsize=10, fontweight="bold")
        ax.set_xlim(x0, x0 + PATCH); ax.set_ylim(y0, y0 + PATCH)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(ZONE_RGB[i]); s.set_linewidth(3)
        dist = np.abs(np.array([x0, y0]) + PATCH / 2 - center[:2]).max()
        full = int(round(PATCH / size)) ** 2
        ax.set_title(f"{i + 1}.  ~{dist:.0f} m away:  {int(round(size * 100))} cm tiles", loc="left", color=INK, fontsize=11)
        ax.text(0, -0.05, f"{full:,} tiles cover this patch\n(5 cm everywhere: 14,400)",
                transform=ax.transAxes, va="top", color=INK2, fontsize=10)
        lo = z["half_extent"]
    fig.suptitle("The same 6 m x 6 m patch of ground, zoomed in, at four distances  -  colours = object type",
                 x=0.01, y=1.0, ha="left", color=INK, fontsize=13)
    fig.savefig(a.out, dpi=130, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {a.out}")
    feed.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
