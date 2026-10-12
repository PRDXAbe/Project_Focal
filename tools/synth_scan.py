"""Analytic KITTI-like LiDAR ray-caster against the generated world manifest (boxes + ground).

Gives realistic scan structure (rings, occlusion, range falloff) with exact per-point labels,
for offline tests/benchmarks without Gazebo. Cylinders/spheres/actors are approximated by boxes.

Usage: python tools/synth_scan.py --x 10 --y -1.5 --out data/synth_000.npz
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "ros2_ws" / "src" / "adaptmap_sim" / "worlds" / "urban_manifest.json"


_MANIFESTS = {}


def manifest_for(world):
    """Manifest path for a world name ('urban', 'urban_s11') or an .sdf/.json path."""
    w = Path(world)
    if w.suffix == ".json":
        return w
    name = w.stem if w.suffix == ".sdf" else str(world)
    return MANIFEST.parent / f"{name}_manifest.json"


def load_boxes(t=0.0, device="cuda", manifest=None):
    from adaptmap import classes
    path = Path(manifest) if manifest else MANIFEST
    if path not in _MANIFESTS:
        _MANIFESTS[path] = json.loads(path.read_text())["objects"]
    objs = _MANIFESTS[path]
    c, h, yaw, lab = [], [], [], []
    for o in objs:
        if o["type"] == "actor":
            x, y, yw = actor_pose(o, t)
            p = [x, y, 0.9]
        else:
            x, y, z, yw = o["pose"]
            p = [x, y, z]
        c.append(p); h.append([s / 2 for s in o["size"]]); yaw.append(yw)
        lab.append(classes.NAMES.index(o["class"]))
    f = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
    return f(c), f(h), f(yaw), f(lab, torch.long)


def actor_pose(o, t):
    traj, period = o["trajectory"], o["period"]
    t = t % period
    for (t0, x0, y0, w0), (t1, x1, y1, _) in zip(traj, traj[1:]):
        if t0 <= t <= t1:
            a = (t - t0) / max(t1 - t0, 1e-6)
            return x0 + a * (x1 - x0), y0 + a * (y1 - y0), w0
    return traj[-1][1], traj[-1][2], traj[-1][3]


def scan(origin, boxes, beams=64, cols=2048, fov=(-24.8, 2.0), max_range=100.0, noise=0.01, device="cuda"):
    """Returns sensor-frame points (N,4) and labels (N,). Ground (z=0) is terrain unless inside a ground box."""
    from adaptmap import classes
    centers, halves, yaws, labels = boxes
    o = torch.tensor(origin, dtype=torch.float32, device=device)
    el = torch.deg2rad(torch.linspace(fov[0], fov[1], beams, device=device))
    az = torch.linspace(-math.pi, math.pi, cols + 1, device=device)[:-1]
    E, A = torch.meshgrid(el, az, indexing="ij")
    d = torch.stack([E.cos() * A.cos(), E.cos() * A.sin(), E.sin()], -1).reshape(-1, 3)
    R = d.shape[0]
    best_t = torch.full((R,), max_range, device=device)
    best_l = torch.full((R,), -1, dtype=torch.long, device=device)
    # ground plane z = 0
    tg = torch.where(d[:, 2] < -1e-6, -o[2] / d[:, 2], torch.full_like(d[:, 2], float("inf")))
    hit = tg < best_t
    best_t[hit] = tg[hit]
    best_l[hit] = classes.NAMES.index("terrain")
    # oriented boxes (yaw only): slab test in box frame, chunked over boxes
    for i in range(0, centers.shape[0], 16):
        c, h, yw, lb = centers[i:i + 16], halves[i:i + 16], yaws[i:i + 16], labels[i:i + 16]
        cy, sy = yw.cos(), yw.sin()
        oc = o[None] - c                                   # (B,3)
        ol = torch.stack([cy * oc[:, 0] + sy * oc[:, 1], -sy * oc[:, 0] + cy * oc[:, 1], oc[:, 2]], -1)
        dl = torch.stack([cy[:, None] * d[None, :, 0] + sy[:, None] * d[None, :, 1],
                          -sy[:, None] * d[None, :, 0] + cy[:, None] * d[None, :, 1],
                          d[None, :, 2].expand(len(c), -1)], -1)  # (B,R,3)
        inv = 1.0 / torch.where(dl.abs() < 1e-9, torch.full_like(dl, 1e-9), dl)
        t1 = (-h[:, None] - ol[:, None]) * inv
        t2 = (h[:, None] - ol[:, None]) * inv
        tn = torch.minimum(t1, t2).max(-1).values
        tf = torch.maximum(t1, t2).min(-1).values
        ok = (tf >= tn) & (tn > 0.05)
        tn = torch.where(ok, tn, torch.full_like(tn, float("inf")))
        tb, bi = tn.min(0)
        hit = tb < best_t
        best_t[hit] = tb[hit]
        best_l[hit] = lb[bi[hit]]
    m = best_l >= 0
    r = best_t[m] + torch.randn(int(m.sum()), device=device) * noise
    pts = d[m] * r[:, None]
    return torch.cat([pts, torch.full_like(r[:, None], 0.25)], 1), best_l[m]


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=10.0)
    ap.add_argument("--y", type=float, default=-1.5)
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--out", default=str(ROOT / "data" / "synth_000.npz"))
    a = ap.parse_args()
    pts, lab = scan([a.x, a.y, 1.75], load_boxes(a.t))
    np.savez(a.out, points=pts.cpu().numpy(), labels=lab.cpu().numpy(), origin=[a.x, a.y, 1.75])
    print(f"{pts.shape[0]} points -> {a.out}")


if __name__ == "__main__":
    main()
