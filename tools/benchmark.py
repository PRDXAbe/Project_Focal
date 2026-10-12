"""Offline end-to-end benchmark: simulated drive -> preprocess -> MinkUNet -> foveated map,
with a uniform 5 cm map fed the same data as the baseline.

Scans come from tools/synth_scan.py (analytic 64-beam ray-cast of the Gazebo world manifest,
pedestrians moving), so exact per-point ground truth is available.

Usage: python tools/benchmark.py [--frames 300] [--speed 2.0] [--no-uniform] [--stats-log run.jsonl]
  --stats-log: also summarise a live ROS run (adaptmap_node stats_log JSONL) instead of simulating.
Writes docs/RESULTS.md and docs/figures/*.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from adaptmap import classes  # noqa: E402
from adaptmap.config import load_config  # noqa: E402
from adaptmap.foveated_map import FoveatedMap  # noqa: E402
from adaptmap.preprocess import preprocess  # noqa: E402
from adaptmap.segmentation import build_segmenter  # noqa: E402
from synth_scan import load_boxes, scan  # noqa: E402

FIG = ROOT / "docs" / "figures"
SENSOR_H = 1.75
# reference palette (light): categorical slots, text, surface
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def sync():
    torch.cuda.synchronize()


def run(args, cfg):
    seg = build_segmenter(cfg)
    fov = FoveatedMap(cfg["zones"], cfg["fusion"])
    uni = None if args.no_uniform else FoveatedMap(
        [{"name": "uniform", "half_extent": cfg["zones"][-1]["half_extent"], "cell": cfg["zones"][0]["cell"]}],
        cfg["fusion"], snap=max(z["cell"] for z in cfg["zones"]))
    from adaptmap.adaptive import ZonePolicy
    ac = cfg.get("adaptive_zones", {})
    policy = ZonePolicy(cfg["zones"], ac, fov.snap) if ac.get("enabled") else None
    conf = torch.zeros(classes.NUM_CLASSES, classes.NUM_CLASSES, dtype=torch.long, device="cuda")
    rec = []
    torch.cuda.reset_peak_memory_stats()
    for f in range(args.frames + 3):
        t = f / 10.0
        x, y = args.x0 + args.speed * t, -1.0
        raw, gt = scan([x, y, SENSOR_H], load_boxes(t))
        sync(); t0 = time.perf_counter()
        clean, ps = preprocess(raw, cfg["preprocess"], cfg["zones"])
        sync(); t1 = time.perf_counter()
        lab, prob = seg.segment(clean)
        sync(); t2 = time.perf_counter()
        xyz = clean[:, :3] + torch.tensor([x, y, SENSOR_H], device="cuda")
        if policy is not None and (z := policy.update(args.speed, t)) is not None:
            fov.set_zones(z)
        fov.integrate(xyz, prob, [x, y], t, origin=[x, y, SENSOR_H])
        sync(); t3 = time.perf_counter()
        tu = 0.0
        if uni is not None:
            uni.integrate(xyz, prob, [x, y], t, origin=[x, y, SENSOR_H])
            sync(); tu = time.perf_counter() - t3
        if f < 3:  # warm-up (spconv kernel tuning)
            continue
        # GT for kept points: re-label by nearest raw index (preprocess keeps order; match via mask)
        gtc = label_clean(raw, gt, clean)
        conf += torch.bincount(gtc * classes.NUM_CLASSES + lab, minlength=classes.NUM_CLASSES ** 2).view(
            classes.NUM_CLASSES, classes.NUM_CLASSES)
        rec.append({"pre": t1 - t0, "seg": t2 - t1, "fuse": t3 - t2, "uni_fuse": tu, "pts": ps["out"], "raw": ps["raw"],
                    "near": fov.zones[0]["half_extent"], "cells": fov.stats()["total_cells"]})
        if f % 50 == 0:
            print(f"frame {f}: seg {1e3*(t2-t1):.0f} ms fuse {1e3*(t3-t2):.1f} ms uniform {1e3*tu:.1f} ms")
    return fov, uni, conf, rec


def label_clean(raw, gt, clean):
    """Ground truth for preprocessed points (match by exact coordinates; preprocess only filters)."""
    key = lambda p: (p[:, 0] * 1e4).round().long() * 4_000_003 + (p[:, 1] * 1e4).round().long() * 7919 + (p[:, 2] * 1e4).round().long()  # noqa: E731
    kr, kc = key(raw[:, :3]), key(clean[:, :3])
    order = torch.argsort(kr)
    pos = torch.searchsorted(kr[order], kc).clamp(max=len(kr) - 1)
    return gt[order[pos]]


def iou(conf):
    tp = conf.diag().float()
    denom = conf.sum(0).float() + conf.sum(1).float() - tp
    return torch.where(conf.sum(1) > 0, tp / denom.clamp(min=1), torch.full_like(tp, float("nan")))


# ---------------------------------------------------------------- figures
def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_cells(stats, mem_f, mem_u, plt):
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6), facecolor=SURF)
    rows = [("Cells", stats["total_cells"], stats["uniform_cells"], "{:,.0f}"),
            ("Map memory (MB)", mem_f, mem_u, "{:,.0f}")]
    for ax, (title, fv, uv, fmt) in zip(axes, rows):
        style(ax)
        ax.barh([1, 0], [uv, fv], color=[ORANGE, BLUE], height=0.55, edgecolor=SURF, linewidth=2)
        ax.set_yticks([1, 0], ["Uniform 5 cm", "AdaptMap foveated"], color=INK)
        for yy, v in ((1, uv), (0, fv)):
            ax.text(v, yy, "  " + fmt.format(v), va="center", color=INK, fontsize=9)
        ax.set_xlim(0, uv * 1.3)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M" if v >= 1e6 else f"{v:,.0f}"))
    fig.suptitle(f"Same 200 x 200 m window: {stats['reduction']:.1f}x fewer cells", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "cells_memory.png", dpi=160)
    plt.close(fig)


def fig_latency(lat, plt):
    stages = [("Preprocess", lat["pre"], BLUE), ("Segmentation (MinkUNet)", lat["seg"], ORANGE),
              ("Foveated fusion", lat["fuse"], AQUA)]
    fig, ax = plt.subplots(figsize=(9, 1.9), facecolor=SURF)
    style(ax)
    left = 0
    for name, v, c in stages:
        ax.barh(0, v, left=left, color=c, height=0.5, edgecolor=SURF, linewidth=2)
        ax.text(left + v / 2, 0.42, f"{name}\n{v:.1f} ms", ha="center", va="bottom", color=INK, fontsize=8.5)
        left += v
    ax.axvline(100, color=INK2, linewidth=1, linestyle="--")
    ax.text(100, -0.42, " 10 Hz budget", color=INK2, fontsize=8.5, va="top")
    ax.set_yticks([])
    ax.set_xlim(0, max(110, left * 1.1))
    ax.set_ylim(-0.6, 1.1)
    ax.set_xlabel("mean latency per scan (ms)", color=INK2, fontsize=9)
    ax.set_title(f"End-to-end {left:.0f} ms per scan ({1000/left:.1f} Hz) on RTX 4050 laptop", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "latency.png", dpi=160)
    plt.close(fig)


def fig_iou(ious, plt):
    names = [classes.NAMES[i] for i in range(1, classes.NUM_CLASSES) if not np.isnan(ious[i])]
    vals = [ious[i] for i in range(1, classes.NUM_CLASSES) if not np.isnan(ious[i])]
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=SURF)
    style(ax)
    yy = np.arange(len(names))[::-1]
    ax.barh(yy, vals, color=BLUE, height=0.6, edgecolor=SURF, linewidth=2)
    ax.set_yticks(yy, names, color=INK)
    for y_, v in zip(yy, vals):
        ax.text(v, y_, f"  {100*v:.0f}%", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_title("Per-class IoU in simulation (pretrained SemanticKITTI weights, no fine-tuning)", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "class_iou.png", dpi=160)
    plt.close(fig)


def fig_map(fov, plt):
    """Top-down view: class colours shaded by elevation, zone squares overlaid."""
    from matplotlib.patches import Patch, Rectangle
    half = 60.0
    nv = fov.nav_view(cell=0.1, half=half)
    cls, el = nv["class"].cpu().numpy(), nv["elevation"].cpu().numpy()
    rgb = classes.COLORS.numpy()[cls].astype(np.float32) / 255
    known = np.isfinite(el)
    shade = np.clip(0.55 + 0.45 * np.nan_to_num(el - el[known].min() if known.any() else el) / 6, 0.55, 1.0)
    img = np.where(known[..., None], rgb * shade[..., None], 0.12)
    ox, oy = nv["origin"]
    fig, ax = plt.subplots(figsize=(8.4, 8.4), facecolor=SURF)
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=[ox, ox + 2 * half, oy, oy + 2 * half], interpolation="nearest")
    cx, cy = fov.center_xy.tolist()
    for z, ls in zip(fov.layers, ("-", "--", "-.", ":")):
        h = z.half
        ax.add_patch(Rectangle((cx - h, cy - h), 2 * h, 2 * h, fill=False, edgecolor="white", linewidth=1.2, linestyle=ls))
        if h <= half:
            ax.text(cx - h + 0.5, cy + h - 2.2, f"{z.name}: {int(z.cell*100)} cm", color="white", fontsize=9)
    ax.plot([cx], [cy], marker="^", color="white", markersize=9)
    present = sorted(set(np.unique(cls[known]).tolist()) - {0})
    ax.legend(handles=[Patch(color=classes.COLORS[i].numpy() / 255, label=classes.NAMES[i]) for i in present],
              loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_xlim(ox, ox + 2 * half); ax.set_ylim(oy, oy + 2 * half)
    ax.set_xlabel("x (m)", color=INK2); ax.set_ylabel("y (m)", color=INK2)
    ax.set_title("AdaptMap 2.5D semantic elevation map (top-down, brighter = higher)", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "map_topdown.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--x0", type=float, default=0.0)
    ap.add_argument("--no-uniform", action="store_true")
    ap.add_argument("--adaptive", action="store_true", help="distance-zoned point thinning before the network")
    ap.add_argument("--no-write", action="store_true", help="print a summary only; leave docs/ untouched")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    cfg["preprocess"]["adaptive_downsample"] = a.adaptive
    FIG.mkdir(parents=True, exist_ok=True)
    fov, uni, conf, rec = run(a, cfg)
    lat = {k: 1e3 * float(np.mean([r[k] for r in rec])) for k in ("pre", "seg", "fuse", "uni_fuse")}
    p95 = {k: 1e3 * float(np.percentile([r[k] for r in rec], 95)) for k in ("pre", "seg", "fuse", "uni_fuse")}
    s = fov.stats()
    mem_f = s["map_bytes"] / 2**20
    mem_u = uni.stats()["map_bytes"] / 2**20 if uni else s["uniform_cells"] * 44 / 2**20
    occ_f = sum(v["occupied"] for v in s["layers"].values())
    occ_u = uni.stats()["layers"]["uniform"]["occupied"] if uni else float("nan")
    ious = iou(conf).cpu().numpy()
    miou = float(np.nanmean(ious[1:]))
    acc = float(conf.diag().sum() / conf.sum())
    peak = torch.cuda.max_memory_allocated() / 2**20

    if a.no_write:
        print(json.dumps({"speed": a.speed, "near_radius_m": rec[-1]["near"], "cells": rec[-1]["cells"],
                          "latency_ms": {k: round(v, 1) for k, v in lat.items()}, "pts_in": round(float(np.mean([r['pts'] for r in rec]))),
                          "miou": round(100 * miou, 1), "acc": round(100 * acc, 1),
                          "iou": {classes.NAMES[i]: round(100 * float(ious[i]), 1) for i in range(1, classes.NUM_CLASSES) if not np.isnan(ious[i])}}))
        return
    fig_cells(s, mem_f, mem_u, plt)
    fig_latency(lat, plt)
    fig_iou(ious, plt)
    fig_map(fov, plt)

    total = lat["pre"] + lat["seg"] + lat["fuse"]
    lines = [
        "# AdaptMap — Results", "",
        f"Offline benchmark (`python tools/benchmark.py --frames {a.frames}`): simulated 10 Hz drive at {a.speed} m/s "
        "through the urban test world, 64-beam KITTI-like LiDAR at 1.75 m, moving pedestrians. "
        "Hardware: RTX 4050 Laptop (6 GB), PyTorch 2.1 / CUDA 12.1.", "",
        "## Map size: foveated vs uniform", "",
        "| | Uniform 5 cm | AdaptMap foveated | Reduction |", "|---|---|---|---|",
        f"| Cells (200 x 200 m window) | {s['uniform_cells']:,} | {s['total_cells']:,} | {s['reduction']:.1f}x |",
        f"| Map memory | {mem_u:,.0f} MB | {mem_f:,.0f} MB | {mem_u/mem_f:.1f}x |",
        f"| Occupied cells at end of run | {occ_u:,} | {occ_f:,} | {occ_u/max(occ_f,1):.1f}x |" if uni else "",
        f"| Fusion time per scan (mean / p95) | {lat['uni_fuse']:.1f} / {p95['uni_fuse']:.1f} ms | {lat['fuse']:.1f} / {p95['fuse']:.1f} ms | {lat['uni_fuse']/max(lat['fuse'],1e-6):.1f}x |" if uni else "",
        "", "Per-zone layout:", "", "| Zone | Extent (Chebyshev) | Cell | Cells |", "|---|---|---|---|",
    ]
    inner = 0
    for z in fov.layers:
        lines.append(f"| {z.name} | {inner:.0f}–{z.half:.0f} m | {z.cell*100:.0f} cm | {z.ring_cells:,} |")
        inner = z.half
    lines += [
        "", "![cells](figures/cells_memory.png)", "",
        "## Latency", "",
        "| Stage | Mean (ms) | p95 (ms) |", "|---|---|---|",
        f"| Preprocess | {lat['pre']:.1f} | {p95['pre']:.1f} |",
        f"| Segmentation (MinkUNet-34, spconv) | {lat['seg']:.1f} | {p95['seg']:.1f} |",
        f"| Foveated fusion | {lat['fuse']:.1f} | {p95['fuse']:.1f} |",
        f"| **End-to-end** | **{total:.1f}** ({1000/total:.1f} Hz) | |", "",
        f"Mean points per scan: {np.mean([r['raw'] for r in rec]):,.0f} raw → {np.mean([r['pts'] for r in rec]):,.0f} after preprocessing. "
        f"Peak GPU memory (model + both maps): {peak:,.0f} MB.", "",
        "![latency](figures/latency.png)", "",
        "## Segmentation accuracy (simulation)", "",
        f"Pretrained SemanticKITTI MinkUNet-34 (69.3 mIoU on real SemanticKITTI val, per mmdet3d model zoo), "
        f"applied zero-shot to simulated scans. Point accuracy **{100*acc:.1f}%**, mIoU **{100*miou:.1f}%** over classes present.", "",
        "| Class | IoU |", "|---|---|",
    ] + [f"| {classes.NAMES[i]} | {100*ious[i]:.1f}% |" for i in range(1, classes.NUM_CLASSES) if not np.isnan(ious[i])] + [
        "",
        "Known sim-to-real gap: the analytic ray-caster models cars/pedestrians/crates as plain boxes, which the real-data "
        "model reads as walls; the Gazebo world uses real car and walking-person meshes. Closing the gap is what "
        "`tools/autolabel.py` + `tools/finetune.py` are for (phase P8).", "",
        "![iou](figures/class_iou.png)", "",
        "## Map snapshot", "", "![map](figures/map_topdown.png)", "",
    ]
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(l for l in lines if l is not None))
    json.dump({"latency_ms": lat, "p95_ms": p95, "stats": s, "mem_mb": [mem_f, mem_u], "miou": miou, "acc": acc,
               "iou": {classes.NAMES[i]: float(ious[i]) for i in range(classes.NUM_CLASSES)}},
              open(ROOT / "docs" / "results.json", "w"), indent=1, default=str)
    print(f"done: e2e {total:.1f} ms, mIoU {100*miou:.1f}, cells {s['total_cells']:,} vs {s['uniform_cells']:,}")


if __name__ == "__main__":
    main()
