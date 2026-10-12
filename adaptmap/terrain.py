"""Terrain analysis on the navigation grid: slope, step, roughness, curbs and a traversability cost.

Input is `FoveatedMap.nav_view(...)` (one uniform grid resampled from the foveated layers, with
`ground`, `elevation`, `hag`, `clearance`, `class`, `filled`). Everything is computed on the GPU.

Cost convention (nav_msgs/OccupancyGrid, as read by Nav2 with trinary_costmap: false):
  0 = free, 1..99 = increasingly undesirable, 100 = lethal, -1 = unknown.
"""
import math

import torch
import torch.nn.functional as F

from . import classes
from .foveated_map import _box

LETHAL, UNKNOWN = 100, -1


def _masked_mean(x, valid, k):
    s = _box(torch.where(valid, x, 0.0), k)
    n = _box(valid.float(), k)
    return torch.where(n > 0, s / n.clamp(min=1), torch.full_like(x, float("nan"))), n


def _nan_pool(x, k, sign):
    """k-radius max (sign=+1) or min (sign=-1) over finite cells."""
    fill = float("-inf") if sign > 0 else float("inf")
    y = torch.where(torch.isfinite(x), x, torch.full_like(x, fill)) * sign
    p = F.max_pool2d(y[None, None], 2 * k + 1, 1, k)[0, 0] * sign
    return torch.where(torch.isfinite(p), p, torch.full_like(p, float("nan")))


def analyze(nv, cfg):
    """Return dict of (n, n) tensors: slope_deg, step, roughness, curb (bool), passable_under (bool),
    cost (int8, OccupancyGrid convention). cfg = config['terrain']."""
    res = nv["resolution"]
    elev, ground, hag, clear = nv["elevation"], nv["ground"], nv["hag"], nv["clearance"]
    cls, filled = nv["class"], nv["filled"]
    known = torch.isfinite(elev)
    names = classes.NAMES
    ground_cls = classes.GROUND.to(cls.device)[cls]

    # slope of the (smoothed) ground surface, degrees
    kg = max(1, int(round(cfg.get("slope_window", 0.6) / 2 / res)))
    gsm, _ = _masked_mean(ground, torch.isfinite(ground), kg)
    gx = (torch.roll(gsm, -1, 0) - torch.roll(gsm, 1, 0)) / (2 * res)
    gy = (torch.roll(gsm, -1, 1) - torch.roll(gsm, 1, 1)) / (2 * res)
    slope = torch.rad2deg(torch.atan(torch.sqrt(gx ** 2 + gy ** 2)))

    # drivable surface: the ground under overhangs the robot fits beneath, the top surface elsewhere
    passable_under = torch.isfinite(clear) & (clear > cfg.get("robot_height", 1.0))
    surf = torch.where(passable_under, ground, elev)

    # step: biggest drop/rise of the drivable surface within one cell's neighbourhood
    ks = max(1, int(round(cfg.get("step_window", 0.2) / res)))
    step = torch.maximum(_nan_pool(surf, ks, 1) - surf, surf - _nan_pool(surf, ks, -1))

    # roughness: std of the drivable surface over a small window
    kr = max(1, int(round(cfg.get("rough_window", 0.4) / 2 / res)))
    m1, n = _masked_mean(surf, known, kr)
    m2, _ = _masked_mean(surf * surf, known, kr)
    rough = torch.sqrt(torch.clamp(m2 - m1 ** 2, min=0.0))

    # curb: road and sidewalk meet within the step window and the height jump is curb-sized
    def near(name):
        return F.max_pool2d((cls == names.index(name)).float()[None, None], 2 * ks + 1, 1, ks)[0, 0] > 0
    lo, hi = cfg.get("curb_min", 0.05), cfg.get("curb_max", 0.30)
    curb = known & near("road") & near("sidewalk_curb") & (step >= lo) & (step <= hi)

    # obstacles: anything standing higher than the robot can step, unless it can be driven under
    tall = known & (hag > cfg.get("max_step", 0.12)) & ~passable_under
    obstacle = tall | (known & ~ground_cls & (cls != 0) & ~passable_under & (hag > 0.05))

    # cost: class prior + slope + roughness, then lethal rules
    prior = torch.zeros(classes.NUM_CLASSES, device=cls.device)
    for name, c in cfg.get("class_cost", {}).items():
        prior[names.index(name)] = c
    cost = prior[cls]
    max_slope = cfg.get("max_slope_deg", 20.0)
    cost = cost + torch.nan_to_num(slope / max_slope, nan=0.0).clamp(max=1.0) * cfg.get("slope_cost", 40.0)
    cost = cost + torch.nan_to_num(rough / cfg.get("rough_max", 0.1), nan=0.0).clamp(max=1.0) * cfg.get("rough_cost", 20.0)
    cost = cost + filled.float() * cfg.get("filled_cost", 10.0)
    cost = cost.clamp(0, 99)
    lethal = obstacle | (torch.nan_to_num(slope, nan=0.0) > max_slope) | \
        (torch.nan_to_num(step, nan=0.0) > cfg.get("max_step", 0.12))
    if cfg.get("curb_lethal", True):
        lethal |= curb
    cost = torch.where(lethal, torch.full_like(cost, LETHAL), cost)
    cost = torch.where(known, cost.round(), torch.full_like(cost, UNKNOWN)).to(torch.int8)
    return {"slope_deg": slope, "step": step, "roughness": rough, "curb": curb,
            "passable_under": passable_under, "obstacle": obstacle, "cost": cost}
