"""Motion-aware keep-out for moving classes (pedestrians) on the navigation grid.

A static keep-out disc marks where a person *is*; a person crossing the robot's path walks into the lane
at the moment the robot arrives, so the planner sees nothing in the way until contact. This module
tracks each moving-class blob between costmap updates and marks the corridor it will sweep over the
next `horizon` seconds as lethal, so the planner stops or replans before the person reaches the lane.

Per update (grid = terrain.analyze()["dynamic"], world coordinates, a few ms on CPU):
  1. connected components of the mask -> blob centroids (world frame);
  2. nearest-neighbour association to existing tracks inside a gate that grows with elapsed time;
  3. velocity = net centroid displacement over the last `window` seconds. A track counts as moving only
     if that covers at least `min_span` s, is faster than `min_speed`, and the path is nearly straight
     (net / travelled >= `min_straight`): a static object whose blob merges with a passer-by or is seen
     from a changing angle has a centroid that jumps back and forth, not one that walks;
  4. moving tracks stamp a capsule from the centroid to lead + v * horizon, `radius` wide, where `lead`
     is the blob's front edge along v. Moving-class cells fade from the map over ~tau_dynamic, so a
     walking person's blob drags a trail and its centroid lags the person by up to ~1 m; the front edge
     does not.
Tracks not seen for `max_age` seconds are dropped.
"""
import math

import numpy as np
import torch
from scipy import ndimage

from .terrain import _disc_dilate


class MotionPredictor:
    def __init__(self, cfg):
        self.horizon = cfg.get("horizon", 2.0)        # s of predicted walking marked lethal
        self.radius = cfg.get("radius", 0.6)          # corridor half-width (m)
        self.min_speed = cfg.get("min_speed", 0.3)    # m/s; slower blobs are treated as standing
        self.max_speed = cfg.get("max_speed", 3.0)    # m/s; faster centroid jumps are mis-associations
        self.gate = cfg.get("gate", 1.0)              # m, association distance (+ max_speed * dt)
        self.window = cfg.get("window", 1.0)          # s of centroid history the velocity is taken over
        self.min_span = cfg.get("min_span", 0.6)      # s of history needed before a track can be moving
        self.min_straight = cfg.get("min_straight", 0.7)
        self.max_age = cfg.get("max_age", 1.0)        # s
        self.min_cells = cfg.get("min_cells", 2)
        self.tracks = []                              # dicts: pos, vel (2,), stamp, hist [(stamp, pos)], moving

    def _blobs(self, mask, origin, res):
        """Connected blobs of the mask: centroids (k, 2) and each blob's cell centres, world frame."""
        lab, n = ndimage.label(mask.cpu().numpy())
        ij = np.argwhere(lab > 0)
        world = np.asarray(origin) + (ij + 0.5) * res      # grid index (x, y) -> world
        k = lab[ij[:, 0], ij[:, 1]]
        cells = [world[k == b] for b in range(1, n + 1)]
        cells = [c for c in cells if len(c) >= self.min_cells]
        return np.array([c.mean(0) for c in cells]).reshape(-1, 2), cells

    def update(self, mask, origin, res, stamp):
        """mask: (n, n) bool, [x, y] indexed, cell (0, 0) at `origin`. Returns the predicted lethal mask."""
        dets, cells = self._blobs(mask, origin, res)
        self.tracks = [t for t in self.tracks if stamp - t["stamp"] <= self.max_age]
        for t in self.tracks:
            t["cells"] = None
        used = set()
        for t in sorted(self.tracks, key=lambda t: -len(t["hist"])):   # established tracks pick first
            dt = stamp - t["stamp"]
            if dt <= 0 or len(dets) == 0:
                continue
            d = np.linalg.norm(dets - (t["pos"] + t["vel"] * dt), axis=1)
            d[list(used)] = np.inf
            j = int(np.argmin(d))
            if d[j] > self.gate + self.max_speed * dt:
                continue
            used.add(j)
            t["pos"], t["stamp"], t["cells"] = dets[j], stamp, cells[j]
            t["hist"] = [h for h in t["hist"] if stamp - h[0] <= self.window] + [(stamp, dets[j])]
            self._estimate(t)
        for j in range(len(dets)):
            if j not in used:
                self.tracks.append({"pos": dets[j], "vel": np.zeros(2), "stamp": stamp, "cells": cells[j],
                                    "hist": [(stamp, dets[j])], "moving": False})
        return self._corridors(mask, origin, res)

    def _estimate(self, t):
        (t0, p0), (t1, p1) = t["hist"][0], t["hist"][-1]
        span = t1 - t0
        if span <= 0:
            t["vel"], t["moving"] = np.zeros(2), False
            return
        net = p1 - p0
        steps = sum(np.linalg.norm(b[1] - a[1]) for a, b in zip(t["hist"], t["hist"][1:]))
        t["vel"] = net / span
        speed = np.linalg.norm(t["vel"])
        t["moving"] = bool(span >= self.min_span and self.min_speed <= speed <= self.max_speed
                           and np.linalg.norm(net) >= self.min_straight * steps)

    def moving(self):
        return [t for t in self.tracks if t["moving"]]

    def _corridors(self, mask, origin, res):
        n = mask.shape[0]
        seed = torch.zeros_like(mask, dtype=torch.bool)
        any_ = False
        for t in self.moving():
            ahead = t["vel"] / np.linalg.norm(t["vel"])
            lead = 0.0 if t["cells"] is None else max(0.0, float(((t["cells"] - t["pos"]) @ ahead).max()))
            end = t["pos"] + ahead * lead + t["vel"] * self.horizon
            k = max(2, int(math.ceil(np.linalg.norm(end - t["pos"]) / (res / 2))) + 1)
            pts = (np.linspace(t["pos"], end, k) - np.asarray(origin)) / res
            ij = np.floor(pts).astype(np.int64)
            ok = (ij >= 0).all(1) & (ij < n).all(1)
            if ok.any():
                ij = torch.from_numpy(ij[ok]).to(mask.device)
                seed[ij[:, 0], ij[:, 1]] = True
                any_ = True
        return _disc_dilate(seed, self.radius / res) if any_ else seed
