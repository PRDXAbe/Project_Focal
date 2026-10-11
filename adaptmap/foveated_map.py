"""Foveated variable-resolution 2.5D semantic elevation map.

The map is a stack of nested square layers centred on the robot. Layer i covers
Chebyshev distance (half_{i-1}, half_i] from the map centre with its own cell
size, so detail is high near the robot and drops with distance. Each layer is a
dense GPU grid; cells inside the next-finer layer are masked out and never
written, so the effective cell count is only the ring.

Every cell stores a decayed running average of per-scan observations:
min/max/mean height, E[z^2] (-> variance), and the class-probability vector.
Weights decay with exp(-dt / tau), where tau is short for dynamic classes
(vehicles, pedestrians), so moving objects fade instead of leaving trails.
The map is rolled (not rebuilt) as the robot moves, in steps of `snap` metres.
"""
import math

import torch

from . import classes

_FIELDS = ("weight", "zmin", "zmax", "zmean", "zsq", "last_t")


class Layer:
    def __init__(self, name, half, cell, inner_half, num_classes, device):
        self.name, self.half, self.cell, self.inner_half = name, half, cell, inner_half
        self.n = int(round(2 * half / cell))
        a = int(round((half - inner_half) / cell))
        self.ring = torch.ones(self.n, self.n, dtype=torch.bool, device=device)
        if inner_half > 0:
            self.ring[a:self.n - a, a:self.n - a] = False
        shape = (self.n, self.n)
        self.weight = torch.zeros(shape, device=device)
        self.zmin = torch.zeros(shape, device=device)
        self.zmax = torch.zeros(shape, device=device)
        self.zmean = torch.zeros(shape, device=device)
        self.zsq = torch.zeros(shape, device=device)
        self.last_t = torch.zeros(shape, device=device)
        self.probs = torch.zeros((num_classes,) + shape, device=device, dtype=torch.float16)
        self.scan_pts = 0

    @property
    def ring_cells(self):
        return int(self.ring.sum())

    def nbytes(self):
        t = [getattr(self, f) for f in _FIELDS] + [self.probs]
        return sum(x.numel() * x.element_size() for x in t)

    def reset(self):
        for f in _FIELDS:
            getattr(self, f).zero_()
        self.probs.zero_()

    def roll(self, sx, sy):
        """Shift contents by (sx, sy) cells (map moved +sx, +sy) and clear new strips."""
        if abs(sx) >= self.n or abs(sy) >= self.n:
            self.reset()
            return
        for f in _FIELDS:
            setattr(self, f, _shift(getattr(self, f), sx, sy, 0))
        self.probs = _shift(self.probs, sx, sy, 1)


def _shift(t, sx, sy, d):
    """Roll grid so index (i, j) now holds old (i + sx, j + sy); zero wrapped cells."""
    t = torch.roll(t, shifts=(-sx, -sy), dims=(d, d + 1))
    ix = t.narrow(d, t.shape[d] - sx, sx) if sx > 0 else t.narrow(d, 0, -sx)
    ix.zero_()
    iy = t.narrow(d + 1, t.shape[d + 1] - sy, sy) if sy > 0 else t.narrow(d + 1, 0, -sy)
    iy.zero_()
    return t


def _pool_tallest(v, k):
    """k x k downsample of a layer view, taking class/confidence from the tallest cell."""
    e = torch.nan_to_num(v["elevation"], nan=float("-inf"))[None, None]
    pe, idx = torch.nn.functional.max_pool2d(e, k, k, return_indices=True)
    idx = idx.view(-1)
    out = {"elevation": pe.view(pe.shape[-2:])}
    out["elevation"][torch.isinf(out["elevation"])] = float("nan")
    for key in ("class", "confidence"):
        out[key] = v[key].reshape(-1)[idx].view(out["elevation"].shape)
    return out


class FoveatedMap:
    def __init__(self, zones, fusion, device="cuda", snap=None):
        self.device = torch.device(device)
        self.fusion = fusion
        self.C = classes.NUM_CLASSES
        cells = [z["cell"] for z in zones]
        self.snap = snap or max(cells)
        for c in cells:
            assert abs(self.snap / c - round(self.snap / c)) < 1e-6, "snap must be a multiple of every cell"
        self.layers, inner = [], 0.0
        for z in zones:
            self.layers.append(Layer(z["name"], z["half_extent"], z["cell"], inner, self.C, self.device))
            inner = z["half_extent"]
        self.halves = torch.tensor([l.half for l in self.layers], device=self.device)
        self.center_idx = None  # integer snap-cell coords of the map centre
        self.stamp = None
        self._dyn = classes.DYNAMIC.to(self.device)

    # ------------------------------------------------------------------ geometry
    @property
    def center_xy(self):
        return self.center_idx.float() * self.snap

    def recenter(self, robot_xy):
        new = torch.round(torch.as_tensor(robot_xy, dtype=torch.float32, device=self.device)[:2] / self.snap).long()
        if self.center_idx is None:
            self.center_idx = new
            return
        d = new - self.center_idx
        if d.abs().sum() == 0:
            return
        dx, dy = int(d[0]), int(d[1])
        for l in self.layers:
            k = int(round(self.snap / l.cell))
            l.roll(dx * k, dy * k)
        self.center_idx = new

    def zone_of(self, xy):
        """Layer index per point (by Chebyshev distance to the map centre), -1 if outside."""
        d = (xy - self.center_xy).abs().max(dim=1).values
        z = torch.bucketize(d, self.halves, right=True)
        z[z >= len(self.layers)] = -1
        return z

    # ------------------------------------------------------------------ update
    def _decay(self, dt):
        if dt <= 0:
            return
        f = self.fusion
        for l in self.layers:
            top = l.probs.argmax(0)
            tau = torch.where(self._dyn[top], f["tau_dynamic"], f["tau_static"])
            l.weight *= torch.exp(-dt / tau)
            l.weight[l.weight < f["min_weight"]] = 0.0

    def integrate(self, xyz, probs, robot_xy, stamp):
        """Fuse one scan. xyz: (N,3) in map frame, probs: (N,C) AdaptMap class probs."""
        xyz = xyz.to(self.device, torch.float32)
        probs = probs.to(self.device, torch.float32)
        self.recenter(robot_xy)
        if self.stamp is not None:
            self._decay(stamp - self.stamp)
        self.stamp = stamp

        zone = self.zone_of(xyz[:, :2])
        cxy = self.center_xy
        for i, l in enumerate(self.layers):
            m = zone == i
            l.scan_pts = int(m.sum())
            if l.scan_pts == 0:
                continue
            p, pr = xyz[m], probs[m]
            ij = torch.floor((p[:, :2] - cxy + l.half) / l.cell).long().clamp_(0, l.n - 1)
            self._fuse(l, ij[:, 0] * l.n + ij[:, 1], p[:, 2], pr, stamp)

    def _fuse(self, l, idx, z, pr, stamp):
        nn = l.n * l.n
        cnt = torch.zeros(nn, device=self.device).index_add_(0, idx, torch.ones_like(z))
        obs = cnt > 0
        zmin = torch.full((nn,), float("inf"), device=self.device).scatter_reduce_(0, idx, z, "amin")
        zmax = torch.full((nn,), float("-inf"), device=self.device).scatter_reduce_(0, idx, z, "amax")
        zsum = torch.zeros(nn, device=self.device).index_add_(0, idx, z)
        zsq = torch.zeros(nn, device=self.device).index_add_(0, idx, z * z)
        psum = torch.zeros(nn, self.C, device=self.device).index_add_(0, idx, pr)

        o = obs.nonzero().squeeze(1)
        n_o = cnt[o]
        w_old = l.weight.view(-1)[o]
        a = w_old / (w_old + 1.0)  # old share; each scan is one unit-weight observation
        b = 1.0 - a

        def blend(field, new):
            v = getattr(l, field).view(-1)
            v[o] = a * v[o] + b * new

        blend("zmin", zmin[o])
        blend("zmax", zmax[o])
        blend("zmean", zsum[o] / n_o)
        blend("zsq", zsq[o] / n_o)
        pv = l.probs.view(self.C, -1)
        pv[:, o] = (a * pv[:, o].float() + b * (psum[o] / n_o[:, None]).T).half()
        l.weight.view(-1)[o] = torch.clamp(w_old + 1.0, max=self.fusion["max_weight"])
        l.last_t.view(-1)[o] = stamp

    # ------------------------------------------------------------------ outputs
    def layer_view(self, l):
        """Per-layer 2.5D grids; cells outside the ring or never observed are NaN / class 0."""
        valid = (l.weight > 0) & l.ring
        nan = torch.tensor(float("nan"), device=self.device)
        conf_p, cls = l.probs.float().max(0)
        conf = conf_p * torch.clamp(l.weight / self.fusion["conf_support"], max=1.0)
        var = torch.clamp(l.zsq - l.zmean ** 2, min=0.0)
        return {
            "elevation": torch.where(valid, l.zmax, nan),
            "ground": torch.where(valid, l.zmin, nan),
            "mean": torch.where(valid, l.zmean, nan),
            "std": torch.where(valid, var.sqrt(), nan),
            "class": torch.where(valid, cls, torch.zeros_like(cls)),
            "confidence": torch.where(valid, conf, torch.zeros_like(conf)),
            "valid": valid,
            "origin": (self.center_xy - l.half).tolist(),  # world xy of cell (0, 0) corner
            "resolution": l.cell,
            "size": l.n,
        }

    def export(self):
        return {l.name: self.layer_view(l) for l in self.layers}

    def nav_view(self, cell=0.1, half=None):
        """Resample all layers onto one uniform grid (nearest, from the owning layer)."""
        half = half or self.layers[-1].half
        n = int(round(2 * half / cell))
        c = (torch.arange(n, device=self.device) + 0.5) * cell - half
        gx, gy = torch.meshgrid(c, c, indexing="ij")
        rel = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)
        zone = self.zone_of(rel + self.center_xy)
        out = {k: torch.full((n * n,), float("nan"), device=self.device) for k in ("elevation", "confidence")}
        out["class"] = torch.zeros(n * n, dtype=torch.long, device=self.device)
        for i, l in enumerate(self.layers):
            m = zone == i
            if not m.any():
                continue
            v = self.layer_view(l)
            res = l.cell
            k = int(round(cell / l.cell))
            if k > 1:  # finer layer: keep the tallest cell per output cell (obstacles survive)
                v, res = _pool_tallest(v, k), l.cell * k
            ij = torch.floor((rel[m] + l.half) / res).long().clamp_(0, v["class"].shape[0] - 1)
            for key in out:
                out[key][m] = v[key][ij[:, 0], ij[:, 1]].to(out[key].dtype)
        return {k: t.view(n, n) for k, t in out.items()} | {"origin": (self.center_xy - half).tolist(), "resolution": cell}

    # ------------------------------------------------------------------ stats
    def stats(self):
        finest = min(l.cell for l in self.layers)
        outer = self.layers[-1].half
        per = {l.name: {"cell": l.cell, "ring_cells": l.ring_cells,
                        "occupied": int(((l.weight > 0) & l.ring).sum()),
                        "scan_pts": l.scan_pts} for l in self.layers}
        total = sum(p["ring_cells"] for p in per.values())
        uniform = int(round(2 * outer / finest)) ** 2
        return {"layers": per, "total_cells": total, "uniform_cells": uniform,
                "reduction": uniform / total, "map_bytes": sum(l.nbytes() for l in self.layers)}
