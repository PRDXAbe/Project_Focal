"""Foveated variable-resolution 2.5D semantic elevation map.

The map is a stack of nested square layers centred on the robot. Layer i covers
Chebyshev distance (half_{i-1}, half_i] from the map centre with its own cell
size, so detail is high near the robot and drops with distance. Each layer is a
dense GPU grid; cells inside the next-finer layer are masked out and never
written, so the effective cell count is only the ring.

Every cell stores a decayed running average of per-scan observations:
min/max/mean height, E[z^2] (-> variance), the class-probability vector, and two
separately-weighted surfaces: the mean height of its *ground* points and the lowest
*non-ground* point (bottom of whatever stands or hangs there). From these the views
derive a ground surface (interpolated under objects), height above ground and
clearance (free space under overhangs such as tree crowns).

Weights decay with exp(-dt / tau), where tau is short for dynamic classes
(vehicles, pedestrians). On top of that, *ray clearing* removes obstacles that the
current scan sees through: a LiDAR ray that passes inside a cell's stored obstacle
span (above its bottom, below its top) without ending there is evidence the cell is
now free. Views also fill small holes between LiDAR rings by interpolation (never
extrapolation), flagged as `filled` with reduced confidence.

The map is rolled (not rebuilt) as the robot moves, in steps of `snap` metres.
"""
import math

import torch
import torch.nn.functional as F

from . import classes

_FIELDS = ("weight", "zmin", "zmax", "zmean", "zsq", "last_t", "gw", "ground", "ow", "omin")


class Layer:
    def __init__(self, name, half, cell, inner_half, num_classes, device, fill_radius=0.0):
        self.name, self.half, self.cell, self.inner_half = name, half, cell, inner_half
        self.fill_radius = fill_radius
        self.n = int(round(2 * half / cell))
        a = int(round((half - inner_half) / cell))
        self.ring = torch.ones(self.n, self.n, dtype=torch.bool, device=device)
        if inner_half > 0:
            self.ring[a:self.n - a, a:self.n - a] = False
        shape = (self.n, self.n)
        for f in _FIELDS:
            setattr(self, f, torch.zeros(shape, device=device))
        self.probs = torch.zeros((num_classes,) + shape, device=device, dtype=torch.float16)
        self.scan_pts = 0
        self.seen = torch.zeros(shape, dtype=torch.bool, device=device)  # got points in the last scan

    @property
    def ring_cells(self):
        return int(self.ring.sum())

    def nbytes(self):
        t = [getattr(self, f) for f in _FIELDS] + [self.probs]
        return sum(x.numel() * x.element_size() for x in t)

    def reset(self, mask=None):
        """Clear all cells, or only the cells in `mask`."""
        if mask is None:
            for f in _FIELDS:
                getattr(self, f).zero_()
            self.probs.zero_()
            return
        for f in _FIELDS:
            getattr(self, f)[mask] = 0.0
        self.probs[:, mask] = 0.0

    def roll(self, sx, sy):
        """Shift contents by (sx, sy) cells (map moved +sx, +sy) and clear new strips."""
        if abs(sx) >= self.n or abs(sy) >= self.n:
            self.reset()
            return
        for f in _FIELDS:
            setattr(self, f, _shift(getattr(self, f), sx, sy, 0))
        self.probs = _shift(self.probs, sx, sy, 1)
        self.seen = _shift(self.seen, sx, sy, 0)


def _shift(t, sx, sy, d):
    """Roll grid so index (i, j) now holds old (i + sx, j + sy); zero wrapped cells."""
    t = torch.roll(t, shifts=(-sx, -sy), dims=(d, d + 1))
    ix = t.narrow(d, t.shape[d] - sx, sx) if sx > 0 else t.narrow(d, 0, -sx)
    ix.zero_()
    iy = t.narrow(d + 1, t.shape[d + 1] - sy, sy) if sy > 0 else t.narrow(d + 1, 0, -sy)
    iy.zero_()
    return t


def _rect_sum(x, a0, a1, b0, b1):
    """For every cell (i, j): sum of x over rows i+a0..i+a1 and cols j+b0..j+b1 (zero outside).

    Integral image in float64, so any window size costs the same and large sums stay exact.
    """
    k = max(abs(a0), abs(a1), abs(b0), abs(b1))
    n0, n1 = x.shape
    p = F.pad(x.double()[None, None], (k + 1, k, k + 1, k))[0, 0]
    I = p.cumsum(0).cumsum(1)
    r = torch.arange(n0, device=x.device) + k + 1
    c = torch.arange(n1, device=x.device) + k + 1
    r0, r1, c0, c1 = r + a0 - 1, r + a1, c + b0 - 1, c + b1
    s = I[r1][:, c1] - I[r0][:, c1] - I[r1][:, c0] + I[r0][:, c0]
    return s.to(x.dtype)


def _box(x, k):
    return _rect_sum(x, -k, k, -k, k)


_VIEW_KEYS = ("elevation", "ground", "hag", "clearance", "class", "confidence", "filled")


def _pool_tallest(v, k):
    """k x k downsample of a layer view, taking all other layers from the tallest cell."""
    e = torch.nan_to_num(v["elevation"], nan=float("-inf"))[None, None]
    pe, idx = F.max_pool2d(e, k, k, return_indices=True)
    idx = idx.view(-1)
    out = {"elevation": pe.view(pe.shape[-2:])}
    out["elevation"][torch.isinf(out["elevation"])] = float("nan")
    for key in _VIEW_KEYS[1:]:
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
            self.layers.append(Layer(z["name"], z["half_extent"], z["cell"], inner, self.C, self.device,
                                     z.get("fill_radius", 0.0)))
            inner = z["half_extent"]
        self.halves = torch.tensor([l.half for l in self.layers], device=self.device)
        self.center_idx = None  # integer snap-cell coords of the map centre
        self.stamp = None
        self._dyn = classes.DYNAMIC.to(self.device)
        self._ground_cls = classes.GROUND.to(self.device)
        self.cleared = 0  # cells reset by ray clearing in the last scan

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

    def _cell_index(self, l, xy):
        ij = torch.floor((xy - self.center_xy + l.half) / l.cell).long().clamp_(0, l.n - 1)
        return ij[:, 0] * l.n + ij[:, 1]

    # ------------------------------------------------------------------ update
    def _decay(self, dt):
        if dt <= 0:
            return
        f = self.fusion
        for l in self.layers:
            top = l.probs.argmax(0)
            tau = torch.where(self._dyn[top], f["tau_dynamic"], f["tau_static"])
            k = torch.exp(-dt / tau)
            for w in (l.weight, l.gw, l.ow):
                w *= k
                w[w < f["min_weight"]] = 0.0

    def integrate(self, xyz, probs, robot_xy, stamp, origin=None):
        """Fuse one scan. xyz: (N,3) in map frame, probs: (N,C) AdaptMap class probs.

        origin: sensor position (3,) in the map frame; enables ray clearing when given.
        """
        xyz = xyz.to(self.device, torch.float32)
        probs = probs.to(self.device, torch.float32)
        self.recenter(robot_xy)
        if self.stamp is not None:
            self._decay(stamp - self.stamp)
        self.stamp = stamp

        is_ground = probs[:, self._ground_cls].sum(1) > 0.5
        zone = self.zone_of(xyz[:, :2])
        for i, l in enumerate(self.layers):
            m = zone == i
            l.scan_pts = int(m.sum())
            l.seen.zero_()
            if l.scan_pts == 0:
                continue
            self._fuse(l, self._cell_index(l, xyz[m, :2]), xyz[m, 2], probs[m], is_ground[m], stamp)
        self.cleared = 0
        if origin is not None and self.fusion.get("ray_clearing", False):
            self._clear_rays(torch.as_tensor(origin, dtype=torch.float32, device=self.device), xyz)

    def _fuse(self, l, idx, z, pr, g, stamp):
        nn = l.n * l.n
        dev = self.device
        cnt = torch.zeros(nn, device=dev).index_add_(0, idx, torch.ones_like(z))
        zmin = torch.full((nn,), float("inf"), device=dev).scatter_reduce_(0, idx, z, "amin")
        zmax = torch.full((nn,), float("-inf"), device=dev).scatter_reduce_(0, idx, z, "amax")
        zsum = torch.zeros(nn, device=dev).index_add_(0, idx, z)
        zsq = torch.zeros(nn, device=dev).index_add_(0, idx, z * z)
        psum = torch.zeros(nn, self.C, device=dev).index_add_(0, idx, pr)
        gf = g.float()
        gcnt = torch.zeros(nn, device=dev).index_add_(0, idx, gf)
        gsum = torch.zeros(nn, device=dev).index_add_(0, idx, z * gf)
        omin = torch.full((nn,), float("inf"), device=dev).scatter_reduce_(
            0, idx, torch.where(g, torch.full_like(z, float("inf")), z), "amin")

        o = (cnt > 0).nonzero().squeeze(1)
        l.seen.view(-1)[o] = True
        n_o = cnt[o]
        w_old = l.weight.view(-1)[o]
        a = w_old / (w_old + 1.0)  # old share; each scan is one unit-weight observation
        b = 1.0 - a

        def blend(field, new, at=o, aa=a, bb=b):
            v = getattr(l, field).view(-1)
            v[at] = aa * v[at] + bb * new

        blend("zmin", zmin[o])
        blend("zmax", zmax[o])
        blend("zmean", zsum[o] / n_o)
        blend("zsq", zsq[o] / n_o)
        pv = l.probs.view(self.C, -1)
        pv[:, o] = (a * pv[:, o].float() + b * (psum[o] / n_o[:, None]).T).half()
        wmax = self.fusion["max_weight"]
        l.weight.view(-1)[o] = torch.clamp(w_old + 1.0, max=wmax)
        l.last_t.view(-1)[o] = stamp

        # ground surface and obstacle bottom, each with its own observation weight
        gcells = (gcnt > 0).nonzero().squeeze(1)
        ocells = torch.isfinite(omin).nonzero().squeeze(1)
        for wname, field, cells, new in (("gw", "ground", gcells, gsum[gcells] / gcnt[gcells]),
                                         ("ow", "omin", ocells, omin[ocells])):
            if cells.numel() == 0:
                continue
            w = getattr(l, wname).view(-1)
            wo = w[cells]
            aa = wo / (wo + 1.0)
            blend(field, new, cells, aa, 1.0 - aa)
            w[cells] = torch.clamp(wo + 1.0, max=wmax)

    def _clear_rays(self, origin, xyz):
        """Remove stored obstacles that this scan's rays pass through (see module docstring)."""
        f = self.fusion
        rays = xyz[:: max(1, int(f.get("clear_ray_stride", 4)))]
        d = rays - origin
        L = d.norm(dim=1)
        keep = (L > 1.0) & (L < f.get("clear_range", 50.0))
        d, L = d[keep], L[keep]
        if d.shape[0] == 0:
            return
        K = int(f.get("clear_samples", 48))
        t = (torch.arange(K, device=self.device, dtype=torch.float32) + 0.5) / K  # (K,)
        q = origin + d[:, None, :] * t[None, :, None]                                # (R,K,3)
        along = L[:, None] * t[None, :]
        ok = (along > 0.5) & (L[:, None] - along > f.get("clear_end_margin", 0.3))   # skip sensor + endpoint
        q = q[ok]
        margin = f.get("clear_margin", 0.2)
        zone = self.zone_of(q[:, :2])
        for i, l in enumerate(self.layers):
            m = zone == i
            if not m.any():
                continue
            idx = self._cell_index(l, q[m, :2])
            z = q[m, 2]
            ow, omin, zmax, seen = l.ow.view(-1)[idx], l.omin.view(-1)[idx], l.zmax.view(-1)[idx], l.seen.view(-1)[idx]
            inside = (ow > 0) & ~seen & (z > omin + margin) & (z < zmax - margin)
            if not inside.any():
                continue
            hits = torch.zeros(l.n * l.n, device=self.device).index_add_(0, idx[inside], torch.ones_like(z[inside]))
            hit = (hits >= f.get("clear_min_hits", 1)).view(l.n, l.n)
            for w in (l.weight, l.ow):
                w[hit] *= f.get("clear_factor", 0.5)
            gone = hit & (l.ow < f["min_weight"])
            if gone.any():
                l.reset(gone)
                self.cleared += int(gone.sum())

    # ------------------------------------------------------------------ outputs
    def layer_view(self, l):
        """Per-layer 2.5D grids.

        elevation  top surface (observed, or interpolated ground where `filled`)
        ground     ground height: observed ground, else interpolated from nearby ground cells
        hag        height above ground (elevation - ground)
        clearance  free height under the lowest obstacle point (NaN where nothing hangs/stands)
        filled     cell had no data; ground interpolated between neighbours on opposite sides
        Cells outside the ring or unknown are NaN / class 0.
        """
        f = self.fusion
        valid = (l.weight > 0) & l.ring
        nan = torch.tensor(float("nan"), device=self.device)
        pf = l.probs.float()
        conf_p, cls = pf.max(0)
        conf = conf_p * torch.clamp(l.weight / f["conf_support"], max=1.0)
        var = torch.clamp(l.zsq - l.zmean ** 2, min=0.0)

        # ground surface: own ground where observed, else average of ground cells within ground_radius
        has_g = (l.gw > 0) & valid
        kg = max(1, int(math.ceil(f.get("ground_radius", 2.0) / l.cell)))
        gsum, gcnt = _box(torch.where(has_g, l.ground, 0.0), kg), _box(has_g.float(), kg)
        ground = torch.where(has_g, l.ground, torch.where(gcnt > 0, gsum / gcnt.clamp(min=1), nan))

        # hole filling between rings: unknown cell with ground support on BOTH opposite sides
        filled = torch.zeros_like(valid)
        elevation = torch.where(valid, l.zmax, nan)
        out_cls = torch.where(valid, cls, torch.zeros_like(cls))
        out_conf = torch.where(valid, conf, torch.zeros_like(conf))
        kf = int(round(l.fill_radius / l.cell))
        if kf >= 1:
            gm = (has_g & self._ground_cls[cls]).float()
            # each side is a narrow band in line with the cell, so a ring running parallel to the test
            # direction can't count as support on both sides (that would be extrapolation)
            bw = max(1, kf // 3)
            side = lambda a0, a1, b0, b1: _rect_sum(gm, a0, a1, b0, b1) > 0  # noqa: E731
            between = (side(-kf, -1, -bw, bw) & side(1, kf, -bw, bw)) | (side(-bw, bw, -kf, -1) & side(-bw, bw, 1, kf))
            filled = ~valid & l.ring & between & torch.isfinite(ground)
            if filled.any():
                gidx = self._ground_cls.nonzero().squeeze(1)
                pg = torch.stack([_box(pf[c] * gm, kf) for c in gidx.tolist()])  # ground-class votes
                fill_cls = gidx[pg.argmax(0)]
                fill_conf = 0.5 * _box(conf * gm, kf) / _box(gm, kf).clamp(min=1)
                elevation = torch.where(filled, ground, elevation)
                out_cls = torch.where(filled, fill_cls, out_cls)
                out_conf = torch.where(filled, fill_conf, out_conf)
        known = valid | filled
        has_o = (l.ow > 0) & valid
        return {
            "elevation": elevation,
            "ground": torch.where(known, ground, nan),
            "hag": torch.where(known, elevation - ground, nan),
            "clearance": torch.where(has_o, l.omin - ground, nan),
            "mean": torch.where(valid, l.zmean, nan),
            "std": torch.where(valid, var.sqrt(), nan),
            "class": out_cls,
            "confidence": out_conf,
            "valid": valid,
            "filled": filled,
            "known": known,
            "origin": (self.center_xy - l.half).tolist(),  # world xy of cell (0, 0) corner
            "resolution": l.cell,
            "size": l.n,
        }

    def export(self):
        return {l.name: self.layer_view(l) for l in self.layers}

    def nav_view(self, cell=0.1, half=None, views=None):
        """Resample all layers onto one uniform grid (nearest, from the owning layer).

        views: an `export()` result to reuse (layer views are the expensive part).
        """
        half = half or self.layers[-1].half
        n = int(round(2 * half / cell))
        c = (torch.arange(n, device=self.device) + 0.5) * cell - half
        gx, gy = torch.meshgrid(c, c, indexing="ij")
        rel = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)
        zone = self.zone_of(rel + self.center_xy)
        out = {k: torch.full((n * n,), float("nan"), device=self.device)
               for k in ("elevation", "ground", "hag", "clearance", "confidence")}
        out["class"] = torch.zeros(n * n, dtype=torch.long, device=self.device)
        out["filled"] = torch.zeros(n * n, dtype=torch.bool, device=self.device)
        for i, l in enumerate(self.layers):
            m = zone == i
            if not m.any():
                continue
            v = views[l.name] if views is not None else self.layer_view(l)
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
                "reduction": uniform / total, "map_bytes": sum(l.nbytes() for l in self.layers),
                "cleared": self.cleared}
