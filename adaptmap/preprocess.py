"""Point-cloud preprocessing on the GPU: invalid/range/ROI/self-body/outlier filtering."""
import torch


def _voxel_keys(xyz, size):
    v = torch.floor(xyz / size).long()
    v -= v.min(0).values
    dims = v.max(0).values + 1
    return (v[:, 0] * dims[1] + v[:, 1]) * dims[2] + v[:, 2], v, dims


def isolation_filter(xyz, voxel, min_pts):
    """Drop points in sparse voxels that also have no occupied 6-neighbour (cheap SOR stand-in)."""
    if xyz.shape[0] == 0:
        return torch.ones(0, dtype=torch.bool, device=xyz.device)
    key, v, dims = _voxel_keys(xyz, voxel)
    uniq, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    sparse = cnt[inv] < min_pts
    if not sparse.any():
        return torch.ones_like(sparse)
    keep = ~sparse
    cand = sparse.nonzero().squeeze(1)
    has_nb = torch.zeros(cand.shape[0], dtype=torch.bool, device=xyz.device)
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        nv = v[cand] + torch.tensor(d, device=xyz.device)
        inb = ((nv >= 0) & (nv < dims)).all(1)
        nk = (nv[:, 0] * dims[1] + nv[:, 1]) * dims[2] + nv[:, 2]
        pos = torch.searchsorted(uniq, nk).clamp(max=uniq.numel() - 1)
        has_nb |= inb & (uniq[pos] == nk)
    keep[cand] = has_nb
    return keep


def adaptive_downsample(xyz, zones):
    """Compute-side foveation: keep one point per (zone cell / 2) voxel, by planar distance."""
    d = xyz[:, :2].abs().max(1).values
    keep = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    inner = 0.0
    for z in zones:
        m = ((d >= inner) & (d < z["half_extent"])).nonzero().squeeze(1)
        inner = z["half_extent"]
        if m.numel() == 0:
            continue
        key, _, _ = _voxel_keys(xyz[m], z["cell"] / 2)
        _, inv = torch.unique(key, return_inverse=True)
        first = torch.full((int(inv.max()) + 1,), m.numel(), device=xyz.device, dtype=torch.long)
        first.scatter_reduce_(0, inv, torch.arange(m.numel(), device=xyz.device), "amin")
        keep[m[first]] = True
    return keep


def preprocess(points, cfg, zones=None):
    """points: (N, >=3) tensor in sensor frame [x, y, z, (intensity)].

    Returns filtered (M, 4) points with intensity replaced by a constant, plus per-stage counts.
    """
    stats = {"raw": points.shape[0]}
    xyz = points[:, :3]
    m = torch.isfinite(xyz).all(1)
    r = xyz.norm(dim=1)
    m &= (r > cfg["min_range"]) & (r < cfg["max_range"])
    stats["valid_range"] = int(m.sum())
    m &= (xyz[:, 2] > cfg["z_min"]) & (xyz[:, 2] < cfg["z_max"])
    x0, x1, y0, y1, z0, z1 = cfg["self_box"]
    body = (xyz[:, 0] > x0) & (xyz[:, 0] < x1) & (xyz[:, 1] > y0) & (xyz[:, 1] < y1) & (xyz[:, 2] > z0) & (xyz[:, 2] < z1)
    m &= ~body
    xyz = xyz[m]
    stats["roi"] = xyz.shape[0]
    xyz = xyz[isolation_filter(xyz, cfg["outlier_voxel"], cfg["outlier_min_pts"])]
    stats["denoised"] = xyz.shape[0]
    if cfg.get("adaptive_downsample") and zones:
        xyz = xyz[adaptive_downsample(xyz, zones)]
    stats["out"] = xyz.shape[0]
    inten = torch.full((xyz.shape[0], 1), cfg["constant_intensity"], device=xyz.device, dtype=xyz.dtype)
    return torch.cat([xyz, inten], 1), stats
