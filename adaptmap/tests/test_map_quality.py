"""P2: ray clearing, ground surface / height above ground, clearance under overhangs, hole filling."""
import pytest
import torch

from adaptmap import classes
from adaptmap.config import load_config
from adaptmap.foveated_map import FoveatedMap, _rect_sum

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = load_config()


def make_map():
    return FoveatedMap(CFG["zones"], CFG["fusion"], device=DEV)


def onehot(n, name):
    p = torch.zeros(n, classes.NUM_CLASSES)
    p[:, classes.NAMES.index(name)] = 1.0
    return p


def grid(x0, x1, y0, y1, step, z):
    g = torch.stack(torch.meshgrid(torch.arange(x0, x1, step), torch.arange(y0, y1, step), indexing="ij"), -1).reshape(-1, 2)
    return torch.cat([g + step / 2, torch.full((g.shape[0], 1), float(z))], 1)


def box(x0, x1, y0, y1, z0, z1, step=0.025):
    return torch.cat([grid(x0, x1, y0, y1, step, z) for z in torch.arange(z0, z1 + 1e-6, 0.1).tolist()])


def cell(m, layer, x, y):
    l = m.layers[layer]
    ox, oy = (m.center_xy - l.half).tolist()
    return int((x - ox) / l.cell), int((y - oy) / l.cell)


def test_rect_sum_matches_brute_force():
    x = torch.rand(23, 17, device=DEV)
    for a0, a1, b0, b1 in ((-2, 2, -3, 3), (-4, -1, -2, 2), (1, 3, -1, 0)):
        got = _rect_sum(x, a0, a1, b0, b1)
        exp = torch.zeros_like(x)
        for i in range(23):
            for j in range(17):
                r0, r1, c0, c1 = max(0, i + a0), min(22, i + a1), max(0, j + b0), min(16, j + b1)
                if r0 <= r1 and c0 <= c1:
                    exp[i, j] = x[r0:r1 + 1, c0:c1 + 1].sum()
        assert torch.allclose(got, exp, atol=1e-4)


def _ghost_then_rays(ray_z):
    """A static 1 m box at x=4 m, then scans whose rays pass through it at height ray_z."""
    m = make_map()
    pts = box(4.0, 4.4, -0.2, 0.2, 0.0, 1.0)
    m.integrate(pts, onehot(len(pts), "building_wall"), [0, 0], 0.0, origin=[0, 0, ray_z])
    far = grid(8.0, 8.2, -0.3, 0.3, 0.02, ray_z)  # what the rays actually hit now (a wall 8 m away)
    for k in range(1, 7):
        m.integrate(far, onehot(len(far), "building_wall"), [0, 0], 0.1 * k, origin=[0, 0, ray_z])
    i, j = cell(m, 0, 4.2, 0.0)
    return m, m.layer_view(m.layers[0])["valid"][i, j].item()


def test_ray_clearing_removes_object_seen_through():
    m, still_there = _ghost_then_rays(0.5)
    assert not still_there
    assert m.stats()["cleared"] >= 0  # stat exists


def test_rays_above_object_do_not_clear_it():
    _, still_there = _ghost_then_rays(1.5)
    assert still_there


def test_clearance_under_overhang_and_height_above_ground():
    m = make_map()
    road = grid(2.0, 3.0, 2.0, 3.0, 0.025, 0.0)
    crown = box(2.0, 3.0, 2.0, 3.0, 3.0, 4.0)
    pts = torch.cat([road, crown])
    pr = torch.cat([onehot(len(road), "road"), onehot(len(crown), "vegetation")])
    m.integrate(pts, pr, [0, 0], 0.0)
    v = m.layer_view(m.layers[0])
    i, j = cell(m, 0, 2.5, 2.5)
    assert v["ground"][i, j].item() == pytest.approx(0.0, abs=0.02)
    assert v["hag"][i, j].item() == pytest.approx(4.0, abs=0.05)
    assert v["clearance"][i, j].item() == pytest.approx(3.0, abs=0.05)


def test_ground_interpolated_under_object():
    m = make_map()
    road = grid(3.0, 7.0, 3.0, 7.0, 0.025, 0.2)
    road = road[~(((road[:, 0] - 5).abs() < 0.6) & ((road[:, 1] - 5).abs() < 0.6))]  # no ground under the box
    obj = box(4.5, 5.5, 4.5, 5.5, 0.3, 1.7)
    pts = torch.cat([road, obj])
    m.integrate(pts, torch.cat([onehot(len(road), "road"), onehot(len(obj), "building_wall")]), [0, 0], 0.0)
    v = m.layer_view(m.layers[0])
    i, j = cell(m, 0, 5.0, 5.0)
    assert v["ground"][i, j].item() == pytest.approx(0.2, abs=0.02)
    assert v["hag"][i, j].item() == pytest.approx(1.5, abs=0.05)


def test_hole_fill_interpolates_between_rings_but_not_beyond():
    m = make_map()
    # ground "rings" every 1 m in the 10 cm zone (y = 12..15), like LiDAR rings at ~20 m
    rings = torch.cat([grid(12.0, 16.0, y, y + 0.1, 0.025, 0.0) for y in (12.0, 13.0, 14.0, 15.0)])
    m.integrate(rings, onehot(len(rings), "road"), [0, 0], 0.0)
    v = m.layer_view(m.layers[1])
    i, j = cell(m, 1, 14.0, 13.55)      # between two rings -> filled
    assert v["filled"][i, j] and not v["valid"][i, j]
    assert v["class"][i, j].item() == classes.NAMES.index("road")
    assert v["elevation"][i, j].item() == pytest.approx(0.0, abs=0.02)
    i, j = cell(m, 1, 14.0, 15.65)      # beyond the last ring -> left unknown
    assert not v["filled"][i, j] and not v["known"][i, j]


def _walk_through(ground_clear_step):
    """Road at x=4 m is seen for 10 scans, a pedestrian crosses it for 3, then the road is clear again.
    Returns the max height above ground left in its cells 2 s (6 scans at 3 Hz) later."""
    m = make_map()
    m.fusion = dict(m.fusion, ground_clear_step=ground_clear_step)
    road = grid(2.0, 6.0, -1.0, 1.0, 0.025, 0.0)
    ped = box(3.8, 4.2, -0.2, 0.2, 0.1, 1.7)
    for k in range(19):  # no origin: ray clearing is off, only ground clearing can help
        pts, probs = road, onehot(len(road), "road")
        if 10 <= k < 13:
            pts, probs = torch.cat([road, ped]), torch.cat([probs, onehot(len(ped), "pedestrian")])
        m.integrate(pts, probs, [0, 0], 0.3 * k)
    v = m.layer_view(m.layers[0])
    i0, j0 = cell(m, 0, 3.8, -0.2)
    i1, j1 = cell(m, 0, 4.2, 0.2)
    return torch.nan_to_num(v["hag"][i0:i1 + 1, j0:j1 + 1], nan=0.0).max().item()


def test_ground_clearing_removes_pedestrian_trail():
    assert _walk_through(1e9) > CFG["terrain"]["max_step"]  # without it a lethal residue remains
    assert _walk_through(CFG["fusion"]["ground_clear_step"]) < 0.05


def _step_in(appear_step):
    """Road seen for 10 scans, then a pedestrian steps onto it: height above ground after one scan."""
    m = make_map()
    m.fusion = dict(m.fusion, appear_step=appear_step)
    road = grid(2.0, 6.0, -1.0, 1.0, 0.025, 0.0)
    ped = box(3.8, 4.2, -0.2, 0.2, 0.1, 1.7)
    for k in range(11):
        pts, probs = road, onehot(len(road), "road")
        if k == 10:
            pts, probs = torch.cat([road, ped]), torch.cat([probs, onehot(len(ped), "pedestrian")])
        m.integrate(pts, probs, [0, 0], 0.3 * k)
    v = m.layer_view(m.layers[0])
    i, j = cell(m, 0, 4.0, 0.0)
    return v["hag"][i, j].item(), classes.NAMES[int(v["class"][i, j])]


def test_new_obstacle_appears_in_one_scan():
    hag, _ = _step_in(1e9)
    assert hag < 0.3                                  # averaged in at ~1/11: still looks like road
    hag, cls = _step_in(CFG["fusion"]["appear_step"])
    assert hag > 1.5 and cls == "pedestrian"


def test_ground_clearing_keeps_obstacle_still_seen():
    m = make_map()
    road = grid(2.0, 6.0, -1.0, 1.0, 0.025, 0.0)
    crate = box(3.8, 4.2, -0.2, 0.2, 0.1, 0.8)
    for k in range(20):
        pts = torch.cat([road, crate])
        probs = torch.cat([onehot(len(road), "road"), onehot(len(crate), "other_obstacle")])
        m.integrate(pts, probs, [0, 0], 0.3 * k)
    i, j = cell(m, 0, 4.0, 0.0)
    assert m.layer_view(m.layers[0])["hag"][i, j].item() > 0.7


def test_footprint_clearing():
    """Stale obstacle cells under the robot (blind disc, can't be re-observed) are forgotten; a box just in
    front of the footprint is kept."""
    m = make_map()
    road = grid(2.0, 6.0, -1.0, 1.0, 0.025, 0.0)
    under, ahead = box(3.7, 4.0, -0.2, 0.2, 0.1, 1.2), box(4.7, 4.9, -0.1, 0.1, 0.1, 0.8)
    pts = torch.cat([road, under, ahead])
    probs = torch.cat([onehot(len(road), "road"), onehot(len(under), "vehicle"), onehot(len(ahead), "other_obstacle")])
    for k in range(5):
        m.integrate(pts, probs, [0, 0], 0.3 * k)
    m.clear_footprint((4.0, 0.0), 0.0, 0.495, 0.335)
    v = m.layer_view(m.layers[0])
    i, j = cell(m, 0, 3.8, 0.0)
    assert not (v["hag"][i, j] > 0.12)                 # unknown (NaN) or ground, not an obstacle
    i, j = cell(m, 0, 4.8, 0.0)
    assert v["hag"][i, j].item() > 0.7
