import math

import pytest
import torch

from adaptmap import classes
from adaptmap.config import load_config
from adaptmap.foveated_map import FoveatedMap
from adaptmap.preprocess import preprocess

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = load_config()


def make_map(**fusion):
    return FoveatedMap(CFG["zones"], CFG["fusion"] | fusion, device=DEV)


def one_hot(n, name):
    p = torch.zeros(n, classes.NUM_CLASSES)
    p[:, classes.NAMES.index(name)] = 1.0
    return p


def test_cell_counts_match_design():
    s = make_map().stats()
    rings = [l["ring_cells"] for l in s["layers"].values()]
    assert rings == [160_000, 320_000, 102_400, 120_000]
    assert s["total_cells"] == 702_400
    assert s["uniform_cells"] == 16_000_000
    assert s["reduction"] == pytest.approx(22.78, rel=1e-3)


def test_zone_assignment_chebyshev():
    m = make_map()
    m.recenter([0.0, 0.0])
    xy = torch.tensor([[5, 0], [9.99, -9.99], [10, 0], [0, 29], [-45, 3], [70, 70], [101, 0]], device=DEV).float()
    assert m.zone_of(xy).tolist() == [0, 0, 1, 1, 2, 3, -1]


def test_height_and_class_fusion_per_zone():
    m = make_map()
    # a 1 m-tall box (pole) at 5 m and at 70 m, plus flat road at z=0
    pts, probs = [], []
    for cx in (5.0, 70.0):
        g = torch.stack(torch.meshgrid(torch.linspace(-0.2, 0.2, 9), torch.linspace(-0.2, 0.2, 9), indexing="ij"), -1).reshape(-1, 2)
        for z in torch.linspace(0, 1.0, 11):
            pts.append(torch.cat([g + torch.tensor([cx, 0.0]), torch.full((g.shape[0], 1), float(z))], 1))
            probs.append(one_hot(g.shape[0], "pole"))
    rg = torch.stack(torch.meshgrid(torch.arange(1.0, 5.0, 0.025), torch.arange(3.0, 7.0, 0.025), indexing="ij"), -1).reshape(-1, 2)
    road = torch.cat([rg + 0.0125, torch.zeros(rg.shape[0], 1)], 1)
    pts.append(road)
    probs.append(one_hot(road.shape[0], "road"))
    m.integrate(torch.cat(pts), torch.cat(probs), [0.0, 0.0], stamp=0.0)

    near = m.layer_view(m.layers[0])
    i, j = int((5.0 + 10) / 0.05), int((0.0 + 10) / 0.05)
    assert near["elevation"][i, j].item() == pytest.approx(1.0, abs=1e-4)
    assert near["class"][i, j].item() == classes.NAMES.index("pole")

    far = m.layer_view(m.layers[3])
    i, j = int((70.0 + 100) / 0.5), int((0.0 + 100) / 0.5)
    assert far["elevation"][i, j].item() == pytest.approx(1.0, abs=1e-4)
    assert far["class"][i, j].item() == classes.NAMES.index("pole")

    ri, rj = int((3.0 + 10) / 0.05), int((5.0 + 10) / 0.05)
    assert near["class"][ri, rj].item() == classes.NAMES.index("road")
    assert near["elevation"][ri, rj].item() == pytest.approx(0.0, abs=1e-4)
    # inner region of coarse layers stays empty
    assert not far["valid"][~m.layers[3].ring].any()


def test_dynamic_cells_fade_static_persist():
    m = make_map()
    ped = torch.tensor([[3.0, 3.0, 1.0]])
    wall = torch.tensor([[-3.0, -3.0, 2.0]])
    m.integrate(torch.cat([ped, wall]), torch.cat([one_hot(1, "pedestrian"), one_hot(1, "building_wall")]), [0, 0], 0.0)
    m.integrate(torch.zeros(0, 3), torch.zeros(0, classes.NUM_CLASSES), [0, 0], 3.0)  # 3 s, nothing seen
    v = m.layer_view(m.layers[0])
    pi, pj = int(13 / 0.05), int(13 / 0.05)
    wi, wj = int(7 / 0.05), int(7 / 0.05)
    assert not v["valid"][pi, pj]          # pedestrian gone (tau 0.5 s)
    assert v["valid"][wi, wj]              # wall remembered (tau 20 s)


def test_rolling_keeps_world_position():
    m = make_map()
    m.integrate(torch.tensor([[2.0, 1.0, 0.7]]), one_hot(1, "pole"), [0, 0], 0.0)
    m.integrate(torch.zeros(0, 3), torch.zeros(0, classes.NUM_CLASSES), [1.5, -0.5], 0.1)
    v = m.layer_view(m.layers[0])
    ox, oy = v["origin"]
    i, j = int((2.0 - ox) / 0.05), int((1.0 - oy) / 0.05)
    assert v["elevation"][i, j].item() == pytest.approx(0.7, abs=1e-4)


def test_nav_view_shapes():
    m = make_map()
    m.integrate(torch.tensor([[2.0, 1.0, 0.7], [60.0, 0.0, 1.5]]), one_hot(2, "vehicle"), [0, 0], 0.0)
    nv = m.nav_view(cell=0.5)
    assert nv["elevation"].shape == (400, 400)
    assert torch.isfinite(nv["elevation"]).sum() == 2


def test_preprocess_filters():
    pts = torch.tensor([
        [5.0, 0, 0, 0.3],          # keep
        [float("nan"), 0, 0, 0],   # invalid
        [0.1, 0.1, 0, 0],          # too close
        [150.0, 0, 0, 0],          # too far
        [5.0, 0, 9.0, 0],          # above ROI
    ], device=DEV)
    cloud = torch.cat([pts, torch.tensor([[5.05, 0.02, 0.01, 0.3], [5.02, -0.03, 0.0, 0.3]], device=DEV)])
    out, stats = preprocess(cloud, CFG["preprocess"])
    assert out.shape == (3, 4)
    assert torch.all(out[:, 3] == CFG["preprocess"]["constant_intensity"])
    assert stats["raw"] == 7
