"""P3: terrain analysis on scenes fused through the real map (points -> FoveatedMap -> nav_view)."""
import math

import pytest
import torch

from adaptmap import classes, terrain
from adaptmap.config import load_config
from adaptmap.foveated_map import FoveatedMap

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = load_config()
TC = CFG["terrain"]


def onehot(n, name):
    p = torch.zeros(n, classes.NUM_CLASSES)
    p[:, classes.NAMES.index(name)] = 1.0
    return p


def plane(x0, x1, y0, y1, zf, step=0.025):
    g = torch.stack(torch.meshgrid(torch.arange(x0, x1, step), torch.arange(y0, y1, step), indexing="ij"), -1).reshape(-1, 2)
    return torch.cat([g, zf(g[:, 0], g[:, 1])[:, None]], 1)


def run(parts):
    """parts: list of (points, class name). Returns (nav_view, terrain result, cell lookup fn)."""
    m = FoveatedMap(CFG["zones"], CFG["fusion"], device=DEV)
    pts = torch.cat([p for p, _ in parts])
    pr = torch.cat([onehot(len(p), c) for p, c in parts])
    m.integrate(pts, pr, [0, 0], 0.0)
    nv = m.nav_view(cell=TC["costmap_cell"], half=20.0)
    t = terrain.analyze(nv, TC)
    ox, oy = nv["origin"]
    at = lambda x, y: (int((x - ox) / nv["resolution"]), int((y - oy) / nv["resolution"]))  # noqa: E731
    return nv, t, at


def test_flat_road_is_free():
    _, t, at = run([(plane(2, 8, -3, 3, lambda x, y: torch.zeros_like(x)), "road")])
    assert t["cost"][at(5, 0)].item() == 0


def test_curb_detected_and_not_crossable():
    road = plane(2, 8, 2, 5, lambda x, y: torch.zeros_like(x))
    walk = plane(2, 8, 5, 8, lambda x, y: torch.full_like(x, 0.15))
    _, t, at = run([(road, "road"), (walk, "sidewalk_curb")])
    assert t["curb"][at(5, 4.95)] or t["curb"][at(5, 5.05)]
    assert t["cost"][at(5, 5.0)].item() == terrain.LETHAL
    assert t["cost"][at(5, 3.0)].item() == 0                        # road
    assert t["cost"][at(5, 7.0)].item() == TC["class_cost"]["sidewalk_curb"]  # sidewalk top is fine


@pytest.mark.parametrize("deg,lethal", [(10, False), (30, True)])
def test_slope(deg, lethal):
    k = math.tan(math.radians(deg))
    _, t, at = run([(plane(2, 8, -3, 3, lambda x, y: k * x), "terrain")])
    assert t["slope_deg"][at(5, 0)].item() == pytest.approx(deg, abs=2.0)
    assert (t["cost"][at(5, 0)].item() == terrain.LETHAL) == lethal


def test_obstacle_lethal_but_overhang_passable():
    ground = plane(2, 10, -3, 3, lambda x, y: torch.zeros_like(x))
    wall = torch.cat([plane(4, 4.4, -1, 1, lambda x, y, z=z: torch.full_like(x, z)) for z in (0.3, 0.6, 0.9)])
    crown = torch.cat([plane(7, 8, -1, 1, lambda x, y, z=z: torch.full_like(x, z)) for z in (2.5, 3.0, 3.5)])
    _, t, at = run([(ground, "road"), (wall, "building_wall"), (crown, "vegetation")])
    assert t["cost"][at(4.2, 0)].item() == terrain.LETHAL
    assert t["passable_under"][at(7.5, 0)]
    assert t["cost"][at(7.5, 0)].item() < terrain.LETHAL
