"""P7: speed-adaptive zones - policy and content-preserving resampling."""
import pytest
import torch

from adaptmap import classes
from adaptmap.adaptive import ZonePolicy
from adaptmap.config import load_config
from adaptmap.foveated_map import FoveatedMap

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = load_config()


def onehot(n, name):
    p = torch.zeros(n, classes.NUM_CLASSES)
    p[:, classes.NAMES.index(name)] = 1.0
    return p


def block(cx, cy, h, size=0.3, step=0.02):
    g = torch.stack(torch.meshgrid(torch.arange(-size / 2, size / 2, step), torch.arange(-size / 2, size / 2, step),
                                   indexing="ij"), -1).reshape(-1, 2) + torch.tensor([cx, cy])
    return torch.cat([torch.cat([g, torch.full((len(g), 1), z)], 1) for z in torch.linspace(0, h, 6).tolist()])


def top_at(m, x, y):
    i = int(m.zone_of(torch.tensor([[x, y]], device=DEV))[0])
    l = m.layers[i]
    v = m.layer_view(l)
    ox, oy = v["origin"]
    return l.cell, v["elevation"][int((x - ox) / l.cell), int((y - oy) / l.cell)].item()


def test_policy_grows_fast_shrinks_with_hysteresis():
    p = ZonePolicy(CFG["zones"], CFG["adaptive_zones"], snap=0.5)
    assert p.update(1.0, 0.0) is None                     # slow: stays at the 10 m base
    z = p.update(8.0, 1.0)                                # fast: grows immediately
    assert z is not None and z[0]["half_extent"] > 20.0
    assert p.update(1.0, 2.0) is None                     # slowed down: not yet
    assert p.update(1.0, 4.0) is None
    z = p.update(1.0, 5.5)                                # after shrink_delay
    assert z is not None and z[0]["half_extent"] == CFG["zones"][0]["half_extent"]
    assert p.target(100.0) <= CFG["zones"][1]["half_extent"] - 0.5  # near zone stays inside the next one


def test_resize_keeps_obstacles_both_ways():
    m = FoveatedMap(CFG["zones"], CFG["fusion"], device=DEV)
    pts = torch.cat([block(15.0, 0.0, 1.2), block(4.0, 3.0, 0.8)])
    m.integrate(pts, onehot(len(pts), "pole"), [0, 0], 0.0)
    cell, h = top_at(m, 15.0, 0.0)
    assert cell == pytest.approx(0.10) and h == pytest.approx(1.2, abs=0.05)

    grow = [dict(z) for z in CFG["zones"]]
    grow[0]["half_extent"] = 20.0
    m.set_zones(grow)                                     # 15 m is now in the 5 cm zone (coarse -> fine)
    cell, h = top_at(m, 15.0, 0.0)
    assert cell == pytest.approx(0.05) and h == pytest.approx(1.2, abs=0.05)
    assert top_at(m, 4.0, 3.0)[1] == pytest.approx(0.8, abs=0.05)
    assert m.stats()["total_cells"] > 702_400

    m.set_zones(CFG["zones"])                             # back to 10 m (fine -> coarse, tallest wins)
    cell, h = top_at(m, 15.0, 0.0)
    assert cell == pytest.approx(0.10) and h == pytest.approx(1.2, abs=0.05)
    assert m.stats()["total_cells"] == 702_400
