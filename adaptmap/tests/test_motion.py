"""P5: motion-aware keep-out. A walking blob gets a lethal corridor ahead of it; a standing one does not."""
import numpy as np
import torch

from adaptmap.config import load_config
from adaptmap.motion import MotionPredictor

MC = load_config()["terrain"]["motion"]
RES, N, ORIGIN = 0.2, 100, (-10.0, -10.0)   # 20 x 20 m grid centred on the world origin


def person_mask(x, y):
    """A 0.5 m blob (what the map gives for a pedestrian) at world (x, y)."""
    m = torch.zeros(N, N, dtype=torch.bool)
    i, j = round((x - ORIGIN[0]) / RES), round((y - ORIGIN[1]) / RES)
    m[i - 1:i + 2, j - 1:j + 2] = True
    return m


def cell(m, x, y):
    return bool(m[int((x - ORIGIN[0]) / RES), int((y - ORIGIN[1]) / RES)])


def walk(v, steps=6, hz=3.0, y0=-6.0):
    """A pedestrian crossing (walking +y at x = 0) seen at the costmap rate; returns the last prediction."""
    mp = MotionPredictor(MC)
    for k in range(steps):
        pred = mp.update(person_mask(0.0, y0 + v * k / hz), ORIGIN, RES, k / hz)
    return mp, pred, y0 + v * (steps - 1) / hz


def test_crossing_pedestrian_blocks_path_ahead():
    mp, pred, y = walk(1.2)
    (t,) = mp.moving()
    assert abs(t["vel"][1] - 1.2) < 0.15 and abs(t["vel"][0]) < 0.15
    ahead = y + 1.2 * MC["horizon"] * 0.9
    assert cell(pred, 0.0, y + 1.0) and cell(pred, 0.0, ahead)     # the lane it is walking into
    assert not cell(pred, 0.0, y - 1.5)                            # not behind it
    assert not cell(pred, 2.0, y + 1.0)                            # not beside the corridor


def test_standing_pedestrian_and_new_blob_not_predicted():
    _, pred, _ = walk(0.0)
    assert not pred.any()
    mp = MotionPredictor(MC)                                       # 2 sightings: not confirmed yet
    for k in range(2):
        pred = mp.update(person_mask(0.0, -6.0 + 0.4 * k), ORIGIN, RES, k / 3)
    assert not pred.any()


def test_two_pedestrians_tracked_separately():
    mp = MotionPredictor(MC)
    for k in range(6):
        m = person_mask(-4.0, -6.0 + 0.4 * k) | person_mask(4.0, 6.0 - 0.4 * k)
        mp.update(m, ORIGIN, RES, k / 3)
    vy = sorted(t["vel"][1] for t in mp.moving())
    assert len(vy) == 2 and np.allclose(vy, [-1.2, 1.2], atol=0.15)


def test_jittering_static_blob_not_predicted():
    """A pole partly labelled pedestrian: its centroid jumps 0.6 m back and forth as a passer-by merges
    with it or the view angle changes. Fast, but going nowhere, so it gets no corridor."""
    mp = MotionPredictor(MC)
    for k in range(9):
        pred = mp.update(person_mask(0.0, 0.6 * (k % 2)), ORIGIN, RES, k / 3)
        assert not pred.any()


def test_corridor_starts_at_blob_front_not_lagging_centroid():
    """The map fades a walker's cells over ~0.5 s, so its blob trails behind it; the corridor is measured
    from the front edge (the person), not the centroid (the middle of the trail)."""
    mp = MotionPredictor(MC)
    for k in range(6):
        y = -6.0 + 0.4 * k
        m = torch.zeros(N, N, dtype=torch.bool)
        m[49:52, round((y - 1.0 - ORIGIN[1]) / RES):round((y - ORIGIN[1]) / RES) + 2] = True   # 1 m trail
        pred = mp.update(m, ORIGIN, RES, k / 3)
    assert cell(pred, 0.0, y + 1.2 * MC["horizon"] + 0.6)      # person + walk + corridor half-width
