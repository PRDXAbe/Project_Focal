"""tools/fix_labels.py: ground-contact relabelling must not touch objects with no ground nearby."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fix_labels import fix  # noqa: E402

ROAD, VEHICLE = 1, 7


def test_fix_labels_only_relabels_points_at_ground_level(tmp_path):
    """Road visible from 4 m out (inside that, the LiDAR's blind disc); a car beside the robot at 1-2 m
    (no ground within 1 m of it), and one at 6 m whose box-labelled bottom points are really road."""
    g = np.stack(np.meshgrid(np.arange(-8, 8, 0.2), np.arange(-8, 8, 0.2)), -1).reshape(-1, 2)
    g = g[np.linalg.norm(g, axis=1) > 4.0]
    road = np.c_[g, np.full(len(g), -1.73)]
    near = np.c_[np.random.uniform(-1, 1, (300, 1)), np.random.uniform(1.0, 2.0, (300, 1)), np.random.uniform(-1.4, -0.2, (300, 1))]
    far = np.c_[np.random.uniform(5.5, 7.5, (300, 1)), np.random.uniform(-1, 1, (300, 1)), np.random.uniform(-1.4, -0.2, (300, 1))]
    feet = np.c_[np.random.uniform(5.5, 7.5, (50, 1)), np.random.uniform(-1, 1, (50, 1)), np.full((50, 1), -1.70)]
    pts = np.r_[road, near, far, feet].astype(np.float32)
    lab = np.r_[np.full(len(road), ROAD), np.full(650, VEHICLE)].astype(np.uint32)
    seq = tmp_path / "seq"
    (seq / "velodyne").mkdir(parents=True)
    (seq / "labels").mkdir()
    np.c_[pts, np.zeros(len(pts), np.float32)].astype(np.float32).tofile(seq / "velodyne" / "000000.bin")
    lab.tofile(seq / "labels" / "000000.label")
    new, n = fix(seq / "labels" / "000000.label")
    o = len(road)
    assert (new[o:o + 300] == VEHICLE).all()          # car beside the robot: untouched (was relabelled road)
    assert (new[o + 300:o + 600] == VEHICLE).all()    # car body above the ground: untouched
    assert (new[o + 600:] == ROAD).all() and n == 50  # its ground-level points: road
