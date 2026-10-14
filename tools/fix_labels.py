"""Fix ground-contact label noise in already-recorded auto-labelled scans (made before autolabel's
ground-contact rule): object-labelled points within `lift` of the local ground (lowest ground-labelled
point within 1 m) take the class of the nearest ground-labelled point; points with no ground-labelled
point within 1 m are left alone. Originals go to labels_raw/, and labels/ is always regenerated from them.

Usage: .venv/bin/python tools/fix_labels.py data/sim_adapt data/sim_adapt_test
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

GROUND = np.array([1, 2, 3])  # road, sidewalk_curb, terrain


def fix(lab_file, lift=0.10, radius=1.0):
    lab = np.fromfile(lab_file, np.uint32)
    p = np.fromfile(Path(lab_file).parent.parent / "velodyne" / (Path(lab_file).stem + ".bin"), np.float32).reshape(-1, 4)
    g = np.isin(lab, GROUND)
    obj = (lab > 0) & ~g
    if g.sum() == 0 or obj.sum() == 0:
        return lab, 0
    gp, gl = p[g], lab[g]
    tree = cKDTree(gp[:, :2])
    dist, nn = tree.query(p[obj, :2], k=16, distance_upper_bound=radius)
    valid = np.isfinite(dist)
    nn_c = np.where(valid, nn, 0)
    gz = np.where(valid, gp[nn_c, 2], np.inf).min(1)
    # no ground within `radius` (e.g. next to the robot, inside the LiDAR's blind disc) means no evidence:
    # leave the point alone (gz = inf made every such point "low" and relabelled whole car sides as road)
    low = np.isfinite(gz) & (p[obj, 2] < gz + lift)
    new = lab.copy()
    idx = np.flatnonzero(obj)[low]
    new[idx] = gl[nn_c[low, 0]]
    return new, int(low.sum())


def main():
    for root in sys.argv[1:]:
        for seq in sorted((Path(root) / "sequences").glob("*")):
            raw = seq / "labels_raw"
            if not raw.exists():
                shutil.copytree(seq / "labels", raw)
            changed = total = 0
            for f in sorted(raw.glob("*.label")):
                new, n = fix(f)
                new.tofile(seq / "labels" / f.name)
                changed += n
                total += len(new)
            print(f"{seq.name}: relabelled {changed:,} of {total:,} points ({100 * changed / max(total, 1):.2f} %)")


if __name__ == "__main__":
    main()
