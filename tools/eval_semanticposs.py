"""Real-data segmentation accuracy on SemanticPOSS (Pandora 40-beam LiDAR, Peking University campus).

SemanticPOSS labels ground as one class, so AdaptMap's road / sidewalk / terrain are merged into
'ground' for scoring. Runs the same preprocessing + segmenter as the live pipeline, keeping the real
intensity (scaled 0..1). Appends a section to docs/RESULTS.md unless --no-write.

Usage: .venv/bin/python tools/eval_semanticposs.py --root data/real/semanticposs/dataset [--every 5]
          [--checkpoint checkpoints/minkunet_adapt10.pth --config checkpoints/minkunet_adapt10.py]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adaptmap import classes  # noqa: E402
from adaptmap.config import load_config  # noqa: E402
from adaptmap.preprocess import preprocess  # noqa: E402
from adaptmap.segmentation import build_segmenter  # noqa: E402

EVAL = ["ground", "vegetation", "building_wall", "pole", "vehicle", "pedestrian", "other_obstacle"]
POSS_TO_EVAL = {4: "pedestrian", 5: "pedestrian", 6: "pedestrian", 7: "vehicle", 21: "vehicle", 8: "pole",
                9: "vegetation", 10: "pole", 11: "pole", 12: "pole", 13: "pole", 14: "other_obstacle",
                15: "building_wall", 16: "other_obstacle", 17: "building_wall", 22: "ground"}
ADAPT_TO_EVAL = {"road": "ground", "sidewalk_curb": "ground", "terrain": "ground", "vegetation": "vegetation",
                 "building_wall": "building_wall", "pole": "pole", "vehicle": "vehicle", "pedestrian": "pedestrian",
                 "other_obstacle": "other_obstacle"}


def luts():
    poss = np.full(256, -1, np.int64)
    for k, v in POSS_TO_EVAL.items():
        poss[k] = EVAL.index(v)
    adapt = torch.full((classes.NUM_CLASSES,), -1, dtype=torch.long)
    for k, v in ADAPT_TO_EVAL.items():
        adapt[classes.NAMES.index(k)] = EVAL.index(v)
    return poss, adapt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "real" / "semanticposs" / "dataset"))
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--checkpoint")
    ap.add_argument("--config")
    ap.add_argument("--name", default="pretrained SemanticKITTI MinkUNet-34")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--const-intensity", type=float, default=None, help="replace intensity (models trained on sim data)")
    a = ap.parse_args()
    cfg = load_config()
    cfg["model"].update(cfg["model_real"])  # real sensors: SemanticKITTI-pretrained model
    if a.checkpoint:
        cfg["model"]["checkpoint"], cfg["model"]["config"] = a.checkpoint, a.config
    cfg["preprocess"]["constant_intensity"] = a.const_intensity  # None = keep real intensity
    seg = build_segmenter(cfg)
    poss_lut, adapt_lut = luts()
    adapt_lut = adapt_lut.cuda()
    E = len(EVAL)
    conf = torch.zeros(E, E, dtype=torch.long, device="cuda")
    n = 0
    for seq in sorted((Path(a.root) / "sequences").glob("*")):
        for b in sorted((seq / "velodyne").glob("*.bin"))[:: a.every]:
            p = np.fromfile(b, np.float32).reshape(-1, 4).copy()
            p[:, 3] /= 255.0
            lab = poss_lut[np.fromfile(seq / "labels" / (b.stem + ".label"), np.uint32) & 0xFFFF]
            pts = torch.from_numpy(np.concatenate([p, lab[:, None].astype(np.float32)], 1)).cuda()
            clean, _ = preprocess(pts[:, :4], cfg["preprocess"], cfg["zones"])
            # recover each kept point's label by matching rows (preprocess only filters)
            key = lambda x: (x[:, 0] * 1e4).round().long() * 4_000_003 + (x[:, 1] * 1e4).round().long() * 7919 + (x[:, 2] * 1e4).round().long()  # noqa: E731
            kr, kc = key(pts), key(clean)
            order = torch.argsort(kr)
            gt = pts[order[torch.searchsorted(kr[order], kc).clamp(max=len(kr) - 1)], 4].long()
            pred_a, _ = seg.segment(clean)
            pred = adapt_lut[pred_a]
            k = (gt >= 0) & (pred >= 0)
            conf += torch.bincount(gt[k] * E + pred[k], minlength=E * E).view(E, E)
            n += 1
    tp = conf.diag().float()
    iou = (tp / (conf.sum(0) + conf.sum(1) - tp).clamp(min=1).float()).cpu().numpy()
    acc = float(tp.sum() / conf.sum())
    miou = float(iou.mean())
    res = {"model": a.name, "scans": n, "accuracy": acc, "miou": miou, "iou": {EVAL[i]: float(iou[i]) for i in range(E)}}
    print(json.dumps(res))
    if not a.no_write:
        out = ROOT / "docs" / "semanticposs_results.json"
        allres = json.loads(out.read_text()) if out.exists() else []
        allres = [r for r in allres if r["model"] != a.name] + [res]
        out.write_text(json.dumps(allres, indent=1))
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
