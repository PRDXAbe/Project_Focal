"""Fine-tune the pretrained MinkUNet on auto-labelled simulation scans (SemanticKITTI format).

Usage: python tools/finetune.py --data data/sim_kitti --iters 2000 --out checkpoints/minkunet_sim.pth
Then set model.checkpoint in configs/adaptmap.yaml to the new file.

Keeps the 19-class SemanticKITTI head (so real-data knowledge is retained); AdaptMap classes
without a KITTI equivalent are ignored in the loss.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adaptmap.config import load_config  # noqa: E402

# SemanticKITTI raw id -> train id (standard learning map; 19 = ignore)
RAW_TO_TRAIN = {10: 0, 11: 1, 15: 2, 18: 3, 20: 4, 30: 5, 31: 6, 32: 7, 40: 8, 44: 9, 48: 10, 49: 11,
                50: 12, 51: 13, 70: 14, 71: 15, 72: 16, 80: 17, 81: 18}
LUT = np.full(260, 19, dtype=np.int64)
for k, v in RAW_TO_TRAIN.items():
    LUT[k] = v


def load(data):
    seq = Path(data) / "sequences" / "00"
    files = sorted((seq / "velodyne").glob("*.bin"))
    return [(f, seq / "labels" / (f.stem + ".label")) for f in files]


def augment(p):
    a = random.uniform(-np.pi, np.pi)
    c, s = np.cos(a), np.sin(a)
    p[:, :2] = p[:, :2] @ np.array([[c, -s], [s, c]], np.float32).T
    if random.random() < 0.5:
        p[:, 1] = -p[:, 1]
    p[:, :3] *= random.uniform(0.95, 1.05)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "sim_kitti"))
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=float, default=50.0, help="drop points beyond this range (VRAM)")
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "minkunet_sim.pth"))
    a = ap.parse_args()

    from mmdet3d.apis import init_model
    from mmdet3d.structures import Det3DDataSample, PointData

    cfg = load_config()
    model = init_model(cfg["model"]["config"], cfg["model"]["checkpoint"], device="cuda:0")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    scaler = torch.cuda.amp.GradScaler()
    pairs = load(a.data)
    assert pairs, f"no scans in {a.data}"
    print(f"{len(pairs)} scans; fine-tuning {a.iters} iters")
    for it in range(a.iters):
        fb, fl = random.choice(pairs)
        p = np.fromfile(fb, np.float32).reshape(-1, 4)
        lab = LUT[np.fromfile(fl, np.uint32) & 0xFFFF]
        keep = np.linalg.norm(p[:, :2], axis=1) < a.crop
        p, lab = augment(p[keep].copy()), lab[keep]
        s = Det3DDataSample()
        s.gt_pts_seg = PointData(pts_semantic_mask=torch.from_numpy(lab).cuda())
        data = model.data_preprocessor({"inputs": {"points": [torch.from_numpy(p).cuda()]}, "data_samples": [s]}, True)
        with torch.autocast("cuda", dtype=torch.float16):
            losses = model.loss(data["inputs"], data["data_samples"])
        loss = sum(v for k, v in losses.items() if "loss" in k)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if it % 50 == 0:
            print(f"iter {it}: loss {loss.item():.4f}")
    torch.save({"state_dict": model.state_dict(), "meta": {"source": "adaptmap sim fine-tune"}}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
