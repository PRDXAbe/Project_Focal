"""Fine-tune MinkUNet on auto-labelled simulation scans with an AdaptMap 10-class head.

Data: tools/record_labeled.py output, <data>/sequences/<world>/{velodyne/*.bin, labels/*.label}
(labels are AdaptMap class ids, 0 = unlabelled and ignored).

The SemanticKITTI 19-class classifier is replaced by a 10-class one whose rows start from each
AdaptMap class's main KITTI class (road<-road, vehicle<-car, pedestrian<-person, ...), so training
starts from the pretrained behaviour; `other_obstacle` (no KITTI equivalent) starts from building/
fence with a lower bias, and `unknown` is never predicted. Loss: class-balanced cross-entropy, plus
(--distill) a distillation term that keeps the student close to the frozen pretrained model wherever
that teacher is confident and doesn't contradict the label - so knowledge the sim worlds don't
cover (e.g. a pickup's flat bed is still a vehicle) isn't forgotten.

Usage:
  .venv/bin/python tools/finetune.py --data data/sim_adapt --test data/sim_adapt_test --iters 6000
Writes checkpoints/minkunet_adapt10.pth (best on the test world) + checkpoints/minkunet_adapt10.py
(inference config); set model.checkpoint/config in configs/adaptmap.yaml to use it.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adaptmap import classes  # noqa: E402
from adaptmap.config import load_config  # noqa: E402

C = classes.NUM_CLASSES
# AdaptMap class -> SemanticKITTI train-id rows used to initialise its classifier row
INIT_ROWS = {"road": [8], "sidewalk_curb": [10], "terrain": [16], "vegetation": [14], "building_wall": [12],
             "pole": [17], "vehicle": [0], "pedestrian": [5], "other_obstacle": [12, 13]}


def scans(root):
    out = []
    for seq in sorted((Path(root) / "sequences").glob("*")):
        for b in sorted((seq / "velodyne").glob("*.bin")):
            out.append((b, seq / "labels" / (b.stem + ".label")))
    return out


def load(pair, crop):
    p = np.fromfile(pair[0], np.float32).reshape(-1, 4)
    lab = np.fromfile(pair[1], np.uint32).astype(np.int64)
    keep = np.linalg.norm(p[:, :2], axis=1) < crop
    return p[keep], lab[keep]


def augment(p):
    p = p.copy()
    a = random.uniform(-np.pi, np.pi)
    c, s = np.cos(a), np.sin(a)
    p[:, :2] = p[:, :2] @ np.array([[c, -s], [s, c]], np.float32).T
    if random.random() < 0.5:
        p[:, 1] = -p[:, 1]
    p[:, :3] *= random.uniform(0.95, 1.05)
    p[:, :3] += np.random.normal(0, 0.01, (len(p), 3)).astype(np.float32)
    return p


def build_model(cfg, device="cuda:0"):
    from mmdet3d.apis import init_model
    model = init_model(cfg["model"]["config"], cfg["model"]["checkpoint"], device=device)
    head = model.decode_head
    old = head.conv_seg
    new = torch.nn.Linear(old.in_features, C).to(old.weight.device)
    with torch.no_grad():
        new.weight.zero_()
        new.bias.fill_(-10.0)  # 'unknown' stays off
        for name, rows in INIT_ROWS.items():
            i = classes.NAMES.index(name)
            new.weight[i] = old.weight[rows].mean(0)
            new.bias[i] = old.bias[rows].mean() - (1.0 if name == "other_obstacle" else 0.0)
    head.conv_seg, head.num_classes, head.ignore_index = new, C, 0
    return model


def sample(points, labels=None):
    from mmdet3d.structures import Det3DDataSample, PointData
    s = Det3DDataSample()
    s.gt_pts_seg = PointData(pts_semantic_mask=labels) if labels is not None else PointData()
    return s


@torch.no_grad()
def evaluate(model, pairs, crop, limit):
    model.eval()
    conf = torch.zeros(C, C, dtype=torch.long, device="cuda")
    for pair in pairs[:: max(1, len(pairs) // limit)]:
        p, lab = load(pair, crop)
        pts = torch.from_numpy(p).cuda()
        d = model.data_preprocessor({"inputs": {"points": [pts]}, "data_samples": [sample(pts)]}, False)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model.decode_head.predict(model.extract_feat(d["inputs"]), d["data_samples"])[0]
        pred, gt = logits.argmax(1), torch.from_numpy(lab).cuda()
        k = gt > 0
        conf += torch.bincount(gt[k] * C + pred[k], minlength=C * C).view(C, C)
    model.train()
    tp = conf.diag().float()
    iou = tp / (conf.sum(0) + conf.sum(1) - tp).clamp(min=1).float()
    present = conf.sum(1) > 0
    present[0] = False
    return float(iou[present].mean()), {classes.NAMES[i]: round(100 * float(iou[i]), 1) for i in range(1, C) if present[i]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "sim_adapt"))
    ap.add_argument("--test", default=str(ROOT / "data" / "sim_adapt_test"))
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop", type=float, default=50.0, help="drop points beyond this range (VRAM)")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-scans", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "minkunet_adapt10.pth"))
    ap.add_argument("--distill", type=float, default=1.0, help="weight of the teacher-distillation term (0 = off)")
    ap.add_argument("--distill-conf", type=float, default=0.8)
    a = ap.parse_args()
    cfg = load_config()
    # start (and distil) from the SemanticKITTI-pretrained network, not the config's default model, which is a
    # fine-tuned 10-class one since v2 (its head has no KITTI rows to initialise from)
    if "model_real" in cfg:
        cfg["model"] = {**cfg["model"], **cfg["model_real"]}
    train, test = scans(a.data), scans(a.test)
    assert train and test, "no scans found - run tools/gen_training_data.sh first"

    # class-balanced weights from label frequencies (inverse sqrt, normalised; unlabelled ignored)
    freq = np.zeros(C)
    for pair in train[:: max(1, len(train) // 200)]:
        freq += np.bincount(np.fromfile(pair[1], np.uint32), minlength=C)[:C]
    w = np.where(freq > 0, 1.0 / np.sqrt(freq / freq.sum() + 1e-6), 0.0)
    w[0] = 0.0
    w = w / w[w > 0].mean()
    weight = torch.tensor(w, dtype=torch.float32, device="cuda")
    print("class weights", {classes.NAMES[i]: round(float(w[i]), 2) for i in range(1, C)})

    model = build_model(cfg)
    model.train()
    teacher = None
    if a.distill > 0:
        from mmdet3d.apis import init_model
        teacher = init_model(cfg["model"]["config"], cfg["model"]["checkpoint"], device="cuda:0")
        teacher.eval()
        for q in teacher.parameters():
            q.requires_grad_(False)
    head_params = list(model.decode_head.conv_seg.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("decode_head.conv_seg")]
    opt = torch.optim.AdamW([{"params": rest, "lr": a.lr * 0.2}, {"params": head_params, "lr": a.lr}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr * 0.2, a.lr], total_steps=a.iters, pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler()

    base_miou, base_iou = evaluate(model, test, a.crop, a.eval_scans)
    print(f"step 0 (initialised from KITTI rows): test mIoU {100*base_miou:.1f} {base_iou}", flush=True)
    best, history, t0 = base_miou, [{"iter": 0, "miou": base_miou, "iou": base_iou}], time.time()
    for it in range(1, a.iters + 1):
        p, lab = load(random.choice(train), a.crop)
        pts = torch.from_numpy(augment(p)).cuda()
        s = sample(pts, torch.from_numpy(lab).cuda())
        d = model.data_preprocessor({"inputs": {"points": [pts]}, "data_samples": [s]}, True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model.decode_head.forward(model.extract_feat(d["inputs"]))
        target = torch.cat([x.gt_pts_seg.voxel_semantic_mask for x in d["data_samples"]]).long()
        loss = F.cross_entropy(logits.float(), target, weight=weight, ignore_index=0)
        kd = torch.zeros((), device="cuda")
        if teacher is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                tl = teacher.decode_head.forward(teacher.extract_feat(d["inputs"]))
            tp = classes.kitti_probs_to_adapt(torch.softmax(tl.float(), 1))
            tconf, targ = tp.max(1)
            m = (tconf > a.distill_conf) & ((target == 0) | (targ == target))
            if m.any():
                ls = torch.log_softmax(logits.float()[m], 1)
                kd = (torch.xlogy(tp[m], tp[m]) - tp[m] * ls).sum(1).mean()
            loss = loss + a.distill * kd
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if it % 50 == 0:
            print(f"iter {it}/{a.iters} loss {loss.item():.3f} kd {kd.item():.3f} ({(time.time() - t0) / it:.2f} s/it)", flush=True)
        if it % a.eval_every == 0 or it == a.iters:
            miou, iou = evaluate(model, test, a.crop, a.eval_scans)
            history.append({"iter": it, "miou": miou, "iou": iou})
            print(f"iter {it}: test mIoU {100*miou:.1f} {iou}", flush=True)
            if miou > best:
                best = miou
                torch.save({"state_dict": model.state_dict(), "meta": {"iter": it, "test_miou": miou, "classes": classes.NAMES}}, a.out)
                print(f"  saved best -> {a.out}", flush=True)

    # inference config: same network, 10 AdaptMap classes
    from mmengine import Config
    mc = Config.fromfile(cfg["model"]["config"])
    mc.model.decode_head.num_classes = C
    mc.model.decode_head.ignore_index = 0
    mc.dump(str(Path(a.out).with_suffix(".py")))
    Path(a.out).with_suffix(".json").write_text(json.dumps({"best_test_miou": best, "history": history}, indent=1))
    print(f"done. best test mIoU {100*best:.1f} (start {100*base_miou:.1f})")


if __name__ == "__main__":
    main()
