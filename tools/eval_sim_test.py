"""Evaluate the pretrained, v1 and v2 models on the held-out simulation world (data/sim_adapt_test, seed 21).

Same metric as tools/finetune.py's in-training evaluation (per-class IoU over labelled points, 50 m crop),
but over every test scan. Run after relabelling (tools/fix_labels.py) so the table reflects the current labels.

Usage: .venv/bin/python tools/eval_sim_test.py [--out docs/eval_world21.json]
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from adaptmap.config import load_config  # noqa: E402
from finetune import build_model, evaluate, scans  # noqa: E402

MODELS = [  # (row name, config, checkpoint, 10-class head already trained)
    ("pretrained (KITTI head mapped)", "checkpoints/minkunet.py", "checkpoints/minkunet.pth", False),
    ("fine-tuned v1: 6 worlds", "checkpoints/minkunet_adapt10.py", "checkpoints/minkunet_adapt10_v1.pth", True),
    ("fine-tuned v2: 12 worlds + distillation", "checkpoints/minkunet_adapt10.py",
     "checkpoints/minkunet_adapt10_v2.pth", True),
    ("fine-tuned v3: v2 recipe on corrected labels, default", "checkpoints/minkunet_adapt10_v3.py",
     "checkpoints/minkunet_adapt10_v3.pth", True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(ROOT / "data" / "sim_adapt_test"))
    ap.add_argument("--crop", type=float, default=50.0)
    ap.add_argument("--out", default=str(ROOT / "docs" / "eval_world21.json"))
    a = ap.parse_args()
    pairs = scans(a.test)
    rows = []
    for name, config, ckpt, trained in MODELS:
        cfg = copy.deepcopy(load_config())
        cfg["model"]["config"], cfg["model"]["checkpoint"] = str(ROOT / config), str(ROOT / ckpt)
        if trained:
            from mmdet3d.apis import init_model
            model = init_model(cfg["model"]["config"], cfg["model"]["checkpoint"], device="cuda:0")
        else:
            model = build_model(cfg)
        miou, iou = evaluate(model, pairs, a.crop, len(pairs))
        rows.append({"model": name, "miou": round(miou, 4), "iou": {k: round(v / 100, 4) for k, v in iou.items()},
                     "scans": len(pairs)})
        print(f"{name}: mIoU {100 * miou:.1f}  {iou}", flush=True)
        del model
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
