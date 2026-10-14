# Project status, last updated 2026-10-14

## ▶ Resume here (new conversation)
These two files (`CLAUDE.md` + this one) are the full context. **Don't re-explore the codebase or
re-run old benchmarks.** Everything below is verified, committed and pushed (working tree clean
after the P5 commit).

To start the next session:
1. Continue with **"Not done", item 1** below, then the remaining items in order.
2. After each item: run `pytest`, write the result JSON to `docs/`, run `python3 tools/make_results.py`,
   update this file, then commit and push. Author PRDXAbe, **no AI attribution**.
3. Running the Nav2 stack: `ros2 launch adaptmap_nav nav.launch.py rviz:=false`; scenarios:
   `.venv/bin/python tools/nav_scenarios.py --repeat 3` (about 13 min); stop with `bash tools/stop_demo.sh`.

Plan: phases P1–P9 (completion plan, approved 2026-10-12). Every phase is committed and pushed
separately.

## Checklist

### Done ✅
| Phase | What | Key result |
|---|---|---|
| P0 | 1-day prototype: sim world + rover, pipeline, foveated map, ROS node, docs | `ceeb782` |
| P1 | Performance: fp16 MinkUNet, async publishing, topic parameters | segmentation 65 → 28.5 ms; offline 37.5 ms/scan (27 Hz); mapping core live 9.3 scans/s |
| P2 | Map quality: ray clearing, ground surface, height above ground, clearance under overhangs, ring-gap hole filling | 6 tests |
| P3 | Terrain: slope, step, roughness, curb detection, traversability costmap (`/adaptmap/costmap`) | 5 tests; viewer "Traversability" colouring |
| P4 | Fine-tune for AdaptMap's 10 classes: seeded worlds, live auto-labelling (ground-contact fix), 10-class head, distillation | held-out demo world mIoU 51.4 → **90.2**; world 21: 49.0 → **88.4** (corrected labels, see P4 follow-up) |
| P6 | Real data | KITTI drive 0005: full pipeline 52 ms/scan (19 Hz) + video. SemanticPOSS: pretrained 43.6 mIoU; sim model with constant intensity 38.3 (pedestrians 7 → 53 %) |
| P7 | Speed-adaptive zones (`set_zones`, `ZonePolicy`) | 5 cm zone 10 m → 15.5 m @ 5 m/s → 25 m @ 8 m/s (1.33 M cells) |
| P8 (part) | CI (GitHub Actions, CPU pytest), launch_testing smoke test (3/3 pass), `make_results.py`, README/METHOD/TESTING updates | |
| P4 follow-up | "v2 only 46 % on world-21 vehicles" was a **label bug**: `tools/fix_labels.py` relabelled whole car sides as road wherever no ground was visible within 1 m (next to the robot, inside the LiDAR blind disc; `gz = inf` made every point "low"). Fixed with a test, labels regenerated from `labels_raw` for s11–s16 and s21 (≈80 % fewer relabelled points; the rest are at ground level). New `tools/eval_sim_test.py` → `docs/eval_world21.json` (all 245 scans) | world 21: v2 **88.4** mIoU, vehicle **93 %** (was reported 73.8 / 46 %); v1 79.0 / 48 % (it was trained only on the corrupted worlds). v2 is best on both held-out worlds |
| v3 retrain | v2 recipe on the corrected labels (`checkpoints/minkunet_adapt10_v3.pth`, log `data/finetune_model_v3.log`). `tools/finetune.py` now starts from `model_real` (it crashed starting from the 10-class default) | world 21 **89.6** (v2 88.4), demo world 90.4 (v2 90.2), but demo pedestrians 78 vs 80 %: **v2 stays the default** (navigation validated with it) |
| Housekeeping | Reviewed `data/demo_*`; five unused runs deleted 2026-10-14 (user OK). Kept `data/demo_20261010_212110/bag` (demo-world evaluation set) and `data/demo_20261012_222528` (source of `docs/live_summary.json`) | |
| P5 | Nav2 on the AdaptMap costmap (SmacPlanner2D + RegulatedPurePursuit), 5 scripted scenarios run 3× each. New method parts: *appearance* (a new obstacle replaces a cell in one scan instead of being averaged in over ~1 s), ground clearing, 1 m pedestrian keep-out, **motion-aware keep-out** (`adaptmap/motion.py`: pedestrian tracks → lethal corridor of the next 2.5 s, measured from the blob's front edge), footprint clearing (map + published costmap never lethal under the rover), costmap every scan | **15/15 runs pass** (`docs/nav_results.json`); 31 tests. Video `docs/figures/nav_demo.mp4` (local, choppy 1.3 fps: sim + renderer share the GPU; re-record with `--top` if needed). Known limitation: the rover sometimes yields *on* the crossing line instead of before it; a Gazebo actor then walks into the stopped rover (footnote in RESULTS) |

### Not done ⬜ (next steps, in order)
1. **Real-data accuracy (optional).** Fine-tune on mixed sim + SemanticPOSS sequences 00–04,
   with intensity dropout/augmentation, testing on 05. This would give one model for both sim and
   real data.
2. **P8 polish:**
   - ~~`docs/TESTING.md` numbers~~: done 2026-10-13 (speed table, "Known mistakes", 5.4× map
     update, v1/v2 wording in `make_results.py`).
   - ~~`docs/METHOD.md`~~: done 2026-10-14 (§3.4 ground clearing / appearance / footprint clearing,
     §3.7 navigation rules, new §3.9 motion-aware keep-out, §4 training pipeline, §5 file map, §6
     current results, §7 limitations, §8 claims 9–16), README intro.
   - ~~Demo videos~~: decided 2026-10-14 to keep `*.mp4` out of git (local only, `docs/figures/`).
3. **P9 (optional).** A more complex robot URDF (e.g. a Husky description, or Go2 + CHAMP). Only
   the extrinsics (`lidar_offset`), topics and footprint should need changing.
## Where things are
| Thing | Location |
|---|---|
| Models | `checkpoints/minkunet.pth` (pretrained, real data), `minkunet_adapt10.pth` = `_v2.pth` (sim default), `_v1.pth` (6 worlds, no distillation), `_v3.pth` (v2 recipe on corrected labels, slightly better mIoU, not default); training logs `data/finetune*.log` |
| Sim training data | `data/sim_adapt/sequences/urban_s{11..16,31..36}` (4,139 scans, 11 GB), test `data/sim_adapt_test/sequences/urban_s21` (245) |
| Real data | `data/real/kitti_raw/2011_09_26/...drive_0005_sync`, `data/real/semanticposs/dataset` |
| Demo-world eval bag | `data/demo_20261010_212110/bag` |
| Result JSONs | `docs/*.json` → `python3 tools/make_results.py` → `docs/RESULTS.md` |
| Plan file | `~/.claude/plans/lorenzini-sih-ppt-pdf-ok-refactored-plum.md` |

## Headline numbers (from docs/RESULTS.md)
- **Map:** 702,400 cells vs 16 M uniform (22.8× fewer), 19× less memory, map update 5.4× faster.
- **Offline pipeline:** 37.5 ms/scan (27 Hz). Live with Gazebo + visualisation: 6.0 scans/s.
  Mapping core alone: 9.3 scans/s (the sensor rate).
- **Segmentation:** fine-tuned v2 vs pretrained: demo world 90.2 vs 51.4 mIoU; test world 21: 88.4 vs
  49.0.
- **Real data:** KITTI 19 Hz end to end; SemanticPOSS 43.6 mIoU (pretrained, unseen 40-beam
  sensor).
- **Navigation:** Nav2 on the AdaptMap costmap, 15/15 scenario runs (crossing pedestrian, parked car,
  curb refusal, 55 m route, U-turn).
