# Project status, last updated 2026-10-18

## ▶ Resume here (new conversation)
These two files (`CLAUDE.md` + this one) are the full context. **Don't re-explore the codebase or
re-run old benchmarks.** Everything below is verified, committed and pushed (working tree clean
after the P5 commit).

To start the next session:
1. Continue with **"Not done", item 1** below (the UI), then the optional items.
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
| Stress test | Five test worlds of rising difficulty (`tools/make_tier_worlds.py`, 30 scenarios, geometry self-check) run by `tools/run_tier_suite.sh`. Fixes it forced: **one-sided step rule** (walls grew a lethal 0.2 m fringe: a 1.4 m opening looked 0.9 m), **robot_height 1.85 m** (incl. LiDAR mast; a 1.3 m bar was "passable"), **default model v3** (v2 called a car's side road from beside it → froze in an empty lane), Nav2 global window 260 m (a goal outside it came back as a false *success*). 11.6 min soak: no leak, no slowdown | **78/90** (tiers 15/15, 13/15, 16/18, 24/24, 10/18), **0 runs drove into a pedestrian**; demo world 14/15 with v3. Failures: stuck after yielding in a tight corridor (6), yields on the crossing line (2+1 demo), touched a 25 cm crate (2) / parked car (1) in the narrow street, over-cautious with a runner (1) |
| P5 | Nav2 on the AdaptMap costmap (SmacPlanner2D + RegulatedPurePursuit), 5 scripted scenarios run 3× each. New method parts: *appearance* (a new obstacle replaces a cell in one scan instead of being averaged in over ~1 s), ground clearing, 1 m pedestrian keep-out, **motion-aware keep-out** (`adaptmap/motion.py`: pedestrian tracks → lethal corridor of the next 2.5 s, measured from the blob's front edge), footprint clearing (map + published costmap never lethal under the rover), costmap every scan | **15/15 runs pass** (`docs/nav_results.json`); 31 tests. Video `docs/figures/nav_demo.mp4` (local, choppy 1.3 fps: sim + renderer share the GPU; re-record with `--top` if needed). Known limitation: the rover sometimes yields *on* the crossing line instead of before it; a Gazebo actor then walks into the stopped rover (footnote in RESULTS) |

### Not done ⬜ (next steps, in order)
1. **Judge-facing UI** (user's next request): one clear dashboard that tells the story: the "why"
   (702 k vs 16 M cells, tile zones), "what it sees" (clean top-down map, predicted pedestrian paths),
   "it works" (robot driving with plain captions, pass counts from RESULTS), scenario buttons.
2. **Optional navigation improvements** found by the stress test (all Nav2/behaviour, not the map):
   move off again after yielding in tight corridors (RPP collision look-ahead / recovery tuning);
   stop *before* a crossing line; less conservative corridor for fast runners; check whether the
   25 cm crate contact is the planner or the crate fading inside the blind disc.
3. **Real-data accuracy (optional).** Fine-tune on mixed sim + SemanticPOSS sequences 00–04 with
   intensity dropout, testing on 05: one model for sim and real data.
4. **P9 (optional).** A bigger robot (Husky / Go2): extrinsics (`lidar_offset`), topics, footprint,
   `terrain.robot_height`.

## Where things are
| Thing | Location |
|---|---|
| Models | `checkpoints/minkunet.pth` (pretrained, real data); `minkunet_adapt10_v1.pth` (6 worlds), `_v2.pth` (12 worlds + distillation), `_v3.pth` (v2 recipe on corrected labels; **config default**), `minkunet_adapt10.pth` = copy of v2. Training logs `data/finetune_model_v{1,2,3}.log`, histories `checkpoints/minkunet_adapt10_v*.json` |
| Sim training data | `data/sim_adapt/sequences/urban_s{11..16,31..36}` (4,139 scans, 11 GB), test `data/sim_adapt_test/sequences/urban_s21` (245) |
| Real data | `data/real/kitti_raw/2011_09_26/...drive_0005_sync`, `data/real/semanticposs/dataset` |
| Demo-world eval bag | `data/demo_20261010_212110/bag` |
| Other local data | `data/demo_20261012_222528` (source of `docs/live_summary.json`), `data/coreonly` (mapping core alone, 9.3 scans/s), `data/live` (demo videos); `data/tiers_<date>/` = logs of `tools/run_tier_suite.sh` |
| Result JSONs | `docs/*.json` → `python3 tools/make_results.py` → `docs/RESULTS.md` |
| Plan file | `~/.claude/plans/lorenzini-sih-ppt-pdf-ok-refactored-plum.md` |

## Headline numbers (from docs/RESULTS.md)
- **Map:** 702,400 cells vs 16 M uniform (22.8× fewer), 19× less memory, map update 5.4× faster.
- **Offline pipeline:** 37.5 ms/scan (27 Hz). Live with Gazebo + visualisation: 6.0 scans/s.
  Mapping core alone: 9.3 scans/s (the sensor rate).
- **Segmentation:** fine-tuned v3 (default) vs pretrained: demo world 90.4 vs 51.4 mIoU; test world 21:
  89.6 vs 49.0.
- **Real data:** KITTI 19 Hz end to end; SemanticPOSS 43.6 mIoU (pretrained, unseen 40-beam
  sensor).
- **Navigation:** demo world 14/15; stress test over five worlds of rising difficulty 78/90, never
  driving into a pedestrian. 11.6 min continuous run: no memory growth or slowdown.
