# Project status, last updated 2026-11-08

## ▶ Resume here (new conversation)
These two files (`CLAUDE.md` + this one) are the full context. **Don't re-explore the codebase or
re-run old benchmarks.** Everything below is verified, committed and pushed.

To start the next session:
1. Continue with **"Not done"** below (all optional).
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
| v3 retrain | v2 recipe on the corrected labels (`checkpoints/minkunet_adapt10_v3.pth`, log `data/finetune_model_v3.log`). `tools/finetune.py` now starts from `model_real` (it crashed starting from the 10-class default) | world 21 **89.6** (v2 88.4), demo world 90.4 (v2 90.2), but demo pedestrians 78 vs 80 %: v2 stayed the default at first; the stress test then made **v3 the default** (see below) |
| Housekeeping | Reviewed `data/demo_*`; five unused runs deleted 2026-10-14 (user OK). Kept `data/demo_20261010_212110/bag` (demo-world evaluation set) and `data/demo_20261012_222528` (source of `docs/live_summary.json`) | |
| Stress test | Five test worlds of rising difficulty (`tools/make_tier_worlds.py`, 30 scenarios, geometry self-check) run by `tools/run_tier_suite.sh`. Fixes it forced: **one-sided step rule** (walls grew a lethal 0.2 m fringe: a 1.4 m opening looked 0.9 m), **robot_height 1.85 m** (incl. LiDAR mast; a 1.3 m bar was "passable"), **default model v3** (v2 called a car's side road from beside it → froze in an empty lane), Nav2 global window 260 m (a goal outside it came back as a false *success*). 11.6 min soak: no leak, no slowdown | **78/90** (tiers 15/15, 13/15, 16/18, 24/24, 10/18), **0 runs drove into a pedestrian**; demo world 14/15 with v3. Failures: stuck after yielding in a tight corridor (6), yields on the crossing line (2+1 demo), touched a 25 cm crate (2) / parked car (1) in the narrow street, over-cautious with a runner (1) |
| P5 | Nav2 on the AdaptMap costmap (SmacPlanner2D + RegulatedPurePursuit), 5 scripted scenarios run 3× each. New method parts: *appearance* (a new obstacle replaces a cell in one scan instead of being averaged in over ~1 s), ground clearing, 1 m pedestrian keep-out, **motion-aware keep-out** (`adaptmap/motion.py`: pedestrian tracks → lethal corridor of the next 2.5 s, measured from the blob's front edge), footprint clearing (map + published costmap never lethal under the rover), costmap every scan | **15/15 runs pass** (`docs/nav_results.json`); 31 tests. Video `docs/figures/nav_demo.mp4` (local, choppy 1.3 fps: sim + renderer share the GPU; re-record with `--top` if needed). Known limitation: the rover sometimes yields *on* the crossing line instead of before it; a Gazebo actor then walks into the stopped rover (footnote in RESULTS) |
| Judge UI | `bash tools/dashboard.sh [tierN]` → local page (`tools/dashboard.py` + `.html`, Flask, no CDN): why (cells, zones, adaptive zones from `docs/*.json`), live top-down map (semantic / planner view, Nav2 path, pedestrians + predicted 2.5 s corridor), scenario buttons with live captions and a Stop button, measured results; `--offline` shows recorded video + results. `nav_scenarios.Runner.run` got a `progress` callback and an `abort` flag | Verified live in the demo world: pedestrian_crossing and around_lane_car passed from the page, curb goal correctly refused, Stop works |
| Nav improvements | From the stress test (`8b39a50`, `8afa3e3`): people are timing constraints for the planner (`/adaptmap/costmap_plan`: keep-outs costly, not lethal; robot_radius 0.40 so no detours through curb gaps), controller `failure_tolerance` 20 s (wait out a crossing group), keep-outs seeded only from person cells seen within 0.3 s (a walker's fading trail blocks nothing), decay at the expected rate under the class probabilities, the driven trail counts as known ground (recoveries no longer abort on the blind disc), corridor radius 1.0 m, planner margin around the footprint only for cells not seen for 1 s (without that gate the rover crept into a parked car in 1 of 3 curb-refusal runs; after: 3/3, 1.4 m clear). tier5 crate start moved out of the blind disc; `make_tier_worlds.py` checks it | Full eval *before* the 1 s gate: tiers **78/90** (15, 15, 16, 22, 10 of 15, 15, 18, 24, 18), demo world 14/15. Parked street 13 → 15/15; speed bump 3 → 1/3 (stalls with its front against the bump in every run old and new; physics). After the gate: curb refusal 3/3, tier 5 8/12 (was 10/18). 39 tests |
| P9 | Robot profiles `configs/robots/{rover,ugv}.yaml` (extrinsics, topics, self box, footprint, height, Nav2 footprint) selected by `robot:=` / `--robot` / `ROBOT=ugv bash tools/dashboard.sh`; 1.40 × 0.95 m 4-wheel skid-steer UGV (`ros2_ws/src/adaptmap_sim/robots/ugv/rover/`), LiDAR 1.30 m up, 0.35 m ahead; scans shifted to the model's 1.73 m for segmentation only (`85875c9`) | UGV, 1 run each, no retraining: tier 1 5/5, demo world 5/5, tier 4 7/8 (stops before the 1.4 m opening without contact: 0.2 m costmap cells eat it; radius 0.50 didn't help) → `docs/nav_*_ugv.json`, RESULTS section |
| v4 mixed fine-tune (trained, not evaluated) | `tools/finetune.py --real data/real/semanticposs/dataset --iters 12000 --z-jitter 0.25` (sim + SemanticPOSS 00–04, test 05, intensity dropout 0.5, set-valued ground loss) → `checkpoints/minkunet_adapt10_v4.pth`, log `data/finetune_model_v4.log` (82 min) | Best (iter 10000; in-training eval, 60 scans, constant intensity): sim world 21 **79.9** (v3 89.6 on all scans; vegetation only 19 %), SemanticPOSS 05 **58.3** (7 merged classes; same protocol at step 0: 42.0). Later steps overfit the sim score (74.5 at 12000) |
| v4 evaluated (2026-11-08) | `eval_sim_test.py --only v4`, `eval_bag.py`, `eval_semanticposs.py --seqs 05 --every 2 --out semanticposs_seq05.json` (new options: `--only`, `--seqs`, `--out`) | world 21 **81.0** (v3 89.6, vegetation 19 %), demo world **84.6** (v3 90.4). SemanticPOSS held-out seq 05: pretrained 46.0, v3 + const 43.2, **v4 + const 56.0** (pedestrians 74 vs 8 %), v4 with real intensity **21.2** (calls ~96 % of ground vehicle, on training sequences too; cause unknown, in-training eval only used constant intensity). **Decision: nothing changes** (v3 sim default, pretrained `model_real`). METHOD table + §7 updated |
| Full nav rerun (2026-11-08) | `data/navdbg/full_eval.sh` on the final code (v3, all nav improvements), log `data/navdbg/full_eval_20261108.log`, `data/tiers_20261108_090756/` | Tiers **86/90** (15, 15, 18, 24, 14 of 15, 15, 18, 24, 18; was 78/90), **0 runs drove into a pedestrian**; speed bump now 3/3. Demo world 14/15. Remaining failures: tier 5 `group_crossing` **0/3** (+1 `full_run`): stops at x ≈ 59 *on* the crossing band (59–62), the group walks into it, never moves off before the 91 s limit; demo `sidewalk_goal_refused` 1/3 touched parked car `car_9` at 1.2 m/s (the 1 s gate made it rarer, not gone). METHOD §6/§7 rewritten |

### Not done ⬜ (next steps, in order)
1. **Group crossing (tier 5, 0/3):** make the robot yield *before* the crossing band and move off once
   the group has passed (e.g. a keep-out covering the group's whole crossing path, or a stop line at
   the corridor's near edge). Debug: `bash data/navdbg/try.sh tier5 3 group_crossing`.
2. **Curb refusal touches a parked car (demo world, 1/3):** check the run-2 track against the costmap.
3. Optional: UGV runs ×3 on all tiers; a 0.1 m costmap near the robot for the UGV's 1.4 m opening.
4. Optional: fix v4's real-intensity failure (e.g. evaluate real data at both intensities during training,
   lower `--mix`/fewer iterations for sim vegetation), then reconsider v4 + constant intensity as `model_real`.

## Where things are
| Thing | Location |
|---|---|
| Models | `checkpoints/minkunet.pth` (pretrained, real data); `minkunet_adapt10_v1.pth` (6 worlds), `_v2.pth` (12 worlds + distillation), `_v3.pth` (v2 recipe on corrected labels; **config default**), `_v4.pth` (mixed sim + SemanticPOSS; evaluated, not used: fails with real intensity), `minkunet_adapt10.pth` = copy of v2. Training logs `data/finetune_model_v{1,2,3,4}.log`, histories `checkpoints/minkunet_adapt10_v*.json` |
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
- **Navigation:** demo world 14/15; stress test over five worlds of rising difficulty 86/90, never
  driving into a pedestrian. Second robot (1.4 m UGV, no retraining): 17/18 runs. 11.6 min continuous run: no memory growth or slowdown.
