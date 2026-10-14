# AdaptMap — Foveated 2.5D Semantic LiDAR Mapping

SIH26053 prototype (team LORENZINI). This is the pipeline:
LiDAR → preprocessing → sparse-CNN semantic segmentation (MinkUNet) → nested variable-resolution
2.5D semantic elevation map → terrain costmap with predicted pedestrian paths → autonomous driving
with Nav2.

The method is described in [docs/METHOD.md](docs/METHOD.md) and the measurements are in [docs/RESULTS.md](docs/RESULTS.md).

## Setup (Ubuntu 22.04)

1. Install the system packages (ROS 2 Humble, Gazebo Harmonic, grid_map):
   ```
   sudo bash scripts/install_system.sh
   ```
2. Create the Python env and download the model checkpoint:
   ```
   python3 -m venv .venv
   .venv/bin/pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
   .venv/bin/pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
   .venv/bin/pip install -r requirements.txt
   bash scripts/download_checkpoint.sh
   ```
   The helper scripts (`tools/*.sh`) set `ADAPTMAP_ROOT` themselves. If you run `ros2 launch` directly
   from a clone that isn't at `~/Projects/Project_Focal`, export `ADAPTMAP_ROOT=<path to the clone>` first.
3. Build the ROS workspace:
   ```
   source /opt/ros/humble/setup.bash
   cd ros2_ws && colcon build --symlink-install && cd ..
   ```

## Run

| What | Command |
|---|---|
| Unit tests | `.venv/bin/python -m pytest adaptmap/tests` (also run by GitHub Actions on every push) |
| Offline benchmark (no ROS) | `.venv/bin/python tools/benchmark.py --frames 300` (`--speed 8` exercises adaptive zones) |
| **Manual test (you drive, 3D view)** | `bash tools/manual_test.sh` (+ `--video data/my_drive.mp4 [--color class\|zone\|cost] [--top]`); guide: [docs/TESTING.md](docs/TESTING.md) |
| Live sim demo | `source ros2_ws/install/setup.bash && ros2 launch adaptmap_ros adaptmap.launch.py [world:=urban_s21.sdf]` |
| **Autonomous navigation (Nav2)** | `sudo bash scripts/install_nav2.sh` once, then `ros2 launch adaptmap_nav nav.launch.py` and click "2D Goal Pose" in RViz |
| Navigation scenarios (pass/fail) | with the Nav2 launch running: `.venv/bin/python tools/nav_scenarios.py --repeat 3`, then `python3 tools/make_results.py` |
| Navigation scenarios (pass/fail) | with nav running: `.venv/bin/python tools/nav_scenarios.py` |
| Record demo (video + stats + bag) | `bash tools/capture_demo.sh 100 40` (lap seconds, bag seconds; 0 = no bag) |
| Stop everything | `bash tools/stop_demo.sh` |
| Gazebo accuracy from a bag | `.venv/bin/python tools/eval_bag.py data/demo_*/bag` |
| Live-run summary into RESULTS.md | `python3 tools/summarize_live.py data/demo_*/stats.jsonl` |
| Worlds | `.venv/bin/python tools/make_world.py` (demo) or `--seed N` (randomised training world) |
| Training data (auto-labelled) | `bash tools/gen_training_data.sh "11 12 13 14 15 16" 2` (~2,000 scans, 5 GB) |
| Fine-tune (10-class head) | `.venv/bin/python tools/finetune.py --iters 10000` (≈ 35 min on an RTX 4050) |
| Real data (KITTI raw drive) | `.venv/bin/python tools/run_dataset.py --drive <..._sync dir> --calib <date dir> --out data/real/kitti_0005` |
| Teaching figure (tile sizes) | with the sim running: `.venv/bin/python tools/resolution_figure.py` |

### RViz displays

- **2.5D semantic cells:** class-coloured cells drawn at their elevation, with cell size growing with distance.
- **labeled cloud:** the segmented points.
- **nav grid:** the occupancy grid.
- **grid near/high/mid/far:** the per-zone `grid_map` layers.

The demo video is rendered from the ROS topics by `tools/render_live.py`, not screen-captured. On GNOME
Wayland, screen grabs come out black. To record the RViz window itself, use GNOME's recorder
(Ctrl+Shift+Alt+R).

## Layout
```
adaptmap/        core library (no ROS dependency) + tests
configs/         zones, cell sizes, fusion, model
ros2_ws/src/     adaptmap_sim (Gazebo world, rover), adaptmap_ros (node, launch, rviz)
tools/           world generator, LiDAR ray-caster, benchmark, autolabel, finetune, demo recorder
docs/            METHOD.md, RESULTS.md, figures/
```

## Not in this repo

| What | Why | How to get it |
|---|---|---|
| `checkpoints/minkunet.pth` (149 MB) | over GitHub's 100 MB file limit | `bash scripts/download_checkpoint.sh` |
| `data/` (recordings, rosbags) | generated, gigabytes | `bash tools/capture_demo.sh`, `tools/autolabel.py` |
| `docs/figures/*.mp4` (demo videos) | kept local to keep clones small | `bash tools/capture_demo.sh`, `tools/view25d.py --video ...` |
| `.venv/` | machine-specific | setup step 2 |
