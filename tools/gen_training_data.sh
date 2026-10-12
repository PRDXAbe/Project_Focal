#!/usr/bin/env bash
# Generate auto-labelled training/test scans: for each world seed, run the sim (no mapping node, no
# GUIs needed), drive the scripted lap and record labelled scans with tools/record_labeled.py.
# Usage: bash tools/gen_training_data.sh "11 12 13 14 15 16" 2   (seeds, keep every Nth scan)
set -o pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEEDS=${1:-"11 12 13 14 15 16"}; EVERY=${2:-2}; OUT=${3:-$ROOT/data/sim_adapt}
source /opt/ros/humble/setup.bash
source "$ROOT/ros2_ws/install/setup.bash"
export ADAPTMAP_ROOT="$ROOT" QT_QPA_PLATFORM=xcb
for s in $SEEDS; do
  W=urban_s$s
  [ -f "$ROOT/ros2_ws/src/adaptmap_sim/worlds/$W.sdf" ] || "$ROOT/.venv/bin/python" "$ROOT/tools/make_world.py" --seed "$s"
  LOG="$OUT/logs"; mkdir -p "$LOG"
  echo "== world $W"
  ros2 launch adaptmap_sim sim.launch.py world:=$W.sdf > "$LOG/$W.sim.log" 2>&1 &
  # wait for the LiDAR to come up
  for i in $(seq 1 60); do timeout 3 ros2 topic echo --once /lidar/points --field header > /dev/null 2>&1 && break; sleep 1; done
  ros2 run adaptmap_ros waypoint_driver --ros-args -p use_sim_time:=true > "$LOG/$W.drive.log" 2>&1 &
  "$ROOT/.venv/bin/python" "$ROOT/tools/record_labeled.py" --world "$W" --out "$OUT" --every "$EVERY" 2>&1 | grep -v Warn | tail -1
  bash "$ROOT/tools/stop_demo.sh" > /dev/null
  sleep 3
done
du -sh "$OUT"
