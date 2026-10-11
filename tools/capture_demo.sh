#!/usr/bin/env bash
# Full demo capture without screen recording (works on Wayland):
# sim + AdaptMap + scripted lap, rendered map video, stats log, and a short raw-data rosbag.
# Usage: bash tools/capture_demo.sh [lap_seconds=100] [bag_seconds=40, 0 = no bag]
set -o pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAP=${1:-100}; BAGS=${2:-40}
OUT="$ROOT/data/demo_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"
source /opt/ros/humble/setup.bash
source "$ROOT/ros2_ws/install/setup.bash"
export ADAPTMAP_ROOT="$ROOT" QT_QPA_PLATFORM=xcb

ros2 launch adaptmap_ros adaptmap.launch.py stats_log:="$OUT/stats.jsonl" > "$OUT/launch.log" 2>&1 &
for i in $(seq 1 90); do [ -s "$OUT/stats.jsonl" ] && break; sleep 1; done
[ -s "$OUT/stats.jsonl" ] || { echo "AdaptMap produced no output; see $OUT/launch.log"; bash "$ROOT/tools/stop_demo.sh"; exit 1; }
if [ "$BAGS" -gt 0 ]; then
  timeout -s INT "$BAGS" ros2 bag record -s mcap -o "$OUT/bag" /lidar/points /odom /clock > "$OUT/bag.log" 2>&1 & BAGPID=$!
fi
"$ROOT/.venv/bin/python" "$ROOT/tools/render_live.py" --seconds "$LAP" --out "$OUT/demo.mp4" > "$OUT/render.log" 2>&1
[ -n "$BAGPID" ] && wait "$BAGPID" 2>/dev/null
bash "$ROOT/tools/stop_demo.sh" > /dev/null
echo "OUT=$OUT"; tail -1 "$OUT/render.log"; du -sh "$OUT/bag" 2>/dev/null
