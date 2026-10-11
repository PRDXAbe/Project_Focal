#!/usr/bin/env bash
# Manual test: Gazebo + AdaptMap with NO scripted driving, plus the 3D "how the robot sees it" viewer.
# Drive with teleop_twist_keyboard in a second terminal (see docs/TESTING.md).
#
# Usage: bash tools/manual_test.sh            -> interactive 3D window
#        bash tools/manual_test.sh --video F  -> record a side-by-side explainer video to F while you drive
set -o pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /opt/ros/humble/setup.bash
source "$ROOT/ros2_ws/install/setup.bash"
export ADAPTMAP_ROOT="$ROOT" QT_QPA_PLATFORM=xcb
LOG="$ROOT/data/manual_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG"

ros2 launch adaptmap_ros adaptmap.launch.py drive:=false stats_log:="$LOG/stats.jsonl" > "$LOG/launch.log" 2>&1 &
echo "starting Gazebo + AdaptMap (logs: $LOG) ..."
for i in $(seq 1 90); do [ -s "$LOG/stats.jsonl" ] && break; sleep 1; done
[ -s "$LOG/stats.jsonl" ] || { echo "AdaptMap produced no output; see $LOG/launch.log"; bash "$ROOT/tools/stop_demo.sh"; exit 1; }
echo
echo "Ready. In ANOTHER terminal, drive the robot with:"
echo "  source /opt/ros/humble/setup.bash && ros2 run teleop_twist_keyboard teleop_twist_keyboard"
echo "  (i = forward, , = back, j/l = turn, k = stop, q/z = faster/slower)"
echo
"$ROOT/.venv/bin/python" -u "$ROOT/tools/view25d.py" "$@"
echo "viewer closed - shutting down the sim"
bash "$ROOT/tools/stop_demo.sh" > /dev/null
