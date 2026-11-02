#!/usr/bin/env bash
# Judge demo: Gazebo + AdaptMap + Nav2 (no RViz) and the dashboard page (tools/dashboard.py) in the browser.
#
# Usage: bash tools/dashboard.sh                 -> demo world
#        bash tools/dashboard.sh tier3           -> one of the stress-test worlds (tools/make_tier_worlds.py)
#        bash tools/dashboard.sh --offline       -> recorded results and video only, no simulation
# Ctrl-C stops the dashboard and the simulation.
set -o pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORLD="${1:-urban}"
PORT="${PORT:-8080}"
URL="http://localhost:$PORT"
open_browser() { (sleep 3; xdg-open "$URL" > /dev/null 2>&1) & }

if [ "$WORLD" = "--offline" ]; then
    open_browser
    exec "$ROOT/.venv/bin/python" "$ROOT/tools/dashboard.py" --offline --port "$PORT"
fi

source /opt/ros/humble/setup.bash
source "$ROOT/ros2_ws/install/setup.bash"
export ADAPTMAP_ROOT="$ROOT" QT_QPA_PLATFORM=xcb
# same as tools/manual_test.sh: Gazebo on the NVIDIA GPU, CPU math libraries capped at 2 threads
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
LOG="$ROOT/data/dashboard_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG"
trap 'echo "stopping the simulation ..."; bash "$ROOT/tools/stop_demo.sh" > /dev/null 2>&1' EXIT

up=0
for attempt in 1 2 3; do  # Nav2's lifecycle bring-up occasionally stalls (see tools/run_tier_suite.sh)
    bash "$ROOT/tools/stop_demo.sh" > /dev/null 2>&1; sleep 3
    echo "starting Gazebo + AdaptMap + Nav2 ($WORLD, logs: $LOG) ..."
    ros2 launch adaptmap_nav nav.launch.py world:="$WORLD.sdf" rviz:=false > "$LOG/launch.log" 2>&1 &
    for i in $(seq 1 120); do grep -q "Managed nodes are active" "$LOG/launch.log" && break; sleep 1; done
    if grep -q "Managed nodes are active" "$LOG/launch.log"; then up=1; break; fi
    echo "Nav2 did not come up (attempt $attempt), relaunching"
done
[ "$up" = 1 ] || { echo "Nav2 did not come up; see $LOG/launch.log"; exit 1; }

echo "ready: $URL"
open_browser
"$ROOT/.venv/bin/python" "$ROOT/tools/dashboard.py" --world "$WORLD" --port "$PORT" 2> "$LOG/dashboard.log"
