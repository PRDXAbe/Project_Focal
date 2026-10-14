#!/usr/bin/env bash
# Run the Nav2 scenario suites of the tier test worlds (tools/make_tier_worlds.py), one world at a time:
# launch sim + AdaptMap + Nav2 headless, run tools/nav_scenarios.py --world tierN, stop. Results go to
# docs/nav_tierN.json (then: python3 tools/make_results.py); logs to data/tiers_<date>/.
#
# Usage: bash tools/run_tier_suite.sh [REPEAT] [TIER ...]      e.g.  bash tools/run_tier_suite.sh 3 1 2 3 4 5
set -o pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPEAT="${1:-1}"; shift || true
TIERS="${*:-1 2 3 4 5}"
source /opt/ros/humble/setup.bash
source "$ROOT/ros2_ws/install/setup.bash"
export ADAPTMAP_ROOT="$ROOT"
LOG="$ROOT/data/tiers_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG"
for n in $TIERS; do
    echo "=== tier$n (repeat $REPEAT) - logs in $LOG"
    up=0
    for attempt in 1 2 3; do  # Nav2's lifecycle bring-up occasionally stalls (e.g. configuring behavior_server)
        bash "$ROOT/tools/stop_demo.sh" > /dev/null 2>&1; sleep 5
        ros2 launch adaptmap_nav nav.launch.py world:="tier$n.sdf" rviz:=false > "$LOG/tier${n}_launch.log" 2>&1 &
        for i in $(seq 1 120); do grep -q "Managed nodes are active" "$LOG/tier${n}_launch.log" && break; sleep 1; done
        if grep -q "Managed nodes are active" "$LOG/tier${n}_launch.log"; then up=1; break; fi
        echo "tier$n: Nav2 did not come up (attempt $attempt), relaunching"
    done
    [ "$up" = 1 ] || { echo "tier$n: Nav2 did not come up, skipping (see $LOG/tier${n}_launch.log)"; continue; }
    sleep 10
    "$ROOT/.venv/bin/python" -u "$ROOT/tools/nav_scenarios.py" --world "tier$n" --repeat "$REPEAT" 2>&1 \
        | tee "$LOG/tier${n}_scenarios.log" | grep -E "^==|PASS|FAIL|passed|Error|Traceback"
done
bash "$ROOT/tools/stop_demo.sh" > /dev/null 2>&1
echo "done - logs in $LOG"
