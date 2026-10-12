#!/usr/bin/env bash
# Stop every process started by the AdaptMap demo launch.
# Never kills its own ancestors (a calling shell whose command line mentions these names).
ancestors=" "
p=$$
while [ -n "$p" ] && [ "$p" != 0 ]; do ancestors+="$p "; p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); done

kill_matching() {  # $1 = signal, $2 = pgrep -f pattern
  for pid in $(pgrep -f "$2"); do
    case "$ancestors" in *" $pid "*) ;; *) kill "-$1" "$pid" 2>/dev/null ;; esac
  done
}

kill_matching INT "ros2 launch adaptmap"; sleep 4
for p in "gz sim" "rviz2" "parameter_bridge" "adaptmap_node" "waypoint_driver" "static_transform_publisher" \
         "ros2 bag record" "nav2" "controller_server" "planner_server" "bt_navigator"; do
  kill_matching TERM "$p"
done
sleep 1
left=$(for pid in $(pgrep -f "gz sim|rviz2|adaptmap_node"); do case "$ancestors" in *" $pid "*) ;; *) echo "$pid" ;; esac; done)
if [ -n "$left" ]; then echo "some processes still running"; else echo "stopped"; fi
