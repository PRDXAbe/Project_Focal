#!/usr/bin/env bash
# Stop every process started by the AdaptMap demo launch.
# Patterns use [x] brackets so pkill/pgrep never match the shell that runs them.
pkill -INT -f "[r]os2 launch adaptmap" 2>/dev/null; sleep 4
for p in "[g]z sim" "[r]viz2" "[p]arameter_bridge" "[a]daptmap_node" "[w]aypoint_driver" "[s]tatic_transform_publisher" "[r]os2 bag record"; do
  pkill -f "$p" 2>/dev/null
done
sleep 1
if pgrep -f "[g]z sim|[r]viz2|[a]daptmap_node" >/dev/null; then echo "some processes still running"; else echo "stopped"; fi
