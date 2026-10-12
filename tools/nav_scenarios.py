"""Scripted Nav2 scenarios in the demo world, with pass/fail and safety metrics.

Run while `ros2 launch adaptmap_nav nav.launch.py` is up (ROS sourced):
  .venv/bin/python tools/nav_scenarios.py [--out docs/nav_results.json]

Each scenario sends a NavigateToPose goal (odom frame = world frame in this sim) and records:
result, time, distance driven, whether the rover ever left the road (mounted a curb), its minimum
clearance to static obstacles (footprint vs manifest boxes) and to moving pedestrians.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from synth_scan import MANIFEST, actor_pose  # noqa: E402

ROAD_HALF = 4.0          # road is |y| < 4 m; beyond is the 15 cm curb + sidewalk
ROBOT_RADIUS = 0.6       # ~ half-diagonal of the 0.99 x 0.67 m footprint

# (name, goal x, goal y, expect success, time limit s, what it tests)
SCENARIOS = [
    ("pedestrian_crossing", 25.0, -1.0, True, 90, "a pedestrian crosses the road at x = 18 m"),
    ("around_lane_car", 55.0, 1.5, True, 120, "a parked car blocks the lane at x = 45 m, crates nearby"),
    ("sidewalk_goal_refused", 65.0, 6.5, False, 60, "goal behind a 15 cm curb: must not climb it"),
    ("long_route", 110.0, -2.0, True, 200, "55 m past crates and parked cars"),
    ("u_turn", 90.0, 2.0, True, 90, "turn around inside the 8 m road"),
]


def obstacles():
    """Static obstacle footprints from the manifest (non-ground classes): centre, half size, yaw."""
    objs = json.loads(MANIFEST.read_text())["objects"]
    stat = [o for o in objs if o["type"] != "actor" and o["class"] not in ("road", "sidewalk_curb", "terrain")]
    acts = [o for o in objs if o["type"] == "actor"]
    return stat, acts


def box_distance(px, py, o):
    x, y, _, yaw = o["pose"]
    hx, hy = o["size"][0] / 2, o["size"][1] / 2
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = c * (px - x) + s * (py - y), -s * (px - x) + c * (py - y)
    return math.hypot(max(abs(lx) - hx, 0.0), max(abs(ly) - hy, 0.0))


class Runner(Node):
    def __init__(self):
        super().__init__("adaptmap_nav_scenarios")
        self.pose, self.stamp = None, 0.0
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.client = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def on_odom(self, m):
        p = m.pose.pose.position
        self.pose, self.stamp = (p.x, p.y), m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    def spin_for(self, sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def run(self, name, gx, gy, expect, limit, stat, acts):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "odom"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = gx, gy
        goal.pose.pose.orientation.w = 1.0
        fut = self.client.send_goal_async(goal)
        while not fut.done():
            rclpy.spin_once(self, timeout_sec=0.05)
        handle = fut.result()
        t0, dist, last = time.time(), 0.0, self.pose
        left_road, min_static, min_ped = False, float("inf"), float("inf")
        status = GoalStatus.STATUS_REJECTED
        if handle.accepted:
            res = handle.get_result_async()
            while not res.done() and time.time() - t0 < limit:
                rclpy.spin_once(self, timeout_sec=0.05)
                x, y = self.pose
                dist += math.hypot(x - last[0], y - last[1])
                last = (x, y)
                left_road |= abs(y) > ROAD_HALF
                min_static = min(min_static, min(box_distance(x, y, o) for o in stat) - ROBOT_RADIUS)
                for a in acts:
                    ax, ay, _ = actor_pose(a, self.stamp)
                    min_ped = min(min_ped, math.hypot(x - ax, y - ay) - ROBOT_RADIUS - 0.35)
            if res.done():
                status = res.result().status
            else:
                handle.cancel_goal_async()
                self.spin_for(1.0)
                status = GoalStatus.STATUS_CANCELED
        reached = status == GoalStatus.STATUS_SUCCEEDED
        final_err = math.hypot(self.pose[0] - gx, self.pose[1] - gy)
        passed = (reached == expect) and not left_road and min_static > 0.0
        return {"scenario": name, "goal": [gx, gy], "expected_success": expect, "reached": reached,
                "status": int(status), "passed": passed, "time_s": round(time.time() - t0, 1),
                "distance_m": round(dist, 1), "final_error_m": round(final_err, 2), "left_road": left_road,
                "min_clearance_static_m": round(min_static, 2),
                "min_clearance_pedestrian_m": None if math.isinf(min_ped) else round(min_ped, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "nav_results.json"))
    ap.add_argument("--only", help="run only this scenario")
    a = ap.parse_args()
    rclpy.init()
    r = Runner()
    print("waiting for Nav2 (navigate_to_pose) and odometry ...", flush=True)
    while not r.client.wait_for_server(timeout_sec=2.0) or r.pose is None:
        rclpy.spin_once(r, timeout_sec=0.1)
    r.spin_for(5.0)  # let AdaptMap fill the costmap
    stat, acts = obstacles()
    results = []
    for name, gx, gy, expect, limit, why in SCENARIOS:
        if a.only and name != a.only:
            continue
        print(f"== {name}: goal ({gx}, {gy}) - {why}", flush=True)
        res = r.run(name, gx, gy, expect, limit, stat, acts)
        res["description"] = why
        results.append(res)
        print("   " + ("PASS" if res["passed"] else "FAIL") + f"  {json.dumps({k: res[k] for k in ('reached', 'time_s', 'distance_m', 'left_road', 'min_clearance_static_m', 'min_clearance_pedestrian_m')})}", flush=True)
        r.spin_for(2.0)
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"{sum(x['passed'] for x in results)}/{len(results)} scenarios passed -> {a.out}")
    r.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
