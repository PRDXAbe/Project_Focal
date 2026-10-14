"""Scripted Nav2 scenarios in the demo world, with pass/fail and safety metrics.

Run while `ros2 launch adaptmap_nav nav.launch.py [world:=tierN.sdf]` is up (ROS sourced):
  .venv/bin/python tools/nav_scenarios.py [--world tierN] [--out FILE] [--repeat 3] [--only NAME]
The demo world uses the built-in SCENARIOS below (-> docs/nav_results.json); the tier worlds
(tools/make_tier_worlds.py) carry their own scenarios in their manifest (-> docs/nav_tierN.json).

Each scenario sends a NavigateToPose goal (odom frame = world frame in this sim) and records:
result, time, distance driven, whether the rover ever left the road (mounted a curb), its minimum
clearance (from the rover's footprint outline) to static obstacles (manifest boxes) and to moving
pedestrians; 0 or less means contact. Contact with a static obstacle, or with a pedestrian while the rover
is moving, fails the scenario (Gazebo actors walk through a stopped rover; that is reported, not failed).
Before each scenario the rover is teleported to its start pose (gz set_pose), so scenarios are independent.
"""
import argparse
import json
import math
import subprocess
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
from synth_scan import actor_pose  # noqa: E402

WORLDS = ROOT / "ros2_ws" / "src" / "adaptmap_sim" / "worlds"
ROAD_HALF = 4.0          # road is |y| < 4 m; beyond is the 15 cm curb + sidewalk (tier manifests set their own)
ROBOT_TOP = 1.8          # LiDAR on its mast: overhangs below this would hit the real robot
HALF_L, HALF_W = 0.495, 0.335   # rover footprint 0.99 x 0.67 m
PED_RADIUS = 0.35

# (name, start x, y, yaw, goal x, goal y, expect success, time limit s, what it tests)
# Each scenario starts with the rover placed at its start pose, so one outcome can't affect the next.
SCENARIOS = [
    ("pedestrian_crossing", 0.0, -1.0, 0.0, 25.0, -1.0, True, 90, "a pedestrian crosses the road at x = 18 m"),
    ("around_lane_car", 25.0, -1.0, 0.0, 55.0, 1.5, True, 120, "a parked car blocks the lane at x = 45 m, crates nearby"),
    ("sidewalk_goal_refused", 55.0, 1.5, 0.0, 65.0, 6.5, False, 60, "goal behind a 15 cm curb: must not climb it"),
    ("long_route", 55.0, -1.0, 0.0, 110.0, -2.0, True, 200, "55 m past crates and parked cars"),
    ("u_turn", 110.0, -2.0, 0.0, 90.0, 2.0, True, 90, "turn around inside the 8 m road"),
]
MOVING = 0.05  # m/s: a pedestrian touching a rover slower than this walked into it (sim actors don't avoid)


def footprint(x, y, yaw, step=0.05):
    """Points on the rover's footprint outline (world frame); clearances are measured from these."""
    nl, nw = int(2 * HALF_L / step) + 1, int(2 * HALF_W / step) + 1
    ls, ws = np.linspace(-HALF_L, HALF_L, nl), np.linspace(-HALF_W, HALF_W, nw)
    local = np.concatenate([np.stack([ls, np.full(nl, s * HALF_W)], 1) for s in (-1, 1)] +
                           [np.stack([np.full(nw, s * HALF_L), ws], 1) for s in (-1, 1)])
    c, s_ = math.cos(yaw), math.sin(yaw)
    return [(x + c * a - s_ * b, y + s_ * a + c * b) for a, b in local]


def obstacles(manifest, robot_top=ROBOT_TOP):
    """Static obstacles the robot can touch (non-ground classes, bottom below the robot's top), and actors."""
    objs = manifest["objects"]
    stat = [o for o in objs if o["type"] != "actor" and o["class"] not in ("road", "sidewalk_curb", "terrain")
            and o["pose"][2] - o["size"][2] / 2 < robot_top]
    for o in stat:
        o["reach"] = math.hypot(o["size"][0], o["size"][1]) / 2   # centre-to-corner, for the proximity filter
    acts = [o for o in objs if o["type"] == "actor"]
    return stat, acts


def box_distance(px, py, o):
    """Distance from a point to an object's footprint (cylinders and spheres as circles, boxes as rectangles)."""
    x, y, _, yaw = o["pose"]
    if o["type"] in ("cylinder", "sphere"):
        return max(math.hypot(px - x, py - y) - o["size"][0] / 2, 0.0)
    hx, hy = o["size"][0] / 2, o["size"][1] / 2
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = c * (px - x) + s * (py - y), -s * (px - x) + c * (py - y)
    return math.hypot(max(abs(lx) - hx, 0.0), max(abs(ly) - hy, 0.0))


class Runner(Node):
    def __init__(self):
        super().__init__("adaptmap_nav_scenarios")
        self.pose, self.yaw, self.speed, self.stamp = None, 0.0, 0.0, 0.0
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.client = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.speed = math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y)
        self.pose, self.stamp = (p.x, p.y), m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    def spin_for(self, sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def place(self, x, y, yaw):
        """Teleport the rover (Gazebo set_pose) and let it settle and the map refill."""
        req = (f'name: "rover", position: {{x: {x}, y: {y}, z: 0.05}}, '
               f'orientation: {{z: {math.sin(yaw / 2)}, w: {math.cos(yaw / 2)}}}')
        subprocess.run(["gz", "service", "-s", "/world/urban/set_pose", "--reqtype", "gz.msgs.Pose",
                        "--reptype", "gz.msgs.Boolean", "--timeout", "3000", "--req", req],
                       check=True, capture_output=True)
        self.spin_for(3.0)
        if math.hypot(self.pose[0] - x, self.pose[1] - y) > 0.5:
            raise RuntimeError(f"rover not at start ({x}, {y}): odometry says {self.pose}")

    def run(self, name, gx, gy, expect, limit, stat, acts, road_half=ROAD_HALF):
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
        left_road, min_static, min_ped, ped_speed = False, float("inf"), float("inf"), None
        static_name = ped_name = None
        track, last_track = [], -1.0
        status = GoalStatus.STATUS_UNKNOWN  # goal rejected
        if handle.accepted:
            res = handle.get_result_async()
            while not res.done() and time.time() - t0 < limit:
                rclpy.spin_once(self, timeout_sec=0.05)
                x, y = self.pose
                dist += math.hypot(x - last[0], y - last[1])
                last = (x, y)
                left_road |= abs(y) > road_half
                if time.time() - last_track >= 0.5:
                    track.append([round(self.stamp, 1), round(x, 2), round(y, 2), round(self.speed, 2)])
                    last_track = time.time()
                near = [o for o in stat if math.hypot(o["pose"][0] - x, o["pose"][1] - y) - o["reach"] < 8.0]
                fp = footprint(x, y, self.yaw) if near or acts else []
                for o in near:  # 0 = touching
                    d = min(box_distance(px, py, o) for px, py in fp)
                    if d < min_static:
                        min_static, static_name = d, o["name"]
                for a in acts:
                    ax, ay, _ = actor_pose(a, self.stamp)
                    if math.hypot(x - ax, y - ay) < 3.0:
                        d = min(math.hypot(px - ax, py - ay) for px, py in fp) - PED_RADIUS
                        if d < min_ped:
                            min_ped, ped_speed, ped_name = d, self.speed, a["name"]
            if res.done():
                status = res.result().status
            else:
                handle.cancel_goal_async()
                self.spin_for(1.0)
                status = GoalStatus.STATUS_CANCELED
        reached = status == GoalStatus.STATUS_SUCCEEDED
        final_err = math.hypot(self.pose[0] - gx, self.pose[1] - gy)
        ped_hit = min_ped <= 0.0 and ped_speed > MOVING  # the rover drove into a pedestrian
        passed = (reached == expect) and not left_road and min_static > 0.0 and not ped_hit
        return {"scenario": name, "goal": [gx, gy], "expected_success": expect, "reached": reached,
                "status": int(status), "passed": passed, "time_s": round(time.time() - t0, 1),
                "distance_m": round(dist, 1), "final_error_m": round(final_err, 2), "left_road": left_road,
                "min_clearance_static_m": None if math.isinf(min_static) else round(min_static, 2),
                "min_clearance_pedestrian_m": None if math.isinf(min_ped) else round(min_ped, 2),
                "rover_speed_at_min_pedestrian_m_s": None if ped_speed is None else round(ped_speed, 2),
                "drove_into_pedestrian": ped_hit, "closest_static": static_name, "closest_pedestrian": ped_name,
                "track": track}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="urban", help="urban (demo world, built-in scenarios) or tier1 ... tier5")
    ap.add_argument("--out", help="default docs/nav_results.json (urban) or docs/nav_<world>.json")
    ap.add_argument("--only", help="run only this scenario")
    ap.add_argument("--repeat", type=int, default=1, help="run the suite this many times (pedestrian timing varies)")
    a = ap.parse_args()
    rclpy.init()
    r = Runner()
    print("waiting for Nav2 (navigate_to_pose) and odometry ...", flush=True)
    while not r.client.wait_for_server(timeout_sec=2.0) or r.pose is None:
        rclpy.spin_once(r, timeout_sec=0.1)
    r.spin_for(5.0)  # let AdaptMap fill the costmap
    manifest = json.loads((WORLDS / f"{a.world}_manifest.json").read_text())
    stat, acts = obstacles(manifest, manifest.get("robot_top", ROBOT_TOP))
    road_half = manifest.get("road_half", ROAD_HALF)
    scenarios = [(c["name"], *map(float, c["start"]), *map(float, c["goal"]), c["expect"] == "reach", c["limit"],
                  c["what"]) for c in manifest["scenarios"]] if "scenarios" in manifest else SCENARIOS
    out = a.out or str(ROOT / "docs" / ("nav_results.json" if a.world == "urban" else f"nav_{a.world}.json"))
    results = []
    for run in range(a.repeat):
        for name, sx, sy, syaw, gx, gy, expect, limit, why in scenarios:
            if a.only and name != a.only:
                continue
            print(f"== run {run + 1}/{a.repeat} {name}: ({sx}, {sy}) -> ({gx}, {gy}) - {why}", flush=True)
            r.place(sx, sy, syaw)
            res = r.run(name, gx, gy, expect, limit, stat, acts, road_half)
            res["run"], res["world"] = run, a.world
            res["start"] = [sx, sy, syaw]
            res["description"] = why
            results.append(res)
            print("   " + ("PASS" if res["passed"] else "FAIL") + f"  {json.dumps({k: res[k] for k in ('reached', 'time_s', 'distance_m', 'left_road', 'min_clearance_static_m', 'min_clearance_pedestrian_m', 'rover_speed_at_min_pedestrian_m_s')})}", flush=True)
            r.spin_for(2.0)
    Path(out).write_text(json.dumps(results, indent=1))
    print(f"{sum(x['passed'] for x in results)}/{len(results)} scenarios passed -> {out}")
    r.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
