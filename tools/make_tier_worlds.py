"""Generate five navigation test worlds of increasing difficulty (tier1 ... tier5), each with its scenarios.

Every world is a straight street along +x (same building blocks as tools/make_world.py), and its manifest
carries the scenarios that tools/nav_scenarios.py runs (`--world tierN`):
  {"name", "start": [x, y, yaw], "goal": [x, y], "expect": "reach" | "refuse", "limit": s, "what": text}
A refuse scenario passes when the robot does NOT reach the goal (no safe way exists) and touches nothing.

  tier1  Open road        baseline: straight run, lane change, crate on the line, U-turn, sidewalk goal
  tier2  Parked street    double-parked car, a gap only 1.7 m wide, goal between parked cars, slow crosser
  tier3  Pedestrian zone  two crossers, head-on walker in the lane, walker ahead, runner, one stops mid-road
  tier4  Obstacle course  1.4 m passage, 0.55 m gap (refuse), 20 cm box, 4 cm pole, speed bump, 15 cm wall
                          across the road (refuse), overhangs at 2.2 m (pass under) and 1.3 m (refuse: mast)
  tier5  Chaos            narrow street lined with cars, walker stepping out behind a parked car, chicane,
                          group of three crossing, head-on walker, walker stopping in the lane, 125 m run

Usage: .venv/bin/python tools/make_tier_worlds.py   -> worlds/tier{1..5}.sdf + tier{1..5}_manifest.json
"""
import math
import random

import make_world as mw

WALK_W, CURB_H = mw.WALK_W, mw.CURB_H
ROBOT_TOP = 1.8  # LiDAR on its mast (the sim model's mast has no collision, the real robot's would)


def street(seed, road_w, x0, x1, trees=4):
    """Road, raised sidewalks (curbs), building blocks with gaps, poles every 15 m, a few trees."""
    rnd = random.Random(seed)
    cx, L = (x0 + x1) / 2, x1 - x0
    mw.box("road", "road", cx, 0, 0.0, L, road_w, 0.02, color=(0.15, 0.15, 0.15))
    for s in (-1, 1):
        mw.box(f"sidewalk_{'lr'[s > 0]}", "sidewalk_curb", cx, s * (road_w / 2 + WALK_W / 2), 0.0, L, WALK_W, CURB_H,
               color=(0.7, 0.7, 0.68))
    x, i = x0, 0
    while x < x1 - 10:
        w = rnd.uniform(10, 22)
        for s in (-1, 1):
            if rnd.random() < 0.8:
                d, h = rnd.uniform(8, 14), rnd.uniform(6, 18)
                mw.box(f"bldg_{i}_{s}", "building_wall", x + w / 2, s * (road_w / 2 + WALK_W + 1.0 + d / 2), 0.0,
                       w, d, h, color=(0.8, 0.75, 0.6))
        x += w + rnd.uniform(3, 8)
        i += 1
    for k, x in enumerate(range(int(x0) + 5, int(x1), 15)):
        for s in (-1, 1):
            mw.cylinder(f"pole_{k}_{s}", "pole", x, s * (road_w / 2 + 0.5), CURB_H, 0.08, 5.0, (0.3, 0.3, 0.3))
    for k in range(trees):
        x, s = rnd.uniform(x0 + 10, x1 - 10), rnd.choice((-1, 1))
        y = s * (road_w / 2 + WALK_W + 0.7)
        mw.cylinder(f"trunk_{k}", "pole", x, y, 0.0, 0.15, 3.0, (0.4, 0.25, 0.1))
        mw.sphere(f"canopy_{k}", "vegetation", x, y, 4.0, 1.6, (0.1, 0.5, 0.1))


def parked(prefix, xs, y, kinds=(0, 2, 3, 1)):
    for k, x in enumerate(xs):
        mw.car(f"{prefix}{k}", x, y, 0.0 if y > 0 else math.pi, mw.CARS[kinds[k % len(kinds)]])


def crate(name, x, y, sx=0.8, sy=0.8, sz=0.8, z=0.0, cls="other_obstacle", color=(0.8, 0.4, 0.1)):
    mw.box(name, cls, x, y, z, sx, sy, sz, color=color)


def wall_across(name, x, road_w, gap_lo, gap_hi, h=1.0):
    """A wall across the road (x = const) with one opening from y = gap_lo to gap_hi."""
    lo, hi = -road_w / 2, road_w / 2
    if gap_lo > lo:
        mw.box(f"{name}_r", "building_wall", x, (lo + gap_lo) / 2, 0.0, 0.3, gap_lo - lo, h, color=(0.55, 0.5, 0.5))
    if gap_hi < hi:
        mw.box(f"{name}_l", "building_wall", x, (gap_hi + hi) / 2, 0.0, 0.3, hi - gap_hi, h, color=(0.55, 0.5, 0.5))


def sc(name, start, goal, expect, limit, what):
    return {"name": name, "start": list(start), "goal": list(goal), "expect": expect, "limit": limit, "what": what}


def tier1():
    road_w = 8.0
    street(101, road_w, -20, 120)
    parked("car_", [70, 100], -2.9)
    crate("crate_line", 25, -1.0)                       # on the drive line
    crate("crate_0", 12, 3.0)
    crate("crate_1", 45, -3.2)
    mw.actor("ped_0", [(0, 6.5), (40, 6.5)])            # sidewalk walkers only
    mw.actor("ped_1", [(90, -6.5), (55, -6.5)])
    return "Open road", road_w, [
        sc("straight", (50, -1.0, 0), (95, -1.0), "reach", 70, "45 m straight on a clear lane"),
        sc("crate_in_lane", (0, -1.0, 0), (40, -1.0), "reach", 70, "a crate sits on the drive line at x = 25"),
        sc("lane_change", (50, -1.0, 0), (90, 2.0), "reach", 70, "goal in the opposite lane"),
        sc("u_turn", (95, -1.0, 0), (75, 2.0), "reach", 70, "turn around inside the 8 m road"),
        sc("sidewalk_goal", (50, -1.0, 0), (60, 6.5), "refuse", 60, "goal behind a 15 cm curb"),
    ]


def tier2():
    road_w = 8.0
    street(202, road_w, -20, 130)
    parked("pr_", [8, 16, 50, 58, 65, 72, 80, 90, 100, 110], -2.9)       # right side, gap 82-88 (goal)
    parked("pl_", [5, 14, 30, 45, 56, 65, 74, 95, 105], 2.9)            # left side, gap 33-42 (crossing)
    mw.car("double_parked", 33, -0.9, math.pi, mw.CARS[3])               # blocks the right lane
    crate("gap_a", 65, -1.0, 1.0, 1.0, 1.0)                              # with the parked cars, the only way past
    crate("gap_b", 65, 1.7, 1.0, 1.0, 1.0)                               # x = 65 is the 1.7 m gap between these
    mw.actor("ped_slow", [(40, -7), (40, 7)], speed=0.7)
    mw.actor("ped_0", [(0, 6.5), (60, 6.5)])
    mw.actor("ped_1", [(110, -6.5), (60, -6.5)], speed=1.4)
    return "Parked street", road_w, [
        sc("double_parked", (20, -1.5, 0), (46, -1.5), "reach", 80, "an SUV double-parked in the right lane at x = 33"),
        sc("slow_crosser", (28, 0.5, 0), (52, 0.5), "reach", 90, "a pedestrian crosses at 0.7 m/s at x = 40"),
        sc("narrow_gap", (55, 0.3, 0), (76, 0.3), "reach", 90, "crates and parked cars leave a single 1.7 m gap"),
        sc("park_between_cars", (74, -0.5, 0), (85, -2.6), "reach", 80, "goal in a 6 m gap between parked cars"),
        sc("long_route", (0, 0.0, 0), (118, -0.5), "reach", 240, "118 m past everything above"),
    ]


def tier3():
    road_w = 8.0
    street(303, road_w, -20, 200, trees=3)
    mw.actor("cross_a", [(20, -7), (20, 7)], speed=1.0)
    mw.actor("cross_b", [(23, 7), (23, -7)], speed=1.2)                  # opposite direction, staggered
    mw.actor("head_on", [(68, -1.5), (45, -1.5)], speed=1.0)            # walks the rover's lane, both ways
    mw.actor("ahead", [(80, 1.5), (110, 1.5)], speed=0.8)               # walks the left lane
    mw.actor("runner", [(125, -7), (125, 7)], speed=2.0)
    mw.actor("stopper", [(145, -7), (145, -1.5), (145, 7)], speed=1.0, waits={1: 6.0})
    mw.actor("diagonal", [(160, -7), (172, 7)], speed=1.1)
    return "Pedestrian zone", road_w, [
        sc("two_crossers", (0, -1.5, 0), (35, -1.5), "reach", 90, "two people cross in opposite directions at x = 20 and 23"),
        sc("head_on", (38, -1.5, 0), (75, -1.5), "reach", 90, "a person walks down the rover's lane, towards it half the time"),
        sc("walker_ahead", (75, 1.5, 0), (115, 1.5), "reach", 100, "a person walks ahead in the left lane at 0.8 m/s"),
        sc("runner", (110, -1.5, 0), (138, -1.5), "reach", 80, "a runner crosses at 2 m/s at x = 125"),
        sc("stops_in_lane", (130, -1.5, 0), (158, -1.5), "reach", 90, "a person crosses at x = 145 and stands 6 s in the lane"),
        sc("diagonal", (152, -1.5, 0), (185, -1.5), "reach", 90, "a person crosses diagonally (x 160 -> 172)"),
    ]


def tier4():
    road_w = 8.0
    street(404, road_w, -20, 225, trees=2)
    wall_across("passage", 20, road_w, -1.7, -0.3)                       # 1.4 m opening
    wall_across("too_narrow", 50, road_w, -1.3, -0.75)                   # 0.55 m: the rover is 0.67 m wide
    crate("low_box", 75, -1.0, 0.5, 0.5, 0.2)                            # 20 cm: above the 12 cm step limit
    mw.cylinder("thin_pole", "pole", 100, -1.0, 0.0, 0.04, 1.5, (0.9, 0.9, 0.1))
    crate("speed_bump", 125, 0, 0.6, road_w, 0.06, cls="road", color=(0.9, 0.8, 0.1))   # 6 cm: drivable
    mw.box("curb_wall", "other_obstacle", 150, 0, 0.0, 0.4, road_w, 0.15, color=(0.7, 0.7, 0.68))  # 15 cm, full width
    crate("bar_high", 175, 0, 0.3, road_w, 0.3, z=2.2, color=(0.2, 0.2, 0.8))    # bottom 2.2 m: above the mast
    crate("bar_low", 200, 0, 0.3, road_w, 0.3, z=1.3, color=(0.8, 0.2, 0.2))     # bottom 1.3 m: hits the LiDAR mast
    return "Obstacle course", road_w, [
        sc("narrow_passage", (5, -1.0, 0), (35, -1.0), "reach", 80, "a wall across the road with a 1.4 m opening"),
        sc("gap_too_narrow", (38, -1.0, 0), (62, -1.0), "refuse", 60, "a wall across the road with only a 0.55 m gap"),
        sc("low_box", (62, -1.0, 0), (88, -1.0), "reach", 70, "a 20 cm box on the drive line (lost from view up close)"),
        sc("thin_pole", (88, -1.0, 0), (112, -1.0), "reach", 70, "a 4 cm thin pole on the drive line"),
        sc("speed_bump", (112, -1.0, 0), (138, -1.0), "reach", 70, "a 6 cm bump across the road: drive over it"),
        sc("curb_wall", (138, -1.0, 0), (160, -1.0), "refuse", 60, "a 15 cm wall across the whole road: cannot climb"),
        sc("overhang_high", (162, -1.0, 0), (188, -1.0), "reach", 70, "a bar across the road 2.2 m up: drive under"),
        sc("overhang_low", (188, -1.0, 0), (212, -1.0), "refuse", 60, "a bar 1.3 m up: the 1.8 m LiDAR mast would hit it"),
    ]


def tier5():
    road_w = 7.0
    street(505, road_w, -20, 140, trees=4)
    parked("pr_", [-5, 3, 11, 22, 33, 41, 49, 70, 78, 86, 94, 118, 126], -2.5)   # gaps: 25-31 occluded walker,
    parked("pl_", [0, 8, 16, 30, 38, 46, 70, 82, 90, 98, 120, 128], 2.5)         #       55-67 group, 100-116 stopper
    crate("chicane_a", 40, -0.8, 1.0, 1.0, 1.0)
    crate("chicane_b", 47, 0.8, 1.0, 1.0, 1.0)
    crate("low_crate", 75, 0.3, 0.6, 0.6, 0.25)
    mw.actor("occluded", [(26.5, -5.0), (26.5, 5.0)], speed=1.3)         # steps out right behind car pr_3 (x 20-24)
    for k, (x, sp) in enumerate([(59, 1.0), (60.5, 1.05), (62, 0.95)]):
        mw.actor(f"group_{k}", [(x, 5.0), (x, -5.0)], speed=sp)
    mw.actor("head_on", [(96, 0.0), (80, 0.0)], speed=1.0)
    mw.actor("stopper", [(108, -5.0), (108, 0.0), (108, 5.0)], speed=1.0, waits={1: 5.0})
    return "Chaos", road_w, [
        sc("occluded_walker", (5, 0.0, 0), (35, 0.0), "reach", 90, "a person steps out from behind a parked car at x = 26.5"),
        sc("chicane", (32, 0.0, 0), (55, 0.0), "reach", 80, "crates force an S-bend at x = 40 and 47"),
        sc("group_crossing", (50, 0.0, 0), (72, 0.0), "reach", 90, "three people cross together at x = 59-62"),
        sc("head_on_narrow", (72, 0.0, 0), (99, 0.0), "reach", 100, "a person walks down the single lane; a 25 cm crate at x = 75"),
        sc("stopper", (98, 0.0, 0), (116, 0.0), "reach", 90, "a person stops 5 s in the lane at x = 108"),
        sc("full_run", (0, 0.0, 0), (125, 0.0), "reach", 300, "125 m through all of the above"),
    ]


def main():
    for n, fn in enumerate([tier1, tier2, tier3, tier4, tier5], 1):
        mw.objs.clear()
        mw.sdf.clear()
        title, road_w, scenarios = fn()
        start = scenarios[0]["start"]
        mw.write(f"tier{n}", {"tier": n, "title": title, "road_half": road_w / 2, "robot_top": ROBOT_TOP,
                              "scenarios": scenarios}, start=(start[0], start[1], start[2]))


if __name__ == "__main__":
    main()
