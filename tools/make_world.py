"""Generate the AdaptMap urban test world (Gazebo Harmonic SDF) + a ground-truth object manifest.

The manifest lists every labelled object (AdaptMap class, pose, box size, or actor trajectory) so
tools/autolabel.py can produce exact per-point labels for evaluation / fine-tuning.

Usage:
  python tools/make_world.py              demo world  -> worlds/urban.sdf + urban_manifest.json
  python tools/make_world.py --seed 11    randomised training world -> worlds/urban_s11.sdf + urban_s11_manifest.json
    (random buildings, crates/barrels, lane car, trees and 4-7 pedestrians; the drive line y = -1 m stays clear)
"""
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ros2_ws" / "src" / "adaptmap_sim" / "worlds"
FUEL = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/"
CARS = [("Hatchback", (4.2, 1.8, 1.5)), ("Hatchback blue", (4.2, 1.8, 1.5)),
        ("Pickup", (5.6, 2.1, 1.9)), ("SUV", (4.8, 2.0, 1.9))]
WALK = "https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae"

ROAD_W, WALK_W, CURB_H = 8.0, 3.0, 0.15
X0, X1 = -40.0, 130.0

objs, sdf = [], []


def mat(r, g, b):
    return f"<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>"


def box(name, cls, x, y, z, sx, sy, sz, yaw=0.0, color=(0.6, 0.6, 0.6)):
    """Static box with its base at height z."""
    objs.append({"name": name, "class": cls, "type": "box", "pose": [x, y, z + sz / 2, yaw], "size": [sx, sy, sz]})
    sdf.append(f"""
    <model name="{name}"><static>true</static><pose>{x} {y} {z + sz / 2} 0 0 {yaw}</pose>
      <link name="l"><collision name="c"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>
      <visual name="v"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>{mat(*color)}</visual></link></model>""")


def cylinder(name, cls, x, y, z, r, h, color):
    objs.append({"name": name, "class": cls, "type": "cylinder", "pose": [x, y, z + h / 2, 0.0], "size": [2 * r, 2 * r, h]})
    sdf.append(f"""
    <model name="{name}"><static>true</static><pose>{x} {y} {z + h / 2} 0 0 0</pose>
      <link name="l"><collision name="c"><geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry></collision>
      <visual name="v"><geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>{mat(*color)}</visual></link></model>""")


def sphere(name, cls, x, y, z, r, color):
    objs.append({"name": name, "class": cls, "type": "sphere", "pose": [x, y, z, 0.0], "size": [2 * r] * 3})
    sdf.append(f"""
    <model name="{name}"><static>true</static><pose>{x} {y} {z} 0 0 0</pose>
      <link name="l"><collision name="c"><geometry><sphere><radius>{r}</radius></sphere></geometry></collision>
      <visual name="v"><geometry><sphere><radius>{r}</radius></sphere></geometry>{mat(*color)}</visual></link></model>""")


def car(name, x, y, yaw, kind):
    model, size = kind
    objs.append({"name": name, "class": "vehicle", "type": "box", "pose": [x, y, size[2] / 2, yaw], "size": list(size)})
    sdf.append(f"""
    <include><name>{name}</name><uri>{FUEL}{model.replace(' ', '%20')}</uri><static>true</static>
      <pose>{x} {y} 0 0 0 {yaw}</pose></include>""")


def actor(name, waypoints, speed=1.2, waits=None):
    """Walking pedestrian looping through waypoints (closed loop) at constant speed; `waits` maps a
    waypoint index to seconds the pedestrian stands still there before walking on.

    Gazebo moves actors along a spline through the waypoints; with only the corners given it speeds up
    and slows down (up to ~2.5 m off a constant-speed walk), so waypoints are placed ~1 s apart along
    each leg. The manifest trajectory then matches the simulated actor (`synth_scan.actor_pose`)."""
    pts = waypoints + [waypoints[0]]
    t, traj = 0.0, []
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(pts, pts[1:])):
        seg = math.hypot(x1 - x0, y1 - y0)
        yaw = math.atan2(y1 - y0, x1 - x0)
        if waits and waits.get(i):
            traj.append((round(t, 3), x0, y0, yaw))
            t += waits[i]
        n = max(1, math.ceil(seg / speed))
        for k in range(n):
            a = k / n
            traj.append((round(t + a * seg / speed, 3), round(x0 + a * (x1 - x0), 3), round(y0 + a * (y1 - y0), 3), yaw))
        t += seg / speed
    traj.append((round(t, 3), pts[-1][0], pts[-1][1], traj[0][3]))
    objs.append({"name": name, "class": "pedestrian", "type": "actor", "size": [0.7, 0.7, 1.8],
                 "trajectory": traj, "period": t})
    wps = "".join(f"<waypoint><time>{tt:.2f}</time><pose>{x} {y} 1.0 0 0 {yw:.3f}</pose></waypoint>" for tt, x, y, yw in traj)
    sdf.append(f"""
    <actor name="{name}"><skin><filename>{WALK}</filename><scale>1.0</scale></skin>
      <animation name="walk"><filename>{WALK}</filename><interpolate_x>true</interpolate_x></animation>
      <script><loop>true</loop><auto_start>true</auto_start>
        <trajectory id="0" type="walk">{wps}</trajectory></script></actor>""")


DRIVE_Y = -1.0  # scripted laps drive along this line; training worlds keep obstacles >= 1.3 m away


def off_line_y(rnd):
    """A y on the road that keeps a 0.8 m obstacle clear of the rover's drive line."""
    return rnd.choice([rnd.uniform(0.6, 3.3), rnd.uniform(-3.5, -2.6)])


def write(name, meta, start=(0, -1.0, 0)):
    """Write worlds/<name>.sdf (world name stays "urban": tools address /world/urban/...) and its manifest."""
    world = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="urban">
    <physics name="1ms" type="ignored"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <scene><ambient>0.6 0.6 0.6 1</ambient><background>0.7 0.8 0.9 1</background><shadows>false</shadows></scene>
    <light type="directional" name="sun"><cast_shadows>false</cast_shadows><pose>0 0 50 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><direction>-0.4 0.2 -0.9</direction></light>
    <model name="ground"><static>true</static>
      <link name="l"><collision name="c"><geometry><plane><normal>0 0 1</normal><size>400 400</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>400 400</size></plane></geometry>{mat(0.45, 0.6, 0.3)}</visual></link></model>
    <include><uri>model://rover</uri><name>rover</name><pose>{start[0]} {start[1]} 0.02 0 0 {start[2]}</pose></include>
    {''.join(sdf)}
  </world>
</sdf>
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.sdf").write_text(world)
    (OUT / f"{name}_manifest.json").write_text(json.dumps({"objects": objs, "ground_class": "terrain", **meta}, indent=1))
    print(f"wrote {OUT / (name + '.sdf')} with {len(objs)} labelled objects")


def build(seed=None):
    train = seed is not None
    rnd = random.Random(7 if seed is None else seed)
    L = X1 - X0
    cx = (X0 + X1) / 2
    # ground layers: terrain everywhere (ground plane), road strip, raised sidewalks (curbs)
    box("road", "road", cx, 0, 0.0, L, ROAD_W, 0.02, color=(0.15, 0.15, 0.15))
    for s in (-1, 1):
        y = s * (ROAD_W / 2 + WALK_W / 2)
        box(f"sidewalk_{'lr'[s > 0]}", "sidewalk_curb", cx, y, 0.0, L, WALK_W, CURB_H, color=(0.7, 0.7, 0.68))
    # buildings / walls beyond the sidewalks, with gaps
    x = X0
    i = 0
    while x < X1 - 10:
        w = rnd.uniform(10, 22)
        for s in (-1, 1):
            if rnd.random() < 0.8:
                d = rnd.uniform(8, 14)
                h = rnd.uniform(6, 18)
                y = s * (ROAD_W / 2 + WALK_W + 1.0 + d / 2)
                box(f"bldg_{i}_{s}", "building_wall", x + w / 2, y, 0.0, w, d, h, color=(0.8, 0.75, 0.6))
        x += w + rnd.uniform(3, 8)
        i += 1
    # a low wall / fence segment and a grass+slope area on the left at the far end
    box("wall_0", "building_wall", 60, 14.5, 0.0, 20, 0.3, 1.2, color=(0.6, 0.5, 0.45))
    # poles every 15 m on both sidewalks
    for k, x in enumerate(range(int(X0) + 5, int(X1), 15)):
        for s in (-1, 1):
            cylinder(f"pole_{k}_{s}", "pole", x, s * (ROAD_W / 2 + 0.5), CURB_H, 0.08, 5.0, (0.3, 0.3, 0.3))
    # trees (trunk = pole class like SemanticKITTI, canopy = vegetation)
    for k in range(rnd.randint(6, 16) if train else 8):
        x, y = rnd.uniform(-20, 110), rnd.choice((-1, 1)) * rnd.uniform(ROAD_W / 2 + WALK_W + 0.5, ROAD_W / 2 + WALK_W + 1.0)
        cylinder(f"trunk_{k}", "pole", x, y, 0.0, 0.15, 3.0, (0.4, 0.25, 0.1))
        sphere(f"canopy_{k}", "vegetation", x, y, 4.0, 1.6, (0.1, 0.5, 0.1))
    # parked cars along the road edge, a couple in the lane as obstacles
    for k in range(10):
        x = rnd.uniform(-30, 120)
        s = rnd.choice((-1, 1))
        car(f"car_{k}", x, s * (ROAD_W / 2 - 1.1), 0.0 if s > 0 else math.pi, rnd.choice(CARS))
    if not train:
        car("car_lane", 45.0, 1.0, 0.1, CARS[2])
        # misc obstacles: fixed positions, clear of the demo drive line (y = -1.0)
        for k, (x, y) in enumerate([(20, 2.0), (38, 2.2), (60, 2.2), (70, -3.0), (88, 2.0), (95, -3.0)]):
            box(f"crate_{k}", "other_obstacle", x, y, 0.0, 0.8, 0.8, 0.8, color=(0.8, 0.4, 0.1))
    else:
        car("car_lane", rnd.uniform(15, 110), 1.3, rnd.uniform(-0.15, 0.15), rnd.choice(CARS))
        for k in range(rnd.randint(6, 14)):
            x, y = rnd.uniform(5, 115), off_line_y(rnd)
            if rnd.random() < 0.6:
                sz = rnd.uniform(0.4, 1.1)
                box(f"crate_{k}", "other_obstacle", x, y, 0.0, sz, sz * rnd.uniform(0.7, 1.4), rnd.uniform(0.3, 1.2),
                    yaw=rnd.uniform(0, math.pi), color=(rnd.uniform(0.5, 0.9), rnd.uniform(0.2, 0.5), 0.1))
            else:
                cylinder(f"barrel_{k}", "other_obstacle", x, y, 0.0, rnd.uniform(0.25, 0.45), rnd.uniform(0.6, 1.1),
                         (0.2, 0.3, rnd.uniform(0.4, 0.8)))
    # sloped ramp (terrain) for elevation demo
    objs.append({"name": "ramp", "class": "terrain", "type": "box", "pose": [25, -20, 0.5, 0.0], "size": [12, 8, 1.0]})
    sdf.append(f"""
    <model name="ramp"><static>true</static><pose>25 -20 -0.3 0 0.15 0</pose>
      <link name="l"><collision name="c"><geometry><box><size>12 8 1</size></box></geometry></collision>
      <visual name="v"><geometry><box><size>12 8 1</size></box></geometry>{mat(0.35, 0.55, 0.2)}</visual></link></model>""")
    # pedestrians: two along sidewalks, one crossing the road in front of the robot
    if not train:
        actor("ped_0", [(5, 6.5), (40, 6.5)])
        actor("ped_1", [(60, -6.5), (20, -6.5)])
        actor("ped_2", [(18, -7), (18, 7)], speed=1.0)
    else:
        for k in range(rnd.randint(4, 7)):
            if rnd.random() < 0.6:  # walk along a sidewalk
                s, x0 = rnd.choice((-1, 1)), rnd.uniform(0, 90)
                y = s * rnd.uniform(5.0, 6.8)
                actor(f"ped_{k}", [(x0, y), (x0 + rnd.uniform(15, 35), y)], speed=rnd.uniform(0.8, 1.5))
            else:                   # cross the road
                x0 = rnd.uniform(10, 110)
                actor(f"ped_{k}", [(x0, -7.0), (x0 + rnd.uniform(-3, 3), 7.0)], speed=rnd.uniform(0.8, 1.4))

    write("urban" if seed is None else f"urban_s{seed}", {"seed": seed})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, help="build a randomised training world with this seed")
    build(ap.parse_args().seed)
