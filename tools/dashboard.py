"""Judge-facing dashboard: one local web page that tells the AdaptMap story.

  Why       - 702,400 foveated cells instead of 16 M, the zone layout, speed-adaptive zones (from docs/*.json)
  Sees      - live top-down map from the running node: semantic classes or what the planner sees, the
              robot, its Nav2 path and the predicted paths of walking pedestrians
  Works     - scenario buttons (rover placed at the start, Nav2 goal sent, metrics as in nav_scenarios.py),
              a plain-language caption of what the robot is doing, and the recorded pass counts

Usage (normally via tools/dashboard.sh, which also starts the sim + AdaptMap + Nav2):
  .venv/bin/python tools/dashboard.py [--world urban|tierN] [--port 8080] [--offline]
Without a running simulation (or with --offline) the page shows the recorded results and demo video.
Runs from the dashboard are kept in memory only; docs/nav_*.json stay as the suite runs wrote them.
"""
import argparse
import io
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, Response, abort, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from adaptmap import classes  # noqa: E402
from adaptmap.config import load_config  # noqa: E402

DOCS, FIG = ROOT / "docs", ROOT / "docs" / "figures"
WORLDS = ROOT / "ros2_ws" / "src" / "adaptmap_sim" / "worlds"
HALF, NPX = 25.0, 800          # live view: +-25 m around the robot, 800 px square (6.25 cm / px)
PX = 2 * HALF / NPX
BG = np.array([18, 20, 26], np.uint8)
STALE = 3.0                    # s without /adaptmap/stats -> simulation counts as not running


def load(name):
    p = DOCS / name
    return json.loads(p.read_text()) if p.exists() else None


def pick(rows, word):
    """The row whose model name contains `word` (e.g. 'default'), else the last one."""
    return next((r for r in rows if word in r["model"]), rows[-1])


# ------------------------------------------------------------------ recorded results
def summary(cfg, world):
    out = {"world": world, "zones": [{"name": z["name"], "cell": z["cell"], "half": z["half_extent"]}
                                     for z in cfg["zones"]]}
    b = load("results.json")
    if b:
        s, lat = b["stats"], b["latency_ms"]
        for z in out["zones"]:
            z["cells"] = s["layers"][z["name"]]["ring_cells"]
        tot = lat["pre"] + lat["seg"] + lat["fuse"]
        out["map"] = {"cells": s["total_cells"], "uniform_cells": s["uniform_cells"],
                      "reduction": round(s["reduction"], 1), "mem_mb": [round(m) for m in b["mem_mb"]],
                      "mem_ratio": round(b["mem_mb"][1] / b["mem_mb"][0]), "pipeline_ms": round(tot, 1),
                      "hz": round(1000 / tot), "fuse_speedup": round(lat["uni_fuse"] / lat["fuse"], 1)}
    seg = {}
    for key, f in (("test_world", "eval_world21.json"), ("demo_world", "eval_demo_world.json")):
        rows = load(f)
        if rows:
            seg[key] = {"pretrained": round(100 * rows[0]["miou"], 1),
                        "finetuned": round(100 * pick(rows, "default")["miou"], 1)}
    poss = load("semanticposs_results.json")
    if poss:
        seg["semanticposs_pretrained"] = round(100 * poss[0]["miou"], 1)
    out["segmentation"] = seg
    kitti = sorted(DOCS.glob("kitti_*_summary.json"))
    if kitti:
        k = json.loads(kitti[0].read_text())
        out["kitti"] = {"hz": k["hz"], "ms": k["total_ms"], "scans": k["scans"]}
    az = load("adaptive_zones.json")
    if az:
        out["adaptive"] = [{"speed": r["speed"], "radius": r["near_radius_m"], "cells": r["cells"]} for r in az]
    soak = load("soak_summary.json")
    if soak:
        out["soak"] = {"minutes": soak["minutes"], "rss_growth_mb": soak["rss_mb_end"] - soak["rss_mb_after_1_min"]}
    nav = []
    for label, f, manifest in [("Demo world", "nav_results.json", "urban")] + \
            [(None, f"nav_tier{n}.json", f"tier{n}") for n in range(1, 6)]:
        rows = load(f)
        if not rows:
            continue
        mp = WORLDS / f"{manifest}_manifest.json"
        title = label or json.loads(mp.read_text()).get("title", manifest)
        nav.append({"world": manifest, "title": title, "passed": sum(r["passed"] for r in rows), "runs": len(rows),
                    "ped_hits": sum(r["drove_into_pedestrian"] for r in rows)})
    out["nav"] = nav
    out["videos"] = [v for v in ("nav_demo.mp4", "demo.mp4", "explainer_25d.mp4") if (FIG / v).exists()]
    return out


def urban_scenarios():
    """nav_scenarios.SCENARIOS, read without importing it (it needs ROS; the offline page does not)."""
    import ast
    tree = ast.parse((ROOT / "tools" / "nav_scenarios.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SCENARIOS")
    return ast.literal_eval(node.value)


def scenarios(world):
    """(name, start (x, y, yaw), goal (x, y), expect reach, time limit, description) and recorded pass rates."""
    if world == "urban":
        sc = [(n, (sx, sy, syaw), (gx, gy), e, lim, why) for n, sx, sy, syaw, gx, gy, e, lim, why in urban_scenarios()]
        rec = load("nav_results.json") or []
    else:
        m = json.loads((WORLDS / f"{world}_manifest.json").read_text())
        sc = [(c["name"], tuple(map(float, c["start"])), tuple(map(float, c["goal"])), c["expect"] == "reach",
               c["limit"], c["what"]) for c in m["scenarios"]]
        rec = load(f"nav_{world}.json") or []
    hist = {n: [sum(r["passed"] for r in rec if r["scenario"] == n), sum(r["scenario"] == n for r in rec)]
            for n, *_ in sc}
    return sc, hist


# ------------------------------------------------------------------ live ROS side
class Live:
    """Latest AdaptMap / Nav2 state from ROS topics, spun on its own executor thread."""

    def __init__(self, cfg):
        import rclpy
        from nav_msgs.msg import OccupancyGrid, Odometry, Path as PathMsg
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import String
        from render_live import cloud_xyzc

        self.cfg, self.lock = cfg, threading.Lock()
        self.cells, self.cost, self.stats, self.plan = {}, None, None, None
        self.pose, self.speed, self.t_stats, self.rate = (0.0, 0.0, 0.0), 0.0, 0.0, 0.0
        self.node = n = Node("adaptmap_dashboard")
        be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        def on_cells(zone, m):
            xyzc = cloud_xyzc(m)
            with self.lock:
                self.cells[zone] = xyzc

        def on_cost(m):
            g = np.frombuffer(bytes(m.data), np.int8).reshape(m.info.height, m.info.width)
            with self.lock:
                self.cost = (g, m.info.origin.position.x, m.info.origin.position.y, m.info.resolution)

        def on_stats(m):
            s, now = json.loads(m.data), time.time()
            with self.lock:
                if self.stats and s["stamp"] > self.stats["stamp"]:
                    self.rate = 0.8 * self.rate + 0.2 / (s["stamp"] - self.stats["stamp"])
                self.stats, self.t_stats = s, now

        def on_odom(m):
            q = m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            with self.lock:
                self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw)
                self.speed = math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y)

        def on_plan(m):
            p = np.array([[s.pose.position.x, s.pose.position.y] for s in m.poses], np.float32).reshape(-1, 2)
            with self.lock:
                self.plan = p

        for z in cfg["zones"]:
            n.create_subscription(PointCloud2, f"/adaptmap/cells/{z['name']}", lambda m, k=z["name"]: on_cells(k, m), 1)
        n.create_subscription(OccupancyGrid, "/adaptmap/costmap", on_cost, 1)
        n.create_subscription(String, "/adaptmap/stats", on_stats, 10)
        n.create_subscription(Odometry, "/odom", on_odom, be)
        n.create_subscription(PathMsg, "/plan", on_plan, 1)
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(n)
        threading.Thread(target=self.ex.spin, daemon=True).start()
        self.rclpy = rclpy

    def connected(self):
        return time.time() - self.t_stats < STALE

    def snapshot(self):
        with self.lock:
            return dict(cells=dict(self.cells), cost=self.cost, stats=self.stats, plan=self.plan,
                        pose=self.pose, speed=self.speed, rate=self.rate)


class Scenarios:
    """Runs one scenario at a time on a background thread with nav_scenarios.Runner."""

    def __init__(self, world):
        self.world, self.lock = world, threading.Lock()
        self.list, self.hist = scenarios(world)
        self.state = {"name": None, "phase": "idle", "goal": None, "progress": {}, "result": None}
        self.session = []
        self.runner = None
        self.thread = None

    def start(self, name):
        sc = next((s for s in self.list if s[0] == name), None)
        if sc is None:
            return "unknown scenario"
        with self.lock:
            if self.thread and self.thread.is_alive():
                return "a scenario is already running"
            self.state = {"name": name, "phase": "placing", "goal": list(sc[2]), "expect": sc[3],
                          "description": sc[5], "progress": {}, "result": None, "t0": time.time()}
            self.thread = threading.Thread(target=self._run, args=(sc,), daemon=True)
            self.thread.start()
        return None

    def stop(self):
        if self.runner:
            self.runner.abort = True

    def _run(self, sc):
        import rclpy
        from nav_scenarios import ROAD_HALF, ROBOT_TOP, Runner, obstacles
        name, (sx, sy, syaw), (gx, gy), expect, limit, why = sc
        try:
            if self.runner is None:
                self.runner = Runner()
            r = self.runner
            t0 = time.time()
            while (not r.client.wait_for_server(timeout_sec=0.5) or r.pose is None) and time.time() - t0 < 30:
                rclpy.spin_once(r, timeout_sec=0.1)
            if r.pose is None:
                raise RuntimeError("no odometry / Nav2 not up")
            manifest = json.loads((WORLDS / f"{self.world}_manifest.json").read_text())
            stat, acts = obstacles(manifest, manifest.get("robot_top", ROBOT_TOP))
            r.place(sx, sy, syaw)
            self.state["phase"] = "running"
            res = r.run(name, gx, gy, expect, limit, stat, acts, manifest.get("road_half", ROAD_HALF),
                        progress=lambda p: self.state.__setitem__("progress", p))
            res["aborted"] = r.abort
            self.state.update(phase="done", result={k: v for k, v in res.items() if k != "track"})
            self.session.append({"scenario": name, "passed": res["passed"], "aborted": r.abort})
        except Exception as e:  # noqa: BLE001 - shown on the page
            self.state.update(phase="error", result={"error": str(e)})


# ------------------------------------------------------------------ rendering
def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


FONT, SMALL = _font(18), _font(14)


def render(snap, view, cfg, goal):
    """Robot-centred top-down image (x right, y up). view: 'map' (semantic classes) or 'planner' (costmap)."""
    cx, cy, yaw = snap["pose"]
    x0, y0 = cx - HALF, cy - HALF
    img = np.tile(BG, (NPX, NPX, 1))
    if view == "planner" and snap["cost"] is not None:
        g, ox, oy, res = snap["cost"]
        c = x0 + (np.arange(NPX) + 0.5) * PX
        r = y0 + (np.arange(NPX)[::-1] + 0.5) * PX
        gi = np.floor((c - ox) / res).astype(int)
        gj = np.floor((r - oy) / res).astype(int)
        okx, oky = (gi >= 0) & (gi < g.shape[1]), (gj >= 0) & (gj < g.shape[0])
        v = np.full((NPX, NPX), -1, np.int16)
        v[np.ix_(oky, okx)] = g[np.ix_(gj[oky], gi[okx])]
        free = (v >= 0) & (v < 50)
        img[free] = (np.array([46, 52, 60]) + (v[free, None] / 50.0) * np.array([110, 60, -20])).astype(np.uint8)
        img[(v >= 50) & (v < 99)] = (214, 120, 40)
        img[v >= 99] = (235, 64, 64)
    else:
        for z in reversed(cfg["zones"]):  # coarse first so finer zones draw on top
            if z["name"] not in snap["cells"]:
                continue
            xyz, rgb = snap["cells"][z["name"]]
            if len(xyz) == 0:
                continue
            k = int(np.ceil(z["cell"] / PX - 1e-6))
            i = np.floor((xyz[:, 0] - z["cell"] / 2 - x0) / PX).astype(int)
            j = np.floor((xyz[:, 1] - z["cell"] / 2 - y0) / PX).astype(int)
            for di in range(k):
                for dj in range(k):
                    ii, jj = i + di, j + dj
                    ok = (ii >= 0) & (ii < NPX) & (jj >= 0) & (jj < NPX)
                    img[NPX - 1 - jj[ok], ii[ok]] = rgb[ok]

    im = Image.fromarray(img).convert("RGBA")
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    P = lambda x, y: ((x - x0) / PX, NPX - (y - y0) / PX)  # noqa: E731

    # zone outlines (current size: the 5 cm zone grows with speed)
    st = snap["stats"] or {}
    halves = st.get("zone_halves") or [z["half_extent"] for z in cfg["zones"]]
    for z, h in zip(cfg["zones"], halves):
        if h > HALF * 1.45:
            continue
        a, b = P(cx - h, cy + h), P(cx + h, cy - h)
        d.rectangle([a, b], outline=(255, 255, 255, 150), width=2)
        d.text((a[0] + 6, a[1] + 4), f"{int(round(z['cell'] * 100))} cm cells", fill=(255, 255, 255, 220), font=SMALL)

    # Nav2 path and goal
    if snap["plan"] is not None and len(snap["plan"]) > 1 and goal is not None:
        d.line([P(*p) for p in snap["plan"]], fill=(80, 220, 255, 255), width=4)
    if goal is not None:
        gx, gy = P(*goal)
        d.ellipse([gx - 11, gy - 11, gx + 11, gy + 11], outline=(90, 240, 120, 255), width=4)
        d.ellipse([gx - 4, gy - 4, gx + 4, gy + 4], fill=(90, 240, 120, 255))

    # walking pedestrians: where they are and where AdaptMap predicts they will be
    horizon = cfg["terrain"].get("motion", {}).get("horizon", 2.5)
    rad = cfg["terrain"].get("motion", {}).get("radius", 0.6)
    labels = []
    for px_, py_, vx, vy in st.get("moving", []):
        a, b = P(px_, py_), P(px_ + vx * horizon, py_ + vy * horizon)
        d.line([a, b], fill=(255, 70, 70, 90), width=max(3, int(2 * rad / PX)))
        d.line([a, b], fill=(255, 90, 90, 255), width=3)
        ang = math.atan2(-(b[1] - a[1]), b[0] - a[0])
        for s in (2.6, -2.6):
            d.line([b, (b[0] + 14 * math.cos(ang + s), b[1] - 14 * math.sin(ang + s))], fill=(255, 90, 90, 255), width=3)
        d.ellipse([a[0] - 8, a[1] - 8, a[0] + 8, a[1] + 8], fill=(255, 40, 40, 255), outline=(255, 255, 255, 255))
        if all(math.hypot(a[0] - l[0], a[1] - l[1]) > 40 for l in labels):  # one label per person
            d.text((min(max(a[0] + 12, 4), NPX - 130), min(max(a[1] - 22, 4), NPX - 20)), f"walking {math.hypot(vx, vy):.1f} m/s", fill=(255, 220, 220, 255), font=SMALL)
            labels.append(a)

    # the rover's footprint
    hl, hw = cfg["terrain"].get("robot_half_size", (0.495, 0.335))
    c, s_ = math.cos(yaw), math.sin(yaw)
    corners = [P(cx + c * a - s_ * b, cy + s_ * a + c * b) for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
    d.polygon(corners, fill=(255, 255, 255, 255), outline=(20, 20, 20, 255))
    tip = P(cx + c * (hl + 1.2), cy + s_ * (hl + 1.2))
    d.line([P(cx, cy), tip], fill=(255, 255, 255, 255), width=3)

    # scale bar
    d.rectangle([20, NPX - 30, 20 + 10 / PX, NPX - 24], fill=(255, 255, 255, 230))
    d.text((20, NPX - 52), "10 m", fill=(255, 255, 255, 230), font=SMALL)
    im = Image.alpha_composite(im, over).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return buf.getvalue()


def blank(text):
    im = Image.new("RGB", (NPX, NPX), tuple(int(v) for v in BG))
    d = ImageDraw.Draw(im)
    w = d.textlength(text, font=FONT)
    d.text(((NPX - w) / 2, NPX / 2 - 10), text, fill=(200, 205, 215), font=FONT)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return buf.getvalue()


# ------------------------------------------------------------------ captions
def caption(live, sc):
    """One plain sentence about what the robot is doing now."""
    st = sc.state
    if live is None or not live.connected():
        return "The simulator is not running: showing the recorded results and video."
    snap = live.snapshot()
    x, y, _ = snap["pose"]
    near = [m for m in (snap["stats"] or {}).get("moving", []) if math.hypot(m[0] - x, m[1] - y) < 15]
    if st["phase"] == "placing":
        return "Placing the robot at the start and letting the map fill in…"
    if st["phase"] == "error":
        return "The scenario could not run: " + st["result"]["error"]
    if st["phase"] == "done":
        r = st["result"]
        if r.get("aborted"):
            return "Stopped."
        if r["reached"] and r["passed"]:
            return f"Goal reached in {r['time_s']:.0f} s, {r['distance_m']:.0f} m driven, nothing touched."
        if not r["expected_success"] and r["passed"]:
            return "Correctly refused: the only way to that goal is over a curb the robot must not climb."
        why = ("it touched " + str(r["closest_static"])) if (r["min_clearance_static_m"] or 1) <= 0 else \
              "it left the road" if r["left_road"] else "it drove into a pedestrian" if r["drove_into_pedestrian"] else \
              "it did not reach the goal in time"
        return f"Failed: {why}."
    if st["phase"] == "running":
        if snap["speed"] < 0.05 and near:
            return "Waiting: a pedestrian's predicted path (red arrow) crosses the route, so the robot holds back."
        if near:
            return f"Driving at {snap['speed']:.1f} m/s, keeping clear of {len(near)} walking pedestrian(s) and their predicted paths."
        if snap["speed"] < 0.05:
            return "Planning a route on the AdaptMap costmap…"
        return f"Driving to the goal at {snap['speed']:.1f} m/s along the planned path (blue)."
    if near:
        return f"Idle. AdaptMap is tracking {len(near)} walking pedestrian(s) and predicting their paths."
    return "Idle. Pick a scenario on the right to watch the robot drive."


# ------------------------------------------------------------------ web app
def make_app(cfg, world, live):
    app = Flask(__name__, static_folder=None)
    sc = Scenarios(world)
    cache, cache_lock = {}, threading.Lock()

    def frame(view):
        with cache_lock:
            t, jpg = cache.get(view, (0, None))
            if time.time() - t < 0.25:
                return jpg
        if live is None or not live.connected():
            jpg = blank("Simulator not running")
        else:
            jpg = render(live.snapshot(), view, cfg, sc.state["goal"] if sc.state["phase"] in ("running", "placing") else None)
        with cache_lock:
            cache[view] = (time.time(), jpg)
        return jpg

    @app.route("/")
    def index():
        return send_from_directory(Path(__file__).parent, "dashboard.html")

    @app.route("/api/summary")
    def api_summary():
        s = summary(cfg, world)
        s["scenarios"] = [{"name": n, "description": why, "expect": "reach" if e else "refuse", "history": sc.hist[n]}
                          for n, _, _, e, _, why in sc.list]
        s["classes"] = [{"name": n.replace("_", " "), "rgb": [int(v) for v in classes.COLORS[i]]}
                        for i, n in enumerate(classes.NAMES) if i > 0]
        return jsonify(s)

    @app.route("/api/live")
    def api_live():
        out = {"connected": live is not None and live.connected(), "caption": caption(live, sc),
               "scenario": {k: v for k, v in sc.state.items() if k != "t0"}, "session": sc.session}
        if out["connected"]:
            snap = live.snapshot()
            s = snap["stats"]
            out.update(speed=round(snap["speed"], 2), scans_per_s=round(snap["rate"], 1),
                       pipeline_ms=s["timing_ms"].get("total"), near_radius=s["zone_halves"][0],
                       observed=sum(s["occupied"].values()), cells=s["total_cells"], walking=len(s.get("moving", [])))
        p = out["scenario"].get("progress") or {}
        out["scenario"]["progress"] = {k: (None if isinstance(v, float) and math.isinf(v) else
                                           round(v, 2) if isinstance(v, float) else v) for k, v in p.items()}
        return jsonify(out)

    @app.route("/stream/<view>.mjpg")
    def stream(view):
        if view not in ("map", "planner"):
            abort(404)

        def gen():
            while True:
                jpg = frame(view)
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.25)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/frame/<view>.jpg")
    def single_frame(view):
        if view not in ("map", "planner"):
            abort(404)
        return Response(frame(view), mimetype="image/jpeg")

    @app.route("/api/run/<name>", methods=["POST"])
    def run(name):
        if live is None or not live.connected():
            return jsonify({"error": "the simulator is not running"}), 409
        err = sc.start(name)
        return (jsonify({"error": err}), 409) if err else jsonify({"ok": True})

    @app.route("/api/stop", methods=["POST"])
    def stop():
        sc.stop()
        return jsonify({"ok": True})

    @app.route("/media/<path:name>")
    def media(name):
        return send_from_directory(FIG, name)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="urban", help="world the sim runs: urban (demo) or tier1 ... tier5")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--offline", action="store_true", help="recorded results only, no ROS")
    a = ap.parse_args()
    cfg = load_config()
    live = None
    if not a.offline:
        try:
            import rclpy
            rclpy.init()
            live = Live(cfg)
        except ImportError as e:
            print(f"ROS not available ({e}); recorded results only. Source /opt/ros/humble/setup.bash for live mode.")
    print(f"AdaptMap dashboard: http://localhost:{a.port}", flush=True)
    make_app(cfg, a.world, live).run(host="127.0.0.1", port=a.port, threaded=True)


if __name__ == "__main__":
    main()
