"""See the world the way the robot does: raw LiDAR points vs the AdaptMap 2.5D map, in 3D.

Each map cell is drawn as a column whose height is the cell's stored elevation, so road is a flat
tile, a curb is a short step, cars and walls stand up as blocks. That is "2.5D": one height (and one
label) per ground tile. Tiles are small near the robot and grow with distance (the foveated grid);
switch the colouring to "tile size" and use the top-down view to see the zones.

Run while the sim + adaptmap_node are up (ROS sourced):

  Interactive window with a control/legend panel on the right:
    .venv/bin/python tools/view25d.py

  Record a side-by-side explainer video while you drive (Ctrl-C to stop early):
    .venv/bin/python tools/view25d.py --video data/drive_25d.mp4 [--seconds 60] [--color zone] [--top]
"""
import argparse
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from PIL import Image, ImageDraw, ImageFont
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "ros2_ws" / "src" / "adaptmap_ros")]
from adaptmap import classes  # noqa: E402
from adaptmap.config import load_config  # noqa: E402
from adaptmap_ros.adaptmap_node import cloud_to_numpy, quat_to_rot  # noqa: E402

R = o3d.visualization.rendering
LIDAR_Z = 1.73
FLAT = 0.15            # cells lower than this are drawn as flat tiles (no side walls)
BG = (0.07, 0.07, 0.09)
RAW_RGB = np.array([0.85, 0.85, 0.85])
# one colour per zone, finest first (categorical slots: blue, aqua, yellow, orange)
ZONE_RGB = np.array([[42, 120, 214], [27, 175, 122], [237, 161, 0], [235, 104, 52]]) / 255.0
# traversability: one-hue sequential ramp (pale = free, dark orange = costly) + a distinct lethal colour
COST_LO, COST_HI = np.array([253, 232, 214]) / 255.0, np.array([194, 65, 12]) / 255.0
LETHAL_RGB = np.array([120, 20, 30]) / 255.0
UNKNOWN_RGB = np.array([90, 90, 96]) / 255.0


def cost_rgb(cost):
    """OccupancyGrid costs (-1 unknown, 0..99, 100 lethal) -> RGB."""
    t = np.clip(cost, 0, 99)[:, None] / 99.0
    rgb = COST_LO + (COST_HI - COST_LO) * t
    rgb[cost >= 100] = LETHAL_RGB
    rgb[cost < 0] = UNKNOWN_RGB
    return rgb
FONT = "DejaVuSans.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


# ---------------------------------------------------------------- data
class Feed(Node):
    """Latest raw scan (world frame), latest cells per zone, robot pose, stats."""

    def __init__(self, zones):
        super().__init__("adaptmap_view25d")
        self.lock = threading.Lock()
        self.cells, self.raw, self.pose, self.stats, self.version = {}, None, None, None, 0
        latest = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        for i, z in enumerate(zones):
            self.create_subscription(PointCloud2, f"/adaptmap/cells/{z['name']}", lambda m, i=i, z=z: self.on_cells(i, z, m), 1)
        self.create_subscription(PointCloud2, "/lidar/points", self.on_raw, latest)
        self.create_subscription(Odometry, "/odom", self.on_odom, latest)
        self.create_subscription(String, "/adaptmap/stats", lambda m: setattr(self, "stats", m.data), 1)
        self.cost = None  # (grid [ix, iy], origin xy, resolution) from /adaptmap/costmap
        self.create_subscription(OccupancyGrid, "/adaptmap/costmap", self.on_cost, 1)

    def on_cost(self, m):
        a = np.frombuffer(bytes(m.data), np.int8).reshape(m.info.height, m.info.width).T  # -> [ix, iy]
        with self.lock:
            self.cost = (a.copy(), (m.info.origin.position.x, m.info.origin.position.y), m.info.resolution)

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        with self.lock:
            self.pose = (np.array([p.x, p.y, p.z], np.float32), quat_to_rot(q))

    def on_raw(self, m):
        if self.pose is None:
            return
        p = cloud_to_numpy(m)[:, :3]
        p = p[np.isfinite(p).all(1)]
        t, Rm = self.pose
        with self.lock:
            self.raw = (p + [0, 0, LIDAR_Z]) @ Rm.T + t

    def on_cells(self, idx, zone, m):
        a = np.frombuffer(m.data, np.float32).reshape(-1, 4)
        rgb = a[:, 3].copy().view(np.uint32)
        col = np.stack([(rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255], 1) / 255.0
        with self.lock:
            self.cells[zone["name"]] = (a[:, :3].copy(), col, zone["cell"], idx)
            self.version += 1

    def snapshot(self):
        with self.lock:
            return dict(self.cells), self.raw, self.pose, self.stats, self.version


# ---------------------------------------------------------------- geometry
def cell_costs(xyz, cost):
    """Look up the costmap value under each cell centre (-1 where the costmap has no data)."""
    if cost is None:
        return np.full(len(xyz), -1, np.int8)
    grid, (ox, oy), res = cost
    i = ((xyz[:, 0] - ox) / res).astype(int)
    j = ((xyz[:, 1] - oy) / res).astype(int)
    ok = (i >= 0) & (i < grid.shape[0]) & (j >= 0) & (j < grid.shape[1])
    out = np.full(len(xyz), -1, np.int8)
    out[ok] = grid[i[ok], j[ok]]
    return out


def columns_mesh(cells, color_by="class", gap=None, cost=None):
    """One flat-shaded column per cell: top face always, side faces only for raised cells.

    color_by: "class" (object type), "zone" (tile size) or "cost" (traversability). gap: fraction of the tile left empty
    around each tile so the grid itself is visible (defaults: 4% for class, 22% for zone colouring).
    """
    if gap is None:
        gap = 0.22 if color_by == "zone" else 0.04
    V, T, C = [], [], []
    n0 = 0
    for xyz, col, size, zi in cells.values():
        if not len(xyz):
            continue
        if color_by == "zone":
            col = np.broadcast_to(ZONE_RGB[zi], col.shape)
        elif color_by == "cost":
            col = cost_rgb(cell_costs(xyz, cost))
        h = size * (1 - gap) / 2
        top = np.maximum(xyz[:, 2], 0.02)
        x0, x1, y0, y1 = xyz[:, 0] - h, xyz[:, 0] + h, xyz[:, 1] - h, xyz[:, 1] + h
        quads = [(np.stack([np.stack([x0, y0, top], 1), np.stack([x1, y0, top], 1),
                            np.stack([x1, y1, top], 1), np.stack([x0, y1, top], 1)], 1), col)]
        tall = top > FLAT
        if tall.any():
            b = np.zeros(tall.sum(), np.float32)
            X0, X1, Y0, Y1, Z = x0[tall], x1[tall], y0[tall], y1[tall], top[tall]
            dark = col[tall] * 0.62  # side faces darker than tops -> reads as 3D
            for a, c in (((X0, Y0), (X1, Y0)), ((X1, Y0), (X1, Y1)), ((X1, Y1), (X0, Y1)), ((X0, Y1), (X0, Y0))):
                quads.append((np.stack([np.stack([a[0], a[1], b], 1), np.stack([c[0], c[1], b], 1),
                                        np.stack([c[0], c[1], Z], 1), np.stack([a[0], a[1], Z], 1)], 1), dark))
        for q, qc in quads:
            n = q.shape[0]
            V.append(q.reshape(-1, 3))
            C.append(np.repeat(qc, 4, axis=0))
            base = n0 + np.arange(n)[:, None] * 4
            T.append(np.concatenate([base + [0, 1, 2], base + [0, 2, 3]], 0))
            n0 += 4 * n
    m = o3d.geometry.TriangleMesh()
    if V:
        m.vertices = o3d.utility.Vector3dVector(np.concatenate(V).astype(np.float64))
        m.triangles = o3d.utility.Vector3iVector(np.concatenate(T).astype(np.int32))
        m.vertex_colors = o3d.utility.Vector3dVector(np.clip(np.concatenate(C), 0, 1))
    return m


def raw_cloud(raw):
    pc = o3d.geometry.PointCloud()
    if raw is not None and len(raw):
        pc.points = o3d.utility.Vector3dVector(raw.astype(np.float64))
        pc.paint_uniform_color(RAW_RGB)
    return pc


def live_zones(zones, stats):
    """Zones with the radii the node is currently using (speed-adaptive), from /adaptmap/stats."""
    import json
    try:
        halves = json.loads(stats)["zone_halves"]
        return [dict(z, half_extent=h) for z, h in zip(zones, halves)]
    except (TypeError, ValueError, KeyError):
        return zones


def zone_outlines(zones, center):
    """Ground-level squares marking where each tile size starts/ends (map is centred on the robot)."""
    pts, lines, cols = [], [], []
    for i, z in enumerate(zones):
        h = z["half_extent"]
        c = [[center[0] - h, center[1] - h], [center[0] + h, center[1] - h],
             [center[0] + h, center[1] + h], [center[0] - h, center[1] + h]]
        k = len(pts)
        pts += [[x, y, 0.08] for x, y in c]
        lines += [[k, k + 1], [k + 1, k + 2], [k + 2, k + 3], [k + 3, k]]
        cols += [ZONE_RGB[i]] * 4
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.array(pts, float)), o3d.utility.Vector2iVector(np.array(lines)))
    ls.colors = o3d.utility.Vector3dVector(np.array(cols, float))
    return ls


def zone_label_positions(zones, center):
    """Where to put each zone's text label: on the near-left corner of its square."""
    return [(np.array([center[0] - z["half_extent"] + 1.0, center[1] + z["half_extent"] - 1.0, 0.3]),
             f"{int(round(z['cell'] * 100))} cm tiles ({prev:g}-{z['half_extent']:g} m)")
            for z, prev in zip(zones, [0] + [zz["half_extent"] for zz in zones[:-1]])]


def robot_marker(pose):
    t, Rm = pose
    m = o3d.geometry.TriangleMesh.create_box(1.0, 0.7, 0.4)
    m.translate([-0.5, -0.35, 0.05])
    arrow = o3d.geometry.TriangleMesh.create_cone(0.35, 0.9)
    arrow.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi / 2, 0]), center=[0, 0, 0])
    arrow.translate([0.5, 0, 0.3])
    m += arrow
    m.rotate(Rm.astype(np.float64), center=[0, 0, 0])
    m.translate(t.astype(np.float64))
    m.paint_uniform_color([1.0, 1.0, 1.0])
    m.compute_vertex_normals()
    return m


def camera(pose, view):
    """(look_at, eye, up) for 'chase' (behind and above the robot) or 'top' (straight down, 220 m wide)."""
    t, Rm = pose
    fwd = np.array([Rm[0, 0], Rm[1, 0], 0.0])
    fwd /= max(np.linalg.norm(fwd), 1e-6)
    if view == "top":
        return t.astype(float), t + np.array([0, 0, 190.0]), fwd
    return t + 8 * fwd, t - 20 * fwd + [0, 0, 14], np.array([0, 0, 1.0])


def legend_image(color_by, zones, width=300):
    """Legend panel (PIL) for the GUI side bar and the video."""
    f_h, f = font(17, True), font(15)
    rows = [("h", "Object types (colour)")] + [("c", i) for i in range(1, classes.NUM_CLASSES)] + \
           [("s", None), ("h", "Tile size (zones)")] + [("z", i) for i in range(len(zones))] + \
           [("s", None), ("h", "Traversability (cost)")] + [("k", v) for v in (0, 50, 99, 100)] + \
           [("s", None), ("t", "Tiles are small near the robot and"), ("t", "grow with distance. Switch colour to"),
            ("t", "'tile size' + top view to see zones.")]
    img = Image.new("RGB", (width, 26 * len(rows) + 16), (24, 24, 28))
    d = ImageDraw.Draw(img)
    y = 10
    for kind, v in rows:
        if kind == "h":
            d.text((12, y), v, fill=(245, 245, 245), font=f_h)
        elif kind == "c":
            col = tuple(int(c) for c in classes.COLORS[v])
            d.rectangle([14, y + 3, 34, y + 19], fill=col, outline=(200, 200, 200))
            d.text((44, y + 1), classes.NAMES[v].replace("_", " / "), fill=(230, 230, 230), font=f)
        elif kind == "k":
            col = tuple(int(c * 255) for c in cost_rgb(np.array([v]))[0])
            label = {0: "free (e.g. road)", 50: "costly (grass, slope, rough)", 99: "very costly",
                     100: "blocked (obstacle, curb, steep)"}[v]
            d.rectangle([14, y + 3, 34, y + 19], fill=col, outline=(200, 200, 200))
            d.text((44, y + 1), label, fill=(230, 230, 230), font=f)
        elif kind == "z":
            z = zones[v]
            prev = 0 if v == 0 else zones[v - 1]["half_extent"]
            col = tuple(int(c * 255) for c in ZONE_RGB[v])
            d.rectangle([14, y + 3, 34, y + 19], fill=col, outline=(200, 200, 200))
            d.text((44, y + 1), f"{int(round(z['cell'] * 100))} cm tiles, {prev:g}-{z['half_extent']:g} m away",
                   fill=(230, 230, 230), font=f)
        elif kind == "t":
            d.text((12, y + 1), v, fill=(170, 170, 170), font=f)
        y += 26
    return img


# ---------------------------------------------------------------- interactive (Open3D GUI)
class Viewer:
    PANEL_W = 320

    def __init__(self, feed, zones, selftest=None):
        gui = o3d.visualization.gui
        self.gui, self.feed, self.zones = gui, feed, zones
        self.show, self.color_by, self.view, self.follow, self.outlines = "map", "class", "chase", True, True
        self.last_v, self.pending, self.first, self.selftest, self.frames = -1, False, True, selftest, 0
        app = gui.Application.instance
        app.initialize()
        self.win = app.create_window("AdaptMap - how the robot sees the world", 1700, 980)
        em = self.win.theme.font_size
        self.scene = gui.SceneWidget()
        self.scene.scene = R.Open3DScene(self.win.renderer)
        self.scene.scene.set_background([*BG, 1.0])
        self.scene.scene.show_axes(False)
        self.mat_mesh = R.MaterialRecord(); self.mat_mesh.shader = "defaultUnlit"
        self.mat_pts = R.MaterialRecord(); self.mat_pts.shader = "defaultUnlit"; self.mat_pts.point_size = 2.0
        self.mat_line = R.MaterialRecord(); self.mat_line.shader = "unlitLine"; self.mat_line.line_width = 3.0
        self.mat_robot = R.MaterialRecord(); self.mat_robot.shader = "defaultUnlit"
        self.labels = []

        p = gui.Vert(0.4 * em, gui.Margins(0.6 * em, 0.6 * em, 0.6 * em, 0.6 * em))
        p.add_child(gui.Label("Show"))
        show = gui.RadioButton(gui.RadioButton.VERT)
        show.set_items(["2.5D map (what AdaptMap keeps)", "Raw LiDAR dots (what the sensor sends)", "Both overlaid"])
        show.set_on_selection_changed(lambda i: self.set(show=["map", "raw", "both"][i]))
        p.add_child(show)
        p.add_fixed(0.5 * em)
        p.add_child(gui.Label("Colour tiles by"))
        colr = gui.RadioButton(gui.RadioButton.VERT)
        colr.set_items(["Object type", "Tile size (shows the zones)", "Traversability (robot's cost)"])
        colr.set_on_selection_changed(lambda i: self.set(color_by=["class", "zone", "cost"][i]))
        p.add_child(colr)
        p.add_fixed(0.5 * em)
        p.add_child(gui.Label("Camera"))
        cam = gui.Horiz(0.4 * em)
        b1, b2 = gui.Button("Chase"), gui.Button("Top-down")
        b1.set_on_clicked(lambda: self.set(view="chase", follow=True))
        b2.set_on_clicked(lambda: self.set(view="top", follow=True))
        cam.add_child(b1); cam.add_child(b2)
        p.add_child(cam)
        fol = gui.Checkbox("Follow robot (untick to orbit freely)")
        fol.checked = True
        fol.set_on_checked(lambda c: self.set(follow=c))
        p.add_child(fol)
        out = gui.Checkbox("Show zone outlines + labels")
        out.checked = True
        out.set_on_checked(lambda c: self.set(outlines=c))
        p.add_child(out)
        p.add_fixed(0.5 * em)
        self.legend_widget = gui.ImageWidget(o3d.geometry.Image(np.asarray(legend_image("class", zones))))
        p.add_child(self.legend_widget)
        self.status = gui.Label("waiting for the map ...")
        p.add_child(self.status)
        self.panel = p
        self.win.add_child(self.scene)
        self.win.add_child(p)
        self.win.set_on_layout(self.layout)
        threading.Thread(target=self.updater, daemon=True).start()

    def layout(self, ctx):
        r = self.win.content_rect
        self.scene.frame = self.gui.Rect(r.x, r.y, r.width - self.PANEL_W, r.height)
        self.panel.frame = self.gui.Rect(r.get_right() - self.PANEL_W, r.y, self.PANEL_W, r.height)

    def set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.last_v = -1  # force a redraw with the new settings

    def updater(self):
        while True:
            time.sleep(0.15)
            cells, raw, pose, stats, ver = self.feed.snapshot()
            if pose is None or not cells or ver == self.last_v or self.pending:
                continue
            self.last_v = ver
            geo = {"robot": (robot_marker(pose), self.mat_robot)}
            if self.show in ("map", "both"):
                geo["map"] = (columns_mesh(cells, self.color_by, cost=self.feed.cost), self.mat_mesh)
            if self.show in ("raw", "both"):
                geo["raw"] = (raw_cloud(raw), self.mat_pts)
            if self.outlines:
                geo["zones"] = (zone_outlines(live_zones(self.zones, stats), pose[0]), self.mat_line)
            n_cells = sum(len(c[0]) for c in cells.values())
            n_raw = 0 if raw is None else len(raw)
            self.pending = True
            self.gui.Application.instance.post_to_main_thread(self.win, lambda: self.apply(geo, pose, n_cells, n_raw))

    def apply(self, geo, pose, n_cells, n_raw):
        sc = self.scene.scene
        sc.clear_geometry()
        for name, (g, mat) in geo.items():
            sc.add_geometry(name, g, mat)
        for lab in self.labels:
            self.scene.remove_3d_label(lab)
        self.labels = []
        if self.outlines:
            for pos, text in zone_label_positions(live_zones(self.zones, self.feed.stats), pose[0]):
                self.labels.append(self.scene.add_3d_label(pos, text))
        if self.first or self.follow:
            look, eye, up = camera(pose, self.view)
            if self.first:
                bb = o3d.geometry.AxisAlignedBoundingBox(pose[0] - 110, pose[0] + 110)
                self.scene.setup_camera(55.0, bb, look)
                self.first = False
            self.scene.look_at(look, eye, up)
        key = self.color_by
        if getattr(self, "_legend_key", None) != key:
            self.legend_widget.update_image(o3d.geometry.Image(np.asarray(legend_image(key, self.zones))))
            self._legend_key = key
        self.status.text = f"{n_cells:,} map tiles shown | {n_raw:,} raw dots in last scan"
        self.pending = False
        self.frames += 1
        if self.selftest and self.frames == 4:  # render what the window shows to a file, then quit
            def done(img):
                o3d.io.write_image(self.selftest, img)
                print(f"selftest image written to {self.selftest}", flush=True)
                self.quit()
            self.scene.scene.scene.render_to_image(done)

    def quit(self):
        self.gui.Application.instance.quit()

    def run(self):
        # Ctrl-C: Python handlers only run when the GUI thread is in Python, which happens on every
        # map update (apply), so route the signal into a clean application quit.
        signal.signal(signal.SIGINT, lambda *_: self.gui.Application.instance.post_to_main_thread(self.win, self.quit))
        self.gui.Application.instance.run()


# ---------------------------------------------------------------- video
class Recorder:
    W, H = 960, 640

    def __init__(self, zones, color_by, view):
        self.zones, self.color_by, self.view = zones, color_by, view
        self.cost = None
        self.r = R.OffscreenRenderer(self.W, self.H)
        self.r.scene.set_background([*BG, 1.0])
        self.unlit = R.MaterialRecord(); self.unlit.shader = "defaultUnlit"
        self.pts = R.MaterialRecord(); self.pts.shader = "defaultUnlit"; self.pts.point_size = 2.0
        self.line = R.MaterialRecord(); self.line.shader = "unlitLine"; self.line.line_width = 3.0
        self.f_big, self.f, self.f_s = font(26, True), font(19), font(16)
        self.legend = legend_image(color_by, zones, width=330)

    def view_img(self, geoms, pose):
        sc = self.r.scene
        sc.clear_geometry()
        for name, g, mat in geoms:
            sc.add_geometry(name, g, mat)
        look, eye, up = camera(pose, self.view)
        self.r.setup_camera(55.0, look, eye, up)
        return Image.fromarray(np.asarray(self.r.render_to_image()))

    def frame(self, cells, raw, pose):
        robot = robot_marker(pose)
        zl = zone_outlines(self.zones, pose[0])
        n_raw = 0 if raw is None else len(raw)
        left = self.view_img([("raw", raw_cloud(raw), self.pts), ("z", zl, self.line), ("robot", robot, self.unlit)], pose)
        right = self.view_img([("map", columns_mesh(cells, self.color_by, cost=self.cost), self.unlit), ("z", zl, self.line),
                               ("robot", robot, self.unlit)], pose)
        n_cells = sum(len(c[0]) for c in cells.values())
        lw = self.legend.width
        canvas = Image.new("RGB", (2 * self.W + 12 + lw + 12, max(self.H + 150, self.legend.height + 60)), (250, 250, 248))
        canvas.paste(left, (0, 60)); canvas.paste(right, (self.W + 12, 60)); canvas.paste(self.legend, (2 * self.W + 24, 60))
        d = ImageDraw.Draw(canvas)
        d.text((16, 14), "1. What the LiDAR sends (every 0.1 s)", fill=(20, 20, 20), font=self.f_big)
        d.text((self.W + 28, 14), "2. What AdaptMap keeps: a 2.5D map", fill=(20, 20, 20), font=self.f_big)
        d.text((2 * self.W + 28, 14), "Legend", fill=(20, 20, 20), font=self.f_big)
        y = self.H + 70
        d.text((16, y), f"{n_raw:,} separate 3D dots. Nothing says which dot is road, car or wall.", fill=(40, 40, 40), font=self.f)
        d.text((16, y + 26), "Coloured squares on the ground: where each tile size starts and ends.", fill=(40, 40, 40), font=self.f)
        d.text((self.W + 28, y), "The ground is cut into tiles; each stores ONE height and ONE label (2.5D):", fill=(40, 40, 40), font=self.f)
        d.text((self.W + 28, y + 26), "road is flat, curbs are steps, cars and walls stand up.", fill=(40, 40, 40), font=self.f)
        d.text((self.W + 28, y + 52), f"Tiles are 5 cm near the robot, growing to 50 cm far away -> {n_cells:,} tiles shown.",
               fill=(40, 40, 40), font=self.f)
        return canvas


def record(feed, zones, out, seconds, speed, color_by, view):
    rec = Recorder(zones, color_by, view)
    frames = Path(tempfile.mkdtemp(prefix="adaptmap_25d_"))
    n, t0, last_v = 0, time.time(), -1
    print(f"recording to {out} (Ctrl-C to stop)")
    try:
        while not seconds or time.time() - t0 < seconds:
            cells, raw, pose, _, ver = feed.snapshot()
            if pose is None or not cells or ver == last_v:
                time.sleep(0.05)
                continue
            last_v = ver
            rec.cost = feed.cost
            rec.frame(cells, raw, pose).save(frames / f"{n:05d}.png")
            n += 1
    except KeyboardInterrupt:
        pass
    elapsed = time.time() - t0
    fps = max(n / max(elapsed, 1e-6) * speed, 0.5)  # play back at the captured pace (x speed)
    print(f"{n} frames in {elapsed:.0f} s -> {fps:.2f} fps")
    if n:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", f"{fps:.3f}", "-i", str(frames / "%05d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], check=True)
        Image.open(sorted(frames.glob("*.png"))[-1]).save(Path(out).with_suffix(".png"))
        print(f"wrote {out} and {Path(out).with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="record a side-by-side explainer video instead of opening a window")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="video playback speed vs real time")
    ap.add_argument("--color", choices=["class", "zone", "cost"], default="class",
                    help="video: colour tiles by object type, tile size or traversability cost")
    ap.add_argument("--top", action="store_true", help="video: top-down camera instead of chase camera")
    ap.add_argument("--selftest", help=argparse.SUPPRESS)  # render the interactive scene to this file and exit
    a = ap.parse_args()
    zones = load_config()["zones"]
    rclpy.init()
    feed = Feed(zones)
    threading.Thread(target=rclpy.spin, args=(feed,), daemon=True).start()
    try:
        if a.video:
            record(feed, zones, a.video, a.seconds, a.speed, a.color, "top" if a.top else "chase")
        else:
            print("opening viewer; waiting for /adaptmap/cells/* and /odom (is the sim + adaptmap_node running?)")
            Viewer(feed, zones, a.selftest).run()
    except KeyboardInterrupt:
        pass
    finally:
        feed.destroy_node()
        rclpy.try_shutdown()
    # Open3D's GUI leaves native threads behind that abort the interpreter at exit; skip that teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
