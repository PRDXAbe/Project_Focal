# AdaptMap — Technical Disclosure

**Adaptive variable-resolution 2.5D semantic LiDAR mapping for dynamic environment perception**
Team LORENZINI · SIH26053 · prototype v0.1 (simulation)

> This is a technical description of the method, written to support a patent draft. It is not
> legal advice; the claim-style list in §8 is input for a patent agent, not drafted claims.

---

## 1. Problem

Ground robots and vehicles need a map of their surroundings that answers *where can I go, what
is there, and how tall is it* — in real time, on modest compute. Two common representations
fail at opposite ends:

- **Full 3D maps** (voxel grids, point clouds) keep everything but are too heavy to update and
  query at sensor rate on embedded hardware.
- **Uniform 2.5D grids** (one height per cell) are compact per cell, but a single resolution is
  either too coarse near the robot (where obstacles matter most) or wastes memory far away. A
  uniform 5 cm grid over a 200 × 200 m window needs 16,000,000 cells, most of which a LiDAR
  can never fill at range.

## 2. Pipeline overview

```
3D LiDAR scan (x, y, z, intensity, t)
   │
   ├─ 1. Preprocessing        invalid/NaN removal, range gate 0.5–100 m, ROI height window,
   │                          self-body removal, voxel-isolation outlier filter            (~2 ms)
   ├─ 2. Semantic segmentation sparse 3D CNN (MinkUNet-34, fp16) → per-point class probs  (~28 ms)
   ├─ 3. Distance zoning      Chebyshev distance from map centre → zone index per point;
   │                          zone radii adapt to vehicle speed (§3.8)
   ├─ 4. Foveated fusion      per-zone grid binning, decayed running fusion, ray clearing  (~7 ms)
   ├─ 5. Views                ground surface, height above ground, clearance, ring-gap filling
   ├─ 6. Terrain analysis     slope, step, roughness, curbs → traversability cost (§3.7)
   └─ 7. Outputs              per-zone 2.5D layers + a Nav2 costmap for path planning
```

Everything after the sensor runs inside one process on the GPU, so point clouds are never
copied or re-serialised between stages.

## 3. Foveated variable-resolution grid (core method)

### 3.1 Structure

The map is a stack of **nested square layers** centred on the robot. Layer *i* covers the
square ring between Chebyshev radii *Rᵢ₋₁* and *Rᵢ* with its own cell size *cᵢ*:

| Zone | Ring (Chebyshev distance) | Cell size | Cells in ring |
|---|---|---|---|
| near | 0 – 10 m | 5 cm | (20 m / 0.05)² = 160,000 |
| high | 10 – 30 m | 10 cm | (60² − 20²) / 0.1² = 320,000 |
| mid | 30 – 50 m | 25 cm | (100² − 60²) / 0.25² = 102,400 |
| far | 50 – 100 m | 50 cm | (200² − 100²) / 0.5² = 120,000 |
| **total** | 200 × 200 m window | | **702,400** |

A uniform 5 cm grid over the same window has **16,000,000** cells, so the foveated grid is
**22.8× smaller (−95.6 %)**.

*Correction to the proposal deck:* its table took zone areas as the full square (e.g. 10², 30²),
not the ring areas, and mixed a 100 × 100 m total with 100 m radii. The numbers above are the
corrected ring areas over a ±100 m window.

Each layer is stored as a dense GPU array with the inner (finer-owned) square masked out. Every
point is written to exactly one layer, so there is no duplication across resolutions, and
indexing is a single multiply-add (no tree traversal like a quadtree/octree).

### 3.2 Why these distances and cell sizes

Cell size is matched to the **sensor's own sampling density**. A 64-beam spinning LiDAR with
≈0.18° horizontal resolution places neighbouring returns this far apart:

| Range | Point spacing (0.18°) | Chosen cell |
|---|---|---|
| 10 m | 3.1 cm | 5 cm |
| 30 m | 9.4 cm | 10 cm |
| 50 m | 15.7 cm | 25 cm |
| 100 m | 31.4 cm | 50 cm |

Cells finer than the point spacing would mostly stay empty (wasted memory, no added
information), while coarser cells near the robot would hide small obstacles and curbs. So each
zone's cell is about one point-spacing at its outer edge, rounded to a value that divides the
coarsest cell exactly (needed for aligned rolling, §3.4). The near zone (0–10 m) is the
stopping and manoeuvring envelope of a ground robot at typical speeds, so it gets the finest
grid. Radii and cells are configuration values (`configs/adaptmap.yaml`), so they can be
re-tuned for other sensors, speeds or vehicles.

### 3.3 Per-cell contents (2.5D semantic elevation)

For every cell the map keeps a **decayed running average of per-scan observations**:

- `z_min` — lowest point height
- `z_max` — **elevation** (top surface; obstacle height)
- `z_mean`, `E[z²]` → height standard deviation (roughness / vertical structure)
- `p[C]` — fused class-probability vector (10 AdaptMap classes, fp16)
- `w` — observation weight (support), `t_last` — last update time
- `ground`, `w_g` — mean height of the cell's *ground-class* points, with its own weight
- `omin`, `w_o` — lowest *non-ground* point (bottom of whatever stands or hangs there), with its own weight

The outputs derived from these are: top class = argmax *p*, and
**confidence = max(p) · min(1, w / w_conf)**, which combines how sure the classifier is with
how often the cell was observed.

### 3.4 Update rule

For each scan with timestamp *t*:

1. **Rolling recentre.** The map centre is the robot position snapped to the coarsest cell
   (0.5 m). When it moves by *(dx, dy)* snap steps, each layer is shifted by
   *(dx, dy)·(0.5 / cᵢ)* cells and the newly exposed strips are cleared. Because every cᵢ divides
   0.5 m, all layers stay exactly aligned with each other and with the world.
2. **Class-aware temporal decay.** Every cell's weight decays as *w ← w · exp(−Δt / τ)*, where
   τ depends on the cell's current top class: τ_dynamic = 0.5 s for vehicles/pedestrians,
   τ_static = 20 s otherwise. Cells whose weight drops below *w_min* revert to unknown. Moving
   objects therefore fade within about a second instead of leaving "ghost trails", while static
   structure persists when briefly out of view.
3. **Zone assignment.** Chebyshev distance *d = max(|x − cₓ|, |y − c_y|)* picks the layer.
4. **Per-cell scan statistics.** These are GPU scatter reductions over the points of the scan
   in each cell: min, max, mean and mean-square height, plus the mean class-probability vector.
5. **Fusion.** For each observed cell, with *a = w / (w + 1)* (every scan counts as one
   unit-weight observation):
   *x ← a·x + (1 − a)·x_scan* for every stored statistic, *w ← min(w + 1, w_max)*.
   Capping the weight (w_max = 10) keeps the map responsive to change.

6. **Ray clearing (free-space evidence).** A subsample of the scan's rays (every 8th point,
   32 samples per ray, rays up to 50 m) is walked through the grid. A sample that lies *inside* a
   cell's stored obstacle span, more than 0.2 m above its bottom `omin` and more than 0.2 m below
   its top `z_max`, proves that space is now empty. This only counts in cells that received no
   points this scan, and never in the last 0.3 m before the ray's own end point. Each such cell's
   weight and obstacle weight are halved per scan; when the obstacle weight falls below *w_min*
   the cell is reset. Rays passing *over* an object or *under* an overhang (a tree crown) clear
   nothing.

Counting every scan as one observation, however many points it contributes, stops dense
near-range cells from becoming "sticky" compared with sparse far ones. Class-aware decay and ray
clearing are complementary: decay forgets what is no longer seen; clearing removes what is
seen to be gone.

### 3.4b Views: ground surface, clearance, hole filling

Views are computed from the stored statistics when published:

- **Ground surface.** Each cell's own ground height where it has ground points; otherwise the
  mean of ground cells within 2 m (integral-image box sums, any window size at the same cost).
  So a car's cells get the ground height of the road around it.
- **Height above ground** = `z_max` − ground. **Clearance** = `omin` − ground: the free height
  under a tree crown or overhang. This lets a 2.5D map say "drivable underneath", which a single
  height per cell normally can't.
- **Ring-gap filling.** Beyond ~10 m a LiDAR's rings hit the ground far apart: about 0.4 m at
  10 m, 1.6 m at 20 m, 3.6 m at 30 m. An unknown cell is filled with interpolated ground height and
  the majority ground class only when ground cells exist on **both opposite sides** of it within
  the zone's fill radius (0.3 / 1 / 2 / 3 m). The sides are taken as narrow bands in line with
  the cell, so this interpolates between rings but never extrapolates into shadows behind walls.
  Filled cells are flagged and get half confidence.

### 3.5 Optional compute-side foveation

Before segmentation, points can also be thinned per zone to one point per (cᵢ / 2) voxel
(`preprocess.adaptive_downsample`), which cuts network load at range by the same principle.
It is off by default because the pretrained network was trained on full-density scans.

### 3.6 Navigation output

`nav_view()` resamples all layers onto a single uniform grid at any resolution the planner
asks for. Wherever the output cell is coarser than the source layer, it is filled from the
**tallest** source cell, so thin obstacles (poles, pedestrians) are never averaged away. The
ROS node publishes:

- this grid, analysed for terrain (§3.7), as a Nav2 costmap (`nav_msgs/OccupancyGrid`);
- each zone as a `grid_map` with elevation, ground, height-above-ground, clearance, filled, std,
  class, confidence and colour layers, plus a terrain `grid_map`.

### 3.7 Terrain analysis and traversability

On the navigation grid (0.2 m, ±50 m):

- **slope** = gradient of the smoothed ground surface;
- **step** = largest height jump of the *drivable surface* to a neighbouring cell (the drivable
  surface is the ground under passable overhangs and the top surface elsewhere);
- **roughness** = local standard deviation of the drivable surface;
- **curb** = a 5–30 cm step where road and sidewalk cells meet.

The cost (0 free … 99, 100 lethal, −1 unknown) is a class prior (road 0, terrain 20, sidewalk 30,
vegetation 60), plus slope and roughness terms and a penalty for filled cells. It is made lethal
for: anything standing higher than the robot can climb (unless the clearance is higher than the
robot), slopes over the limit, unclimbable steps and curbs. All limits are robot parameters, so
the same map serves a rover (12 cm step, 1 m height) or a car.

### 3.8 Speed-adaptive foveation

The finest zone's radius follows the distance the vehicle needs to react and stop:
*R₀ = clamp(v·t_react + v²/(2a) + margin, R_base, R_max)*, on the 0.5 m snap grid. It grows
immediately when the speed rises and shrinks only after the target has stayed smaller for 3 s
(hysteresis). Resizing **resamples** the map instead of clearing it:

- where tiles become coarser, each new tile takes all its values from the *tallest* old tile it
  covers, so thin obstacles survive;
- where tiles become finer, each new tile copies the old tile it lies in.

At walking or rover speeds the 10 m base already covers the stopping envelope. At 8 m/s
(29 km/h) the 5 cm zone grows to 25 m: 1.33 M cells, still 12× fewer than a uniform 5 cm map.

## 4. Semantic segmentation

- **Network:** MinkUNet-34, a sparse 3D convolutional U-Net (spconv backend) on 5 cm voxels.
  Weights are pretrained on SemanticKITTI (69.3 mIoU on real validation data, mmdetection3d
  model zoo).
- **Why this network:** sparse convolution handles 100k+ point outdoor scans in tens of ms on a
  laptop GPU. Point-based networks such as PointNet++ are slower and less accurate on
  large outdoor scans.
- **Class mapping:** the 19 SemanticKITTI classes are summed into 10 AdaptMap classes in
  probability space, so confidence stays calibrated: road, sidewalk/curb, terrain, vegetation,
  building/wall, pole, vehicle, pedestrian, other obstacle, unknown.
- **Sim-to-real handling:** the simulated sensor copies the KITTI HDL-64 geometry (64 beams,
  +2°/−24.8° vertical FOV, 1.73 m mount). Intensity is replaced with a dataset-mean constant
  because simulated intensity is not physical.
- **Fine-tuning hook:** `tools/autolabel.py` labels simulated scans exactly from the world
  manifest (it agrees with the ray-caster's own labels on 97.7 % of points), and
  `tools/finetune.py` fine-tunes the network on them.

## 5. Prototype implementation

| Component | File |
|---|---|
| Foveated map engine | `adaptmap/foveated_map.py` |
| Preprocessing | `adaptmap/preprocess.py` |
| Segmentation wrapper | `adaptmap/segmentation.py` |
| Classes & mapping | `adaptmap/classes.py` |
| ROS 2 node (GridMap, markers, nav grid, stats) | `ros2_ws/src/adaptmap_ros/adaptmap_ros/adaptmap_node.py` |
| Gazebo world + rover | `tools/make_world.py`, `ros2_ws/src/adaptmap_sim/` |
| Analytic LiDAR simulator | `tools/synth_scan.py` |
| Benchmark | `tools/benchmark.py` → `docs/RESULTS.md` |
| Unit tests | `adaptmap/tests/` |

## 6. Measured results (RTX 4050 laptop)

Full numbers are in `docs/RESULTS.md`.

**Map efficiency.** Offline benchmark: 300 frames of a 10 Hz simulated drive with moving pedestrians, where
the same data is also fed to a uniform 5 cm map for comparison.

| Metric | Uniform 5 cm | AdaptMap | Gain |
|---|---|---|---|
| Cells | 16,000,000 | 702,400 | 22.8× |
| Map memory | 671 MB | 35 MB | 19× |
| Fusion time / scan | 22.4 ms | 3.7 ms | 6.1× |
| End-to-end latency (offline, GPU not shared) | — | 75 ms (13.3 Hz) | |

**Live ROS 2 + Gazebo run.** Gazebo, RViz and the renderer share the same GPU:
- the pipeline takes 144 ms per scan on average;
- 5.4 scans/s are integrated, and the node always takes the newest scan;
- peak GPU memory is 445 MB.

**Segmentation accuracy on Gazebo scans.** 131 scans, ground truth from the world manifest,
pretrained weights used zero-shot:
- point accuracy 89.2 %;
- IoU per class: road 93 %, building/wall 87 %, sidewalk/curb 84 %, vehicle 71 %, pole 35 %,
  pedestrian 23 %;
- terrain (grass) and generic crates are poorly separated from sidewalk and walls. The fine-tuning
  hook is meant to fix these classes.

## 7. Limitations of this prototype

- Simulation only. Odometry is ground truth from the simulator, with no localisation drift.
- No explicit free-space ray clearing; dynamic objects are removed by class-aware decay.
- Curbs come out as sidewalk-class height steps; no dedicated curb detector or traversability
  cost yet (next phase).
- Not yet run on embedded hardware (Jetson / TensorRT).
- Live rate (5.4 scans/s) is below the 10 Hz LiDAR rate when the simulator shares the GPU; the
  offline pipeline alone runs at 13 Hz.

## 8. Candidate novel elements (input for claims)

1. A 2.5D semantic elevation map made of nested, robot-centred square layers whose cell size
   grows with Chebyshev distance, where each point is written to exactly one layer.
2. Choosing each layer's cell size to match the LiDAR's angular sampling spacing at that
   layer's outer range.
3. Per-cell fusion of height statistics *and* a class-probability vector as a weight-decayed
   running average, where the decay time-constant is selected per cell from its current
   semantic class (fast for dynamic classes, slow for static ones).
4. A per-cell confidence that combines classifier certainty with observation support.
5. Rolling all layers in snap steps equal to the coarsest cell, with every finer cell size an
   integer divisor of it, so layers stay mutually aligned without resampling.
6. Producing a planner-resolution navigation grid from the multi-resolution layers by
   max-height pooling, which keeps thin obstacles visible.
7. Optional distance-zoned point thinning before segmentation, using the same zone schedule.
8. An end-to-end GPU pipeline (filter → sparse-CNN segmentation → foveated fusion) at sensor
   rate on laptop-class hardware.
