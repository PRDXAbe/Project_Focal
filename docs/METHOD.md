# AdaptMap — Technical Disclosure

**Adaptive variable-resolution 2.5D semantic LiDAR mapping for dynamic environment perception**
Team LORENZINI · SIH26053 · prototype v0.2 (simulation with Nav2, validated on real KITTI and SemanticPOSS data)

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
   ├─ 4. Foveated fusion      per-zone grid binning, decayed running fusion; ray clearing,
   │                          ground clearing, appearance, footprint clearing (§3.4)     (~7 ms)
   ├─ 5. Views                ground surface, height above ground, clearance, ring-gap filling
   ├─ 6. Terrain analysis     slope, step, roughness, curbs → traversability cost (§3.7)
   ├─ 7. Motion keep-out      pedestrian tracks → lethal corridor of the next 2.5 s (§3.9)
   └─ 8. Outputs              per-zone 2.5D layers + a Nav2 costmap, every scan
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
7. **Ground clearing.** A scan that sees *only ground points* in a cell whose stored top is more
   than 0.1 m above the scan's top contradicts the stored obstacle (the spot a pedestrian just
   left). Its obstacle weight is halved per such scan; once it falls below *w_min* the scan
   *replaces* the cell's statistics instead of being averaged in. Averaging alone (at ~1/w_max per
   scan) left a lethal residue on the road for seconds after a person walked on.
8. **Appearance.** The mirror rule: a scan whose non-ground top is more than 0.1 m above the
   stored top (something walked or drove into the cell) replaces the cell (*a = 0*) and restarts
   its weight. A new obstacle is therefore in the map after one scan; with plain averaging a
   person stepping onto a road cell that had full weight took ~1 s to show up, which put a
   crossing pedestrian ~1 m behind their true position.
9. **Footprint clearing.** Cells under the robot's own footprint are forgotten every scan. The
   robot stands there, and those cells lie in the LiDAR's blind disc (a roof-mounted sensor cannot
   see the ground within a few metres), where neither ray nor ground clearing can observe them; a
   misclassified glimpse of someone passing close would otherwise stay under the robot.

Counting every scan as one observation, however many points it contributes, stops dense
near-range cells from becoming "sticky" compared with sparse far ones. The rules are
complementary: decay forgets what is no longer seen; ray and ground clearing remove what is seen
to be gone; appearance makes new things count at once. Together they make the map *asymmetric
in the safe direction*: obstacles appear in one scan, and disappear as soon as there is evidence.

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

Two rules are specific to navigation:
- **Pedestrian keep-out.** Cells whose top class is pedestrian and that stand more than 0.5 m above
  the ground make every cell within 1 m lethal (personal space plus the distance walked while the
  costmap is in transit). A person is an obstacle whatever its label, since anything taller than
  the step limit is; the label adds the margin.
- **The robot's own footprint is never lethal** in the published costmap. Keep-out discs,
  predicted corridors (§3.9) and step checks reach into neighbouring cells; if they cover the
  robot, the controller and every recovery behaviour see a collision at the current pose and the
  robot deadlocks.

The costmap is built and published on the mapping thread **after every scan** (about 25–45 ms
including terrain analysis), ahead of any visualisation, so it reaches the planner 0.2–0.4 s
after the scan.

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

### 3.9 Motion-aware keep-out for moving people

A keep-out disc marks where a person *is*. A person crossing the robot's path walks into the lane
at the moment the robot arrives, so with a static disc the planner sees nothing in the way until
it is too late. `adaptmap/motion.py` predicts instead:

1. The pedestrian cells of each costmap (§3.7) are split into connected blobs, and each blob's
   centroid is taken in world coordinates.
2. Blobs are associated with existing tracks by nearest neighbour inside a gate that grows with
   the elapsed time (0.5 m + 3 m/s · Δt). Unmatched blobs start tracks; tracks unseen for 1 s are
   dropped.
3. A track's velocity is its **net displacement over the last 1 s**. It counts as moving only if
   the history spans at least 0.6 s, the speed is between 0.3 and 3 m/s, and the path is nearly
   straight (net / travelled distance ≥ 0.7). A static object whose blob merges with a passer-by,
   or is seen from a changing angle, has a centroid that jumps back and forth; it is fast but goes
   nowhere, so it gets no prediction.
4. Each moving track marks a capsule lethal: from the centroid to **the blob's front edge along
   the velocity** plus *v* · 2.5 s, 0.6 m wide on each side. Moving-class cells fade over
   τ_dynamic, so a walking person's blob drags a trail and its centroid lags the person by up to
   ~1 m; the front edge does not.

In the simulated crossing scenario this turned contact at full speed into the robot stopping
about 3 m from the walker and continuing once they had passed.

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
- **Auto-labelled training data:** randomised simulated worlds (`tools/make_world.py --seed N`) are
  driven while `tools/record_labeled.py` saves each preprocessed scan with labels from the world
  manifest's object boxes (`tools/autolabel.py`; 97.7 % agreement with the ray-caster's own
  labels). A **ground-contact rule** is essential: boxes reach down to the ground, so the
  road visible around a pedestrian's feet or under a car's edge would be labelled as the object;
  points less than 8 cm above the ground box under them take the ground class (≈60 % of vehicle
  and pedestrian labels were ground before this rule). Older recordings were repaired with
  `tools/fix_labels.py`, which uses the observed ground instead and leaves points with no ground
  observed within 1 m untouched (next to the robot, inside the blind disc).
- **Fine-tuning** (`tools/finetune.py`): the 19-class KITTI classifier is replaced by a 10-class
  head whose rows start from each AdaptMap class's main KITTI class (road ← road,
  vehicle ← car, …), so training starts from the pretrained behaviour. Loss: class-balanced
  cross-entropy plus **distillation** towards the frozen pretrained network wherever that teacher
  is confident and does not contradict the label. This keeps knowledge the simulated worlds do not
  cover (for example, a pickup's flat bed is still a vehicle). Result: the default sim model
  reaches 88–90 mIoU on two held-out worlds against 49–51 for the pretrained network.

## 5. Prototype implementation

| Component | File |
|---|---|
| Foveated map engine (fusion, clearing, views, resampling) | `adaptmap/foveated_map.py` |
| Terrain analysis and costmap | `adaptmap/terrain.py` |
| Motion-aware keep-out | `adaptmap/motion.py` |
| Speed-adaptive zone policy | `adaptmap/adaptive.py` |
| Preprocessing | `adaptmap/preprocess.py` |
| Segmentation wrapper | `adaptmap/segmentation.py` |
| Classes & mapping | `adaptmap/classes.py` |
| ROS 2 node (costmap, GridMaps, cells, stats) | `ros2_ws/src/adaptmap_ros/adaptmap_ros/adaptmap_node.py` |
| Nav2 configuration | `ros2_ws/src/adaptmap_nav/` |
| Gazebo worlds + rover | `tools/make_world.py`, `ros2_ws/src/adaptmap_sim/` |
| Analytic LiDAR simulator | `tools/synth_scan.py` |
| Training data and fine-tuning | `tools/record_labeled.py`, `tools/autolabel.py`, `tools/finetune.py` |
| Evaluations | `tools/benchmark.py`, `tools/eval_sim_test.py`, `tools/eval_bag.py`, `tools/eval_semanticposs.py`, `tools/run_dataset.py`, `tools/nav_scenarios.py` → `tools/make_results.py` → `docs/RESULTS.md` |
| Unit tests (CPU, CI) | `adaptmap/tests/` |

## 6. Measured results (RTX 4050 laptop)

All numbers come from `docs/RESULTS.md`, which `tools/make_results.py` generates from the
evaluation tools' JSON output.

**Map efficiency.** Offline benchmark of a simulated 10 Hz drive; a uniform 5 cm map is fed the
same data for comparison.

| Metric | Uniform 5 cm | AdaptMap | Gain |
|---|---|---|---|
| Cells (200 × 200 m) | 16,000,000 | 702,400 | 22.8× |
| Map memory | 916 MB | 48 MB | 19× |
| Map update per scan (incl. ray clearing) | 40.0 ms | 7.4 ms | 5.4× |

**Speed.** Full pipeline offline: 37.5 ms per scan (27 Hz): preprocess 1.5 ms, MinkUNet-34 in fp16
28.5 ms, fusion 7.4 ms. Live in ROS 2 with Gazebo, RViz and a video renderer on the same GPU:
6.0 scans/s; the mapping core alone keeps up with the 10 Hz sensor at 9.3 scans/s. Speed-adaptive
zones: the 5 cm zone grows to 15.5 m at 5 m/s and 25 m at 8 m/s, with the map update rising only
from 7.1 to 9.6 ms.

**Segmentation** (mIoU over the 9 AdaptMap classes):

| Data | Pretrained (KITTI) | Fine-tuned sim model (default) |
|---|---|---|
| Held-out sim world (seed 21, 245 scans) | 49.0 | **88.4** |
| Held-out demo world (recorded lap) | 51.4 | **90.2** |
| SemanticPOSS (real, 40-beam, campus; ground merged) | **43.6** | 38.3 (constant intensity) |

**Real data.** KITTI raw drive 0005 (154 real HDL-64 scans with GPS/IMU poses): the full pipeline
runs at 52 ms per scan (19 Hz) with the pretrained model.

**Autonomous navigation.** Nav2 (SmacPlanner2D + Regulated Pure Pursuit) driving on the AdaptMap
costmap in the demo world, 5 scripted scenarios × 3 runs: **15/15 passed**. The scenarios are:
a pedestrian crossing the road, a parked car blocking the lane, a goal behind a 15 cm curb (must be
refused), a 55 m route past crates and cars, and a U-turn inside the road. Pass means the expected
outcome, never leaving the road, no contact with static obstacles and no contact with a pedestrian
while moving. Closest static clearance over all runs: 0.44 m.

## 7. Limitations of this prototype

- Odometry is ground truth in simulation (no localisation drift). On KITTI the GPS/IMU poses are
  used.
- The sim model is trained with constant intensity and is weaker than the pretrained model on
  real data (SemanticPOSS 38.3 vs 43.6 mIoU); real sensors use the pretrained model. A mixed
  sim + real fine-tune would give one model for both.
- Weakest classes: vegetation (61–72 % IoU in simulation) and pedestrians in an unfamiliar world
  (65 %).
- The pedestrian prediction is a constant-velocity extrapolation of the map's own blobs, with no
  intent model. The robot sometimes yields *on* a crossing line instead of before it.
- Low obstacles that a stationary robot can no longer see (inside the blind disc) decay with
  τ_static = 20 s.
- Not yet run on embedded hardware (Jetson / TensorRT). Live rate (6 scans/s) is below the
  10 Hz sensor rate only when the simulator and visualisation share the laptop GPU.

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
9. Free-space evidence from the scan's own rays (ray clearing), restricted to the inside of a
   cell's stored obstacle span, so rays passing over objects or under overhangs clear nothing.
10. Asymmetric fusion for safety: a scan that contradicts a stored obstacle with ground-only
    returns discounts it (ground clearing), and a scan whose top rises above the stored top
    replaces the cell (appearance), so obstacles appear in one scan and vanish on evidence.
11. Forgetting the cells under the robot's footprint each scan, because they lie in the sensor's
    blind disc where no clearing evidence can arrive, and never marking the robot's own footprint
    lethal in the published costmap.
12. A clearance layer (free height under the lowest obstacle point above the ground), letting a
    2.5D map mark space under overhangs as drivable for a given robot height.
13. Ring-gap filling that interpolates ground only between observed cells on opposite sides,
    within a zone-dependent radius, never extrapolating into shadows.
14. Speed-adaptive foveation: the fine zone's radius follows reaction plus braking distance, with
    hysteresis, and resizing resamples the map (max-height when coarsening) instead of clearing it.
15. A motion-aware keep-out computed from the map itself: tracking moving-class blobs, accepting a
    velocity only when the path over a time window is nearly straight, and marking the corridor
    from the blob's *front edge* (not its lagging centroid) for a prediction horizon as lethal.
16. A training pipeline for the segmentation network from simulation alone: randomised worlds,
    exact labels from the world manifest with a ground-contact rule, a reduced-class head
    initialised from the pretrained classifier's rows, and distillation towards the pretrained
    network where it is confident and consistent with the label.
