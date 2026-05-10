# gazebo_wind_plugin

Gazebo **Classic** model plugins that sample a precomputed 3-D wind field from a JSON + VTK ImageData (`.vti`) lookup table (LUT), apply a quadratic drag force to a link, and ship small helpers for demo flight (world-frame hover PID, visual trail markers, **visible rotor spin** on revolute joints). Optional LUT masks support outdoor-only queries and hotspot snapping near buildings.

Longer ops notes: [RBM-2-iris_wind_quad-done-and-gazebo_wind_plugin-review.md](../docs/ops/RBM-2-iris_wind_quad-done-and-gazebo_wind_plugin-review.md), and the RBM-4 narrative spec [RBM-4-feedback-to-gazebo_guangzhou_wind_hires_demo.md](../docs/ops/RBM-4-feedback-to-gazebo_guangzhou_wind_hires_demo.md) (Chinese).

**RBM-4 two-act hires worlds** (mesh-aligned building collision + different hover configs) live in `worlds/guangzhou_demo_pt1_crash.world` and `worlds/guangzhou_demo_pt2_hover.world`, using models `iris_wind_quad_hires_pt1_crash` / `iris_wind_quad_hires_pt2_hover` and static `guangzhou_buildings` (same `buildings.stl` for visual and ODE collision). Optional `demo_building_collision` remains for convex-hull workflows but is **not** included in these worlds by default.

## Requirements

- **Gazebo Classic** (e.g. Gazebo 11) with development headers (`find_package(gazebo)`).
- **CMake** 3.10 or newer.
- **C++17** compiler.
- **VTK 9** with `CommonCore`, `CommonDataModel`, and `IOXML` (`.vti` read path is always enabled via `GAZEBO_WIND_PLUGIN_USE_VTK_VTI=1`).

LUT files (`wind_lut.json`, `wind_lut.vti`) must exist on the machine and be readable at the paths given in SDF. The bundled model SDFs use **absolute paths** as examples; change them to your cache or dataset location before running.

## Build

From this directory:

```bash
cmake -S . -B build
cmake --build build -j
```

Artifacts (shared libraries):

| Library | Sources |
| --- | --- |
| `libWindFieldPlugin.so` | `WindFieldPlugin.cc`, `lut_reader/WindLUT.cc` |
| `libHoverPidPlugin.so` | `HoverPidPlugin.cc` |
| `libTrailMarkerPlugin.so` | `TrailMarkerPlugin.cc` |
| `libContactWatcherPlugin.so` | `ContactWatcherPlugin.cc` |
| `libRotorSpinPlugin.so` | `RotorSpinPlugin.cc` |

Point Gazebo at the build output, for example:

```bash
export GAZEBO_PLUGIN_PATH="$(pwd)/build:${GAZEBO_PLUGIN_PATH}"
```

Alternatively, install or copy the `.so` files into a directory already on `GAZEBO_PLUGIN_PATH` (see Gazebo documentation for plugin search paths).

## Wind LUT (JSON + VTI)

`WindLUT::loadFromJsonAndVti` reads:

1. **JSON** (small; parsed with a minimal in-file parser, no external JSON library): must contain numeric arrays **`origin`**, **`spacing`**, and integer array **`dimensions`** `[Nx, Ny, Nz]`. These must match the VTK grid dimensions.
2. **VTI** (`vtkXMLImageDataReader`): point data must include vector field **`U`** (3 components, m/s). Optional scalar arrays:
   - **`valid_mask`** or **`vtkValidPointMask`** — zero means invalid for interpolation contributions.
   - **`inside_building`** — non-zero marks indoor / non-outdoor cells when present.

Grid indexing follows VTK point order (x fastest): `index = (ix * Ny + iy) * Nz + iz`. `query(x,y,z)` performs trilinear interpolation in LUT coordinates (meters), with out-of-range returning `(0,0,0)`. Invalid corners contribute as zero wind.

Hotspot logging in `WindFieldPlugin` can optionally call `snapHotspotNearestOutdoor` so a probe point that lands inside a masked column is moved in **XY** on the same **z** slice to the nearest outdoor cell (or, without a building mask, to a cell with `|U| ≥` a fallback threshold).

## Plugins and SDF parameters

Register names: `WindFieldPlugin`, `HoverPidPlugin`, `TrailMarkerPlugin`, `ContactWatcherPlugin`, `RotorSpinPlugin` (see `GZ_REGISTER_MODEL_PLUGIN` in each `.cc` file).

### WindFieldPlugin (`libWindFieldPlugin.so`)

Each update: reads link world pose (plus optional LUT offset), samples `(u,v,w)`, forms relative velocity `U_rel = U_wind - v_link`, then applies

`F = 0.5 * rho * C_D * area * |U_rel| * U_rel`

to the configured link. Optional torque: `tau = r × F` with `r = (wind_torque_arm_x, wind_torque_arm_y, wind_torque_arm_z)` when `enable_wind_torque` is true. Setting `arm_z` only excites roll/pitch (legacy single-axis behaviour); adding non-zero `arm_x` / `arm_y` lets horizontal wind also drive a yaw torque (`τ_z = arm_x·fy − arm_y·fx`), giving 6-DOF buffeting instead of yaw-locked drift.

| Element | Type | Default | Description |
| --- | --- | --- | --- |
| `lut_json` | string | *(required)* | Path to LUT sidecar JSON (`origin`, `spacing`, `dimensions`). |
| `lut_vti` | string | *(required)* | Path to VTK ImageData wind file. |
| `link_name` | string | `base_link` | Link receiving force/torque. |
| `rho` | double | `1.225` | Air density (kg/m³). |
| `C_D` | double | `1.0` | Drag coefficient (dimensionless). |
| `area` | double | `0.04` | Reference area (m²). |
| `world_to_lut_offset_x` | double | `0` | Added to link X before LUT query (m). |
| `world_to_lut_offset_y` | double | `0` | Added to link Y before LUT query (m). |
| `world_to_lut_offset_z` | double | `0` | Added to link Z before LUT query (m). |
| `log_every_n` | int | `0` | If greater than zero, print position/wind/force every N steps (`gzmsg`). |
| `hotspot_x` | double | `1420` | Hotspot X for startup diagnostic sample (m). |
| `hotspot_y` | double | `-880` | Hotspot Y for startup diagnostic sample (m). |
| `hotspot_z` | double | `145` | Hotspot Z for startup diagnostic sample (m). |
| `enable_wind_torque` | bool | `false` | Enable `r × F` torque about base link origin. |
| `wind_torque_arm_x` | double | `0` | Moment arm X component (m); excites yaw + cross-axis torque. |
| `wind_torque_arm_y` | double | `0` | Moment arm Y component (m); excites yaw + cross-axis torque. |
| `wind_torque_arm_z` | double | `0.15` | Moment arm Z component (m); excites roll/pitch from horizontal wind. |
| `hotspot_snap_outdoor` | bool | `true` | Before hotspot check, snap to nearest outdoor XY if needed. |
| `hotspot_snap_max_radius_m` | double | `120` | Max XY search radius for snap (m). |
| `hotspot_snap_min_wind` | double | `0.05` | Min `|U|` (m/s) fallback when no building mask. |

### HoverPidPlugin (`libHoverPidPlugin.so`)

World-frame translational PID on position error `target - pos`; applies force to the link each step. Simple integral clamp (anti-windup). XY uses `kp`/`ki`/`kd`; Z uses `kp_z`/`ki_z`/`kd_z` when set, otherwise defaults to the XY gains at load time. Optional roll/pitch damping torque when `enable_attitude_recovery` is true. If `drift_after_seconds ≥ 0`, XY forces are enabled only while `sim_time < drift_after_seconds`, then XY PID is turned off (Z continues).

Setting `gravity_compensation` to true reads the controlled link's mass and the world's gravity at load time and adds a constant `m·|g|` along +Z each step on top of the PID output, so soft Z gains (e.g. `kp_z=4`) no longer leak into a multi-metre steady-state altitude error. The feed-forward stops when `disable_topic` latches with `crash_zero_thrust=true` (free-fall).

| Element | Type | Default | Description |
| --- | --- | --- | --- |
| `link_name` | string | `base_link` | Controlled link. |
| `target_x` | double | `1420` | Setpoint X (m). |
| `target_y` | double | `-880` | Setpoint Y (m). |
| `target_z` | double | `50` | Setpoint Z (m). |
| `kp` | double | `8` | XY position gain. |
| `ki` | double | `0.1` | XY integral gain. |
| `kd` | double | `4` | XY derivative gain. |
| `kp_z` | double | same as `kp` | Z position gain. |
| `ki_z` | double | same as `ki` | Z integral gain. |
| `kd_z` | double | same as `kd` | Z derivative gain. |
| `enable_xy` | bool | `true` | Enable XY PID (subject to `drift_after_seconds`). |
| `enable_attitude_recovery` | bool | `false` | Apply `-attitude_kp * roll/pitch` torques. |
| `attitude_kp` | double | `15` | Attitude recovery gain (lower = visible buffeting tilt; higher = stiff hold). |
| `gravity_compensation` | bool | `false` | If true, add a constant `m·|g|` feed-forward along +Z so Z PID only fights wind, not gravity. |
| `drift_after_seconds` | double | `-1` | If ≥ 0, disable XY PID after this sim time (s). |
| `disable_topic` | string | empty | Optional Gazebo transport topic; first `GzString` message latches `disabled`. |
| `crash_zero_thrust` | bool | `false` | When latched, return zero force/torque each step (free fall). |
| `log_every_n` | int | `250` | Periodic `gzmsg`: error, force, **roll/pitch in degrees**. |

### ContactWatcherPlugin (`libContactWatcherPlugin.so`)

`SensorPlugin` attached to a `<sensor type="contact">` on the drone collision body. Each update reads `Contacts()`, computes the peak normal force, periodically logs `[ContactWatcher] count=N peak_force=…N a=… b=…`. The first time the peak crosses `crash_threshold_n` it logs `[ContactWatcher] CRASH …` and publishes a one-shot `GzString` `"crash"` on `publish_disable_topic` (default `~/hover_pid/disable`) so a subscribed `HoverPidPlugin` can zero its outputs.

| Element | Type | Default | Description |
| --- | --- | --- | --- |
| `crash_threshold_n` | double | `5` | Latch threshold for peak normal force (N). |
| `publish_disable_topic` | string | `~/hover_pid/disable` | Gazebo transport topic for the one-shot disable message. |
| `log_every_n` | int | `50` | Periodic contact summary cadence in sensor updates. |

### RotorSpinPlugin (`libRotorSpinPlugin.so`)

`ModelPlugin` that drives **revolute** joints with the ODE joint motor (`SetParam("fmax")`, `SetParam("vel")`) so propeller **visual** meshes spin at a target angular rate (rad/s). This is **cosmetic**: it does not replace `HoverPidPlugin` / `WindFieldPlugin` forces. Put each rotor on its own lightweight link (no collision) connected to `base_link` with `<joint type="revolute"><axis><xyz>0 0 1</xyz>…`.

| Element | Type | Default | Description |
| --- | --- | --- | --- |
| `<rotor><joint>` | string | *(required)* | Name of the revolute joint to drive. |
| `<rotor><rate>` | double | `0` | Target angular velocity (rad/s); **+** = CCW about +Z, **−** = CW. Alternate signs on diagonal motors so net yaw torque on the body stays small. |
| `max_torque` | double | `0.5` | Joint motor torque cap (N·m); keeps reaction on `base_link` bounded. |
| `spin_up_tau` | double | `1.5` | Seconds to ramp from 0 to full `rate` at load. |
| `spin_down_tau` | double | `1.0` | After `disable_topic` latches, exponential decay time constant (s) to 0 rad/s. |
| `disable_topic` | string | empty | Optional; subscribe to the same topic as `ContactWatcherPlugin` / `HoverPidPlugin` (e.g. `~/hover_pid/disable`) so rotors **spool down** on crash. |
| `log_every_n` | int | `0` | If &gt; 0, periodic `gzmsg` with ramp/decay factors. |

### TrailMarkerPlugin (`libTrailMarkerPlugin.so`)

Samples link pose every `sample_period` seconds and spawns a **static** sphere model via Gazebo transport `~/factory`; deletes oldest markers via `~/request` with `entity_delete` when `max_points` is exceeded. Visual-only; useful for trajectory debugging.

| Element | Type | Default | Description |
| --- | --- | --- | --- |
| `link_name` | string | `base_link` | Tracked link. |
| `sample_period` | double | `0.75` | Seconds between markers. |
| `marker_radius` | double | `0.8` | Sphere radius (m). |
| `max_points` | int | `60` | Max retained markers. |
| `name_prefix` | string | `trail_marker_` | Spawned model name prefix. |
| `color_r` | double | `1.0` | Marker color R. |
| `color_g` | double | `0.3` | Marker color G. |
| `color_b` | double | `0.1` | Marker color B. |
| `color_a` | double | `0.65` | Marker color A. |

## Directory layout

| Path | Role |
| --- | --- |
| `CMakeLists.txt` | Build definition for the three plugins. |
| `WindFieldPlugin.cc` | Wind LUT sampling + drag force/torque model plugin. |
| `HoverPidPlugin.{hh,cc}` | Demo hover / drift PID model plugin. |
| `TrailMarkerPlugin.{hh,cc}` | Visual trail spheres via factory transport. |
| `lut_reader/WindLUT.{hh,cc}` | JSON + VTI load, masks, trilinear `query`, outdoor snap. |
| `worlds/` | Example worlds: `guangzhou_wind.world`, `guangzhou_wind_hires_demo.world`, RBM-4 `guangzhou_demo_pt1_crash.world`, `guangzhou_demo_pt2_hover.world`. |
| `models/` | Local Gazebo models for demos and visuals (see below). |
| `build/` | Local CMake output (safe to delete; do not treat as source). |

### `models/` overview

- **`iris_wind_quad`** — Standard-scale quad visual + collision; `WindFieldPlugin`, `HoverPidPlugin`, `TrailMarkerPlugin` (LUT paths in SDF are examples under `~/wrf_openfoam_coupling_cache/...`).
- **`iris_wind_quad_hires_demo`** — 10× scaled mesh/collision for visibility; wind torque (`wind_torque_arm_z=0.3`), link angular `velocity_decay`, attitude recovery, timed XY drift; hires LUT paths in SDF.
- **`iris_wind_quad_hires_pt1_crash`** / **`iris_wind_quad_hires_pt2_hover`** — Thin wrappers: same meshes via `model://iris_wind_quad_hires_demo/meshes/...`, different `HoverPid` / spawn used by RBM-4 worlds; four `prop_*` links + revolute joints + `RotorSpinPlugin` for **visible** high-speed rotors (spool-down on `~/hover_pid/disable` after crash in pt1).
- **`demo_building_collision`** — Optional static convex-hull collision model (regenerated by `build_demo_collision_model.py`); can include a translucent debug visual. RBM-4 pt1/pt2 worlds use `guangzhou_buildings` mesh collision instead.
- **`wind_arrows_hotspot`** — Static many-link wind arrows (box shaft/head) near the low-res demo region.
- **`wind_arrows_hotspot_hires`** — Dense static arrows using mesh glyph; references `wind_arrow_glyph` mesh URI.
- **`wind_arrow_glyph`** — Shared unit arrow mesh (`meshes/arrow_unit.stl`) and a tiny off-world stub visual for Gazebo model resolution.
- **`guangzhou_buildings`** — Street/canyon `buildings.stl` with matching **visual + mesh collision** (static trimesh).
- **`px4_iris_assets`** — **Stub** model only: prevents Gazebo model-path scans from failing on the nested `repo/` checkout. **Do not** include this for flight demos; use `iris_wind_quad` or `iris_wind_quad_hires_demo`.

Meshes under `models/*/meshes/` (`.stl`, `.dae`) are binary assets used by the SDF files; they are not documented line-by-line here.

## Running a demo

1. Build plugins and set `GAZEBO_PLUGIN_PATH` as above.
2. Add this package’s models to the model path:

   ```bash
   export GAZEBO_MODEL_PATH="/path/to/WRF-OpenFOAM-Coupling/gazebo_wind_plugin/models:${GAZEBO_MODEL_PATH}"
   ```

3. Edit `<lut_json>` / `<lut_vti>` inside `models/iris_wind_quad/model.sdf` (or `iris_wind_quad_hires_demo/model.sdf`) to match your LUT files.
4. Launch Gazebo from the repository root or any cwd, passing the world file:

   ```bash
   gazebo /path/to/WRF-OpenFOAM-Coupling/gazebo_wind_plugin/worlds/guangzhou_wind.world
   gazebo /path/to/WRF-OpenFOAM-Coupling/gazebo_wind_plugin/worlds/guangzhou_wind_hires_demo.world
   gazebo /path/to/WRF-OpenFOAM-Coupling/gazebo_wind_plugin/worlds/guangzhou_demo_pt2_hover.world
   gazebo /path/to/WRF-OpenFOAM-Coupling/gazebo_wind_plugin/worlds/guangzhou_demo_pt1_crash.world
   ```

From the repository root, convenience wrappers (same `build` / `cache` / `arrows` / `smoke` flow as `scripts/run_gazebo_guangzhou_wind_hires_demo.sh`):

```bash
./scripts/run_gazebo_guangzhou_demo_pt2_hover.sh smoke
./scripts/run_gazebo_guangzhou_demo_pt1_crash.sh gui
```

Headless smoke test (manual):

```bash
timeout 20s gzserver /path/to/WRF-OpenFOAM-Coupling/gazebo_wind_plugin/worlds/guangzhou_demo_pt2_hover.world --verbose
```

The worlds set an initial GUI camera near the demo coordinates (~km offsets from origin) and include `sun`, `ground_plane`, buildings, wind arrows, and the iris model.

## Related scripts (repository root)

- `scripts/generate_gazebo_wind_arrows.py` — static wind arrow models.
- `scripts/generate_arrow_unit_stl.py` — unit arrow mesh for hires arrows.
- `scripts/extract_demo_building_collision.py` — V-HACD / CoACD convex decomposition of buildings near hotspot; writes `data/demo_assets/building_*/hull_*.stl` and `collision_manifest.json`. Falls back to LUT `inside_building` mask via `--from-lut wind_lut.vti` when the city STL is unavailable on this host.
- `scripts/build_demo_collision_model.py` — turns the manifest into `gazebo_wind_plugin/models/demo_building_collision/model.sdf` (one `<link>` per hull, `collide_bitmask=0x01`, ODE surface tuning).
- `scripts/verify_collision_alignment.py` — three-way QC slice (LUT mask vs. visual STL vs. hull boundaries) at a chosen Z; nonzero exit if hulls overflow `inside_building` beyond tolerance.
- `scripts/probe_wind_at_hotspot.py` — sample the LUT around the hotspot, compute the dominant wind direction, write `wind_probe_quiver.png`, and print a recommended pt1 spawn pose `<pose>` snippet on the upwind side of the collision union.
- `scripts/check_gazebo_env.sh` — environment sanity checks.

LUT cache path used in many SDF examples: `~/wrf_openfoam_coupling_cache/wind_lut/...` (copy large `.vti` here on WSL per ops docs).

## Collision pipeline (RBM-4 pt1/pt2)

**Default demos:** `guangzhou_buildings` carries collision on the same STL as the visuals (no separate red proxy).

**Optional convex decomposition:** the static `demo_building_collision` model is **regenerated** rather than hand-edited. Use the steps below when you want hull-based collision (e.g. performance tuning or QC vs. the LUT mask) instead of the full building mesh.

```bash
# 1) extract hulls (CoACD preferred; falls back to V-HACD then convex_hull)
python scripts/extract_demo_building_collision.py \
  --input <path>/buildings.stl \
  --hotspot 1470,1350 --radius-m 50

# 2) regenerate the Gazebo collision model from the manifest
python scripts/build_demo_collision_model.py \
  --manifest data/demo_assets/collision_manifest.json \
  --model-dir gazebo_wind_plugin/models/demo_building_collision

# 3) verify hull ⊂ LUT inside_building at z=80 m (exit code 2 on overflow)
python scripts/verify_collision_alignment.py \
  --manifest data/demo_assets/collision_manifest.json \
  --lut ~/wrf_openfoam_coupling_cache/wind_lut/<dataset>/wind_lut.vti \
  --z 80 --out data/demo_assets/qc_alignment.png

# 4) (optional) recommend pt1 spawn on the upwind side
python scripts/probe_wind_at_hotspot.py \
  --lut ~/wrf_openfoam_coupling_cache/wind_lut/<dataset>/wind_lut.vti \
  --manifest data/demo_assets/collision_manifest.json \
  --hotspot 1470,1350,80
```

`./scripts/run_gazebo_guangzhou_demo_pt1_crash.sh collision-build` chains the same four steps with sensible defaults; environment overrides are `BUILDINGS_STL`, `LUT_VTI`, `HOTSPOT`, `RADIUS_M`. When `BUILDINGS_STL` is unset the extractor falls back to the LUT mask (`scipy.ndimage.label` connected components → axis-aligned boxes), so the demo can still run on hosts without the city mesh.

**Drone side** (already in `iris_wind_quad_hires_pt1_crash` / `pt2_hover`): the body collision has `<collide_bitmask>0x01</collide_bitmask>` and a `contact` sensor hosting `libContactWatcherPlugin.so`. On the first `peak_force ≥ crash_threshold_n` the watcher publishes one `GzString` on `~/hover_pid/disable`; the pt1 hover PID has `<crash_zero_thrust>true</crash_zero_thrust>` so the drone goes into free fall. pt2 keeps `crash_zero_thrust=false` and only logs the brush.

## Maintainer

- **Zhixian Yang** — [zyang248@connect.hkust-gz.edu.cn](mailto:zyang248@connect.hkust-gz.edu.cn)
