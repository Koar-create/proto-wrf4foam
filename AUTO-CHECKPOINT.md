# AUTO-CHECKPOINT

> 本文件由Cursor智能体自动维护，用于记录本仓库内任务执行过程、关键决策与可复现命令。

**仓库维护**：Zhixian Yang（zyang248@connect.hkust-gz.edu.cn）

## 2026-05-10

### pt1/pt2 巡检 Demo 调优（2026-05-10 晚）
- **航迹**：由 `buildings.stl` 热点区 XY 点云 → 2D 凸包 → 外扩 `standoff` → 密化折线，与 pt1/pt2 共用（脚本 [`scripts/generate_inspection_waypoints.py`](scripts/generate_inspection_waypoints.py) 可重生成）。
- **pt2**：略减 `area`、风力矩臂、`attitude_kp=4.5`、`barrier_*` 缓和，避免 ±120° 级抖动；路线沿凸包贴楼而非大矩形切角。
- **pt1**：大减 `wind_bias`/`force_scale`/`area`；**WindField** 增加 `disable_topic=~/hover_pid/disable`，撞楼后停风避免“吸在墙上不落”；机体接触 **mu=0.12** 利滑落；**关闭风力矩** + 轻 `attitude_kp` 保持前期可飞；`soften_xy_after_waypoint_index=14`（约 2/3 圈后失控）；旋翼 `spin_down_tau=2.4`。

### pt1/pt2 巡检 Demo（RBM-6）：绕楼航线 + pt1 撞楼 / pt2 安全绕障
- **动机**：由单点悬停升级为「高楼幕墙巡检」叙事；两机共享同一航点序列，表现力优先（pt1 可叠加非真实风偏置）。
- **新插件**：`gazebo_wind_plugin/InspectionPathControllerPlugin.{hh,cc}` → `libInspectionPathControllerPlugin.so`：多航点到达判定、`loop` 闭环、与 `HoverPid` 一致的重力前馈/姿态恢复/`disable_topic`/`crash_zero_thrust`；可选 2D 建筑 AABB + `safety_margin` 距离障碍势场（`barrier_gain` / `barrier_vel_gain`）作为 pt2 的演示型 CBF 替代。
- **风场**：`WindFieldPlugin` 增加 `<wind_bias_x/y/z>`（m/s，加入 LUT 风速）与 `<force_scale>`（默认 1，放大阻力），便于 pt1 保证撞楼。
- **模型**：`iris_wind_quad_hires_pt1_crash` 用 `inspection_path` 替换 `hover_pid`（无 barrier + `soften_xy_after_waypoint_index=5` + 大风偏置）；`iris_wind_quad_hires_pt2_hover` 同航点 + barrier。
- **脚本**：`run_gazebo_guangzhou_demo_pt1_crash.sh` / `pt2_hover.sh` 的 `smoke` 延长至 90s / 75s。
- **实测**：pt1 约 95s 内 `CRASH`→`InspectionPathControllerPlugin disabled`→`RotorSpinPlugin disabled`；pt2 约 78s 内完成航点闭环（`loop: restart waypoints`）且无 `CRASH`。

### 追踪文档署名（README / SKILL / 本文件）
- **范围**：仅 `git ls-files` 内的 Markdown；为原先无维护者信息的文档补齐署名。
- **改动**：[`gazebo_wind_plugin/README.md`](gazebo_wind_plugin/README.md) 文末增加 **Maintainer**；[`.cursor/skills/gazebo-wind-plugin-demo/SKILL.md`](.cursor/skills/gazebo-wind-plugin-demo/SKILL.md) frontmatter 增加 `maintainer` 并在正文标题下增加一行维护者；本文件页眉增加 **仓库维护** 行。

### Gazebo `model.config` 署名（git 追踪范围内）
- **范围**：仅统计 `git ls-files` 所列文件；仓库内凡含 `<author>` 的 `model.config` 均已为 **Zhixian Yang** / **zyang248@connect.hkust-gz.edu.cn**。
- **本次补齐**：[`iris_wind_quad_hires_pt2_hover/model.config`](gazebo_wind_plugin/models/iris_wind_quad_hires_pt2_hover/model.config) 原先仅有 `<name>`，已补上 `<email>` 与其它模型一致。
- **未改动**：`gazebo_wind_plugin/models/px4_iris_assets/repo/` 下上游拷贝的 `model.config` **未被 git 追踪**（`git status` 为未跟踪目录），故不在本次批量范围内；各 `*.dae` 中 COLLADA 的 `<author>Blender User</author>` 为网格导出元数据，保留。

### 旋翼高速旋转（RotorSpinPlugin + 独立 prop link）
- **动机**：prop 若只是 `base_link` 上的 `<visual>`，无法绕轴旋转；需独立 `<link>` + `<joint type="revolute">` + 每步驱动关节角速度。
- **实现**：[`RotorSpinPlugin.{hh,cc}`](gazebo_wind_plugin/RotorSpinPlugin.hh) `ModelPlugin`：对若干 `<rotor><joint>…</joint><rate>…</rate></rotor>` 调用 `joint->SetParam("fmax",0,max_torque)` + `SetParam("vel",0,target)`；`spin_up_tau` 线性 ramp；可选 `disable_topic` 与 ContactWatcher 同源，latch 后 `spin_down_tau` 指数衰减到 0。纯视觉/运动学增强，**不改变** WindField/HoverPid 受力模型。
- **CMake**：[`CMakeLists.txt`](gazebo_wind_plugin/CMakeLists.txt) 增加 `libRotorSpinPlugin.so`。
- **模型**：[`iris_wind_quad_hires_pt2_hover/model.sdf`](gazebo_wind_plugin/models/iris_wind_quad_hires_pt2_hover/model.sdf)、[`iris_wind_quad_hires_pt1_crash/model.sdf`](gazebo_wind_plugin/models/iris_wind_quad_hires_pt1_crash/model.sdf) 将原 4 个 prop visual 拆为 `prop_0..3` link（5 g 点质量、无 collision）+ `prop_*__joint` 绕 Z；`rate=±80 rad/s`，`max_torque=0.5`；pt1 `spin_down_tau=1.2`；两机均 `disable_topic=~/hover_pid/disable`。
- **验证**：pt2 smoke 见 `[RotorSpinPlugin] 4 rotor(s): …`；pt1 CRASH 后日志 `[RotorSpinPlugin] disabled by 'crash' …`。
- **文档**：[`gazebo_wind_plugin/README.md`](gazebo_wind_plugin/README.md) 增加 `libRotorSpinPlugin` 表与 RotorSpinPlugin 参数表。

### 增强 pt1/pt2 demo 的 6-DOF 表现力
- **目标**：让两架机的位移与姿态在演示中可见、可读，而不是被刚性 PID 压平或被低风力淹没。
- **插件扩展**：
  - [`WindFieldPlugin`](gazebo_wind_plugin/WindFieldPlugin.cc) 新增 `<wind_torque_arm_x>` / `<wind_torque_arm_y>`（默认 0），力矩改为 `r×F`，`r=(arm_x, arm_y, arm_z)`；横向臂使水平风也产生 `τ_z = arm_x·fy − arm_y·fx`，激励**偏航**与交叉轴扭，使 demo 6 个自由度都被风激起。向后兼容旧 SDF。
  - [`HoverPidPlugin`](gazebo_wind_plugin/HoverPidPlugin.hh) 新增 `<gravity_compensation>`（默认 false）：load 时读 link 质量与 world 重力，每步在 `fz` 上加常量 `m·|g|`。`crash_zero_thrust` 触发后 plugin 直接 return，FF 也随之停止 → 自由落体不变。
- **pt2 hover 调谐**（[`iris_wind_quad_hires_pt2_hover/model.sdf`](gazebo_wind_plugin/models/iris_wind_quad_hires_pt2_hover/model.sdf)）：
  - 风：`area 0.04→0.25`、`C_D 1.0→1.2`、`wind_torque_arm_z 0.3→0.5`、新增 `wind_torque_arm_x=0.12`。
  - PID：`kp 8→5`、`kd 4→3`、`ki 0.1→0.08`、`kd_z 2→2.5`、`attitude_kp 8→2`；启用 `gravity_compensation=true`。
  - link：`velocity_decay angular 0.4→0.18`。world 初始 yaw=0.20 rad 让航向偏转可读。
  - 现象：稳态 z=80.002 m、xy 偏差 ±2 cm、roll/pitch 在 −1.6°…−1.8° 间小幅摆动 → 风扰可见但稳定。
- **pt1 crash 调谐**（[`iris_wind_quad_hires_pt1_crash/model.sdf`](gazebo_wind_plugin/models/iris_wind_quad_hires_pt1_crash/model.sdf)）：
  - 风：`wind_torque_arm_z 0.3→0.55`、新增 `arm_x=0.18 arm_y=0.08`。
  - PID：`gravity_compensation=true`（其余仍 `enable_xy=false`）。
  - link：`velocity_decay angular 0.4→0.12`；world 初始 yaw=0.35 rad。
  - 现象：60 s smoke 中漂向 +Y，roll 渐发展到 ≤ −62°、pitch ≤ −21°，然后在 `peak_force≈8 N` 撞上 **`guangzhou_buildings::buildings_link::buildings_collision`**（视觉建筑 mesh 本身），HoverPid 由 `~/hover_pid/disable` 失能并自由落体到 `ground_plane`。
- **文档**：[`gazebo_wind_plugin/README.md`](gazebo_wind_plugin/README.md) 同步加 `wind_torque_arm_x/y`、`gravity_compensation` 表项与说明。

### pt1/pt2 spawn 移出建筑（统一到红圈位置）
- **问题**：换成 `buildings.stl` 真碰撞后，pt2 旧 spawn `(1460, 1350, 80)` 与 hover target `(1470, 1350, 80)` 都落在 LUT 建筑包围盒 `x=[1436,1506] y=[1314,1366]` 内，机体直接嵌入墙体；pt1 旧 `(1484.23, 1310.86, 80)` 偏东 14 m，南侧外缘但相机右侧。
- **改动**：
  - [`worlds/guangzhou_demo_pt1_crash.world`](gazebo_wind_plugin/worlds/guangzhou_demo_pt1_crash.world) spawn → `(1471, 1309, 80)`：建筑南侧居中外约 5 m。
  - [`worlds/guangzhou_demo_pt2_hover.world`](gazebo_wind_plugin/worlds/guangzhou_demo_pt2_hover.world) spawn → `(1471, 1309, 80)`：与 pt1 同位。
  - [`models/iris_wind_quad_hires_pt2_hover/model.sdf`](gazebo_wind_plugin/models/iris_wind_quad_hires_pt2_hover/model.sdf) `<hover_pid> target_x/y` → `1471/1309`，避免 PID 把机体拉回建筑内的旧 hotspot。
- **保留**：pt1 `enable_xy=false` 故 hover target 维持原值（不影响行为）；`<wind_field>` 的 `hotspot_*` 仍指 `(1470,1350,80)`，仅做 LUT 吸附/调试，无碰撞影响。
- **物理含义**：主流风 `u=-0.165, v=+0.363 m/s` 仍指北偏东，pt1 会在数秒内被推过 `y=1314` 撞建筑南面；pt2 PID 锁在 `(1471,1309,80)` 上风口悬停，便于稳定观测颠簸。

### 演示碰撞与视觉对齐（去掉红色凸包代理）
- **问题**：`demo_building_collision` 使用凸包/分解网格 + 半透明红 visual，体积远大于 `guangzhou_buildings` 的 `buildings.stl`，无人机撞在「红块」上而非可见楼面。
- **改动**：在 [`gazebo_wind_plugin/models/guangzhou_buildings/model.sdf`](gazebo_wind_plugin/models/guangzhou_buildings/model.sdf) 为同一 `buildings.stl` 增加 `<collision>`（`collide_bitmask=0x01` 及与旧代理一致的 ODE 表面参数）；从 [`guangzhou_demo_pt1_crash.world`](gazebo_wind_plugin/worlds/guangzhou_demo_pt1_crash.world) / [`guangzhou_demo_pt2_hover.world`](gazebo_wind_plugin/worlds/guangzhou_demo_pt2_hover.world) 移除 `model://demo_building_collision` include。
- **保留**：`demo_building_collision` 模型与 `build_demo_collision_model.py` 流水线仍可用于需要凸分解或离线校验的场景；默认演示 world 不再加载。
- **注意**：全量 STL 三角网格碰撞可能比凸包更耗 CPU；若卡顿可再考虑热点区域 CoACD + `--no-visual`。

### 高阶 Gazebo 碰撞方案：V-HACD 凸壳 + ContactWatcherPlugin
- **动机**：截图证据显示 drone 沿 -X 直线穿过 visual 楼面，旧 box 代理 `(1442,1350,123) 28×32×92` 与 visual 几何错位；trail 完全没有偏折；既无 contact sensor 也无 CRASH 日志（详见高阶方案 plan）。
- **离线流水线（新建/重写）**：
  - [`scripts/extract_demo_building_collision.py`](scripts/extract_demo_building_collision.py) 重写：CLI `--input <buildings.stl>` 或 **`--from-lut wind_lut.vti`** 兜底；多 `--hotspot` 支持；CoACD → trimesh.decomposition.convex_decomposition (testVHACD) → `convex_hull` 三级回退；输出 `data/demo_assets/building_<id>/hull_*.stl` + `collision_manifest.json` + `qc_overlay.png`；保留 `--emit-bbox-only` 兜底。
  - [`scripts/build_demo_collision_model.py`](scripts/build_demo_collision_model.py)：从 manifest 生成多 `<link>` 的 [`models/demo_building_collision/model.sdf`](gazebo_wind_plugin/models/demo_building_collision/model.sdf)，每个 `<collision>` 含 `collide_bitmask=0x01`、ODE `max_vel/min_depth`、`mu/mu2/restitution`，可选 transparent visual overlay（默认开）。
  - [`scripts/verify_collision_alignment.py`](scripts/verify_collision_alignment.py)：LUT `inside_building` / 凸壳 / 可选 visual mesh 三方 PyVista 切片 PNG；hull 越界超 `--tolerance-cells` 退出码 2；输出 `data/demo_assets/qc_alignment.png`。
  - [`scripts/probe_wind_at_hotspot.py`](scripts/probe_wind_at_hotspot.py)：LUT 三线性插值 hotspot 周围 `n_xy×n_xy×n_z` 网格抽 (u,v,w)，输出 `wind_probe.txt`/`wind_probe_quiver.png`，并按主流向 + collision union 中心给出 pt1 spawn 推荐 `<pose>`。
- **Gazebo 插件**：
  - [`ContactWatcherPlugin.{hh,cc}`](gazebo_wind_plugin/ContactWatcherPlugin.cc) `SensorPlugin`：周期日志 `[ContactWatcher] count=… peak_force=…N`；首次过 `crash_threshold_n`（默认 5 N）发布 `GzString "crash"` 到 `~/hover_pid/disable`；CMakeLists 增 `libContactWatcherPlugin.so`。
  - [`HoverPidPlugin`](gazebo_wind_plugin/HoverPidPlugin.cc) 增 SDF：`<disable_topic>`（默认空）、`<crash_zero_thrust>`（默认 false，向后兼容）；订阅 `disable_topic` 后 `disabled_=true` + 清积分；当 `crash_zero_thrust=true` 时 `OnUpdate` 直接 return，drone 自由落体。
- **drone SDF（pt1/pt2）**：[`iris_wind_quad_hires_pt1_crash`](gazebo_wind_plugin/models/iris_wind_quad_hires_pt1_crash/model.sdf) / [`pt2_hover`](gazebo_wind_plugin/models/iris_wind_quad_hires_pt2_hover/model.sdf) 的 `<collision>` 加 `collide_bitmask=0x01` + ODE 表面调参；新增 `<sensor type="contact">` 挂 `libContactWatcherPlugin.so`；pt1 `crash_threshold_n=5 + crash_zero_thrust=true`，pt2 `crash_threshold_n=20 + crash_zero_thrust=false`（仅日志）。
- **World 物理收紧**：[`guangzhou_demo_pt1_crash.world`](gazebo_wind_plugin/worlds/guangzhou_demo_pt1_crash.world) / [`pt2_hover.world`](gazebo_wind_plugin/worlds/guangzhou_demo_pt2_hover.world) `max_step_size=0.002`、`real_time_update_rate=500`、`<ode><solver type=quick iters=50 sor=1.3>`、`cfm=1e-4 erp=0.6`；pt1 spawn 改 `(1495,1350,80)`（远离当前 box 视觉重叠）。
- **启动脚本**：[`scripts/run_gazebo_guangzhou_demo_pt1_crash.sh`](scripts/run_gazebo_guangzhou_demo_pt1_crash.sh) 新增 `collision-build` 子命令（一键 extract → build → verify → probe；支持 `BUILDINGS_STL`/`LUT_VTI`/`HOTSPOT`/`RADIUS_M` 环境变量；缺 STL 时自动 LUT 兜底）。
- **文档**：[`gazebo_wind_plugin/README.md`](gazebo_wind_plugin/README.md) 新增「Collision pipeline (RBM-4 pt1/pt2)」一节 + ContactWatcher SDF 表 + HoverPid `disable_topic`/`crash_zero_thrust` 表项。
- **依赖修复**：本机 numpy 2.2.6 (`~/.local`) 与系统 scipy 1.8.0 (`/usr/lib/python3/dist-packages`) ABI 冲突 → `pip install --user --upgrade 'scipy>=1.13'` 装入 scipy 1.15.3 覆盖系统版（trimesh 间接 import scipy 才能加载）。
- **probe 增强**：[`probe_wind_at_hotspot.py`](scripts/probe_wind_at_hotspot.py) 在 `--manifest` 模式下做 raycast，沿 +upwind 方向把 spawn 推到 union XY bbox 外，再加 `--upwind-clearance-m`，避免 spawn 落在 collision box 内（之前 LUT 兜底产生大 box 导致 spawn 内嵌的失效）。
- **pt1 调参**：[`iris_wind_quad_hires_pt1_crash`](gazebo_wind_plugin/models/iris_wind_quad_hires_pt1_crash/model.sdf) `<area>` 0.04 → **0.5**、`<C_D>` 1.0 → **1.2**（pt2 不变；让 ~1 m/s 风能在 demo 时间窗内推动 1.5 kg drone）；spawn `(1484.23, 1310.86, 80)` 紧贴 box south face 外 ~3 m。
- **smoke 验证（pt1，timeout 30s）**：日志依次出现 `[ContactWatcher] CRASH peak_force=507.5N` → `disable msg published on ~/hover_pid/disable` → `[HoverPidPlugin] disabled by 'crash' (crash_zero_thrust=1)` → 后续 `b=ground_plane::link::collision`（drone 落地）。
- **smoke 验证（pt2，timeout 22s）**：HoverPid `hover_error` 收敛到 XY≈0，`roll/pitch` 在 ±0.3° 微抖，无 CRASH，无 disable。
- **后续**：若拿到真实 `buildings.stl`，`BUILDINGS_STL=<path> ./scripts/run_gazebo_guangzhou_demo_pt1_crash.sh collision-build` 即可重生成更精细的 CoACD 凸壳；spawn 与 area 也可按真几何重新 probe。

### hires 热点 / 目标 / 出生高度改为 80 m
- **范围**：`iris_wind_quad_hires_demo` 与 `iris_wind_quad_hires_pt1_crash` / `pt2_hover` 的 `hotspot_z`、`target_z`；`guangzhou_wind_hires_demo.world` 与 RBM-4 两 world 的机体 spawn；GUI 相机 Z 随热点 +40 m；`demo_building_collision` 模型整体上移 +68 m 以保持与旧 12 m 热点相对几何。

### RBM-4：hires 两段式 world + 建筑碰撞代理 + HoverPid Z 增益
- **依据**：[docs/ops/RBM-4-feedback-to-gazebo_guangzhou_wind_hires_demo.md](docs/ops/RBM-4-feedback-to-gazebo_guangzhou_wind_hires_demo.md)。
- **HoverPidPlugin**：[`HoverPidPlugin.{hh,cc}`](gazebo_wind_plugin/HoverPidPlugin.hh) 增加可选 `kp_z`/`ki_z`/`kd_z`（缺省等于 `kp`/`ki`/`kd`）；周期日志增加 `roll`/`pitch`（度）。
- **hires 机体**：[`iris_wind_quad_hires_demo/model.sdf`](gazebo_wind_plugin/models/iris_wind_quad_hires_demo/model.sdf) 增加 `<velocity_decay><angular>0.4</angular></velocity_decay>`；`wind_torque_arm_z` 改为 `0.3`。
- **轻量模型**：[`iris_wind_quad_hires_pt1_crash`](gazebo_wind_plugin/models/iris_wind_quad_hires_pt1_crash/model.sdf)（`enable_xy=false`、`drift_after_seconds=0`）、[`iris_wind_quad_hires_pt2_hover`](gazebo_wind_plugin/models/iris_wind_quad_hires_pt2_hover/model.sdf)（`kp_z/ki_z/kd_z`、`attitude_kp=8`、`drift_after_seconds=-1`；SDF 内注释 stub PID）；mesh URI 指向 `iris_wind_quad_hires_demo/meshes/`。
- **碰撞占位**：[`demo_building_collision`](gazebo_wind_plugin/models/demo_building_collision/model.sdf) 默认 **box** 静态体（热点 (1470,1350,80) 附近）；`meshes/.gitkeep` 供后续 `building_A.stl`。
- **Worlds**：[`guangzhou_demo_pt1_crash.world`](gazebo_wind_plugin/worlds/guangzhou_demo_pt1_crash.world)、[`guangzhou_demo_pt2_hover.world`](gazebo_wind_plugin/worlds/guangzhou_demo_pt2_hover.world)；GUI 相机约相对热点 (-30,-50,+40)。
- **脚本**：[`scripts/extract_demo_building_collision.py`](scripts/extract_demo_building_collision.py)（`trimesh`+`numpy`，50 m 面过滤、连通分量导出、`--emit-bbox-only`）；输出目录 [`data/demo_assets/`](data/demo_assets/.gitkeep)。
- **启动脚本**（与 [`scripts/run_gazebo_guangzhou_wind_hires_demo.sh`](scripts/run_gazebo_guangzhou_wind_hires_demo.sh) 同结构）：[`scripts/run_gazebo_guangzhou_demo_pt1_crash.sh`](scripts/run_gazebo_guangzhou_demo_pt1_crash.sh)、[`scripts/run_gazebo_guangzhou_demo_pt2_hover.sh`](scripts/run_gazebo_guangzhou_demo_pt2_hover.sh)。
- **验证**：`cmake --build gazebo_wind_plugin/build -j`；建议 `timeout 20s gzserver .../guangzhou_demo_pt2_hover.world --verbose`，日志中查找 `roll=`、`pitch=`、`[WindFieldPlugin]` / `[HoverPidPlugin]`。

### gazebo_wind_plugin 英文 README
- **交付**：新增 [`gazebo_wind_plugin/README.md`](gazebo_wind_plugin/README.md)（英文），汇总 CMake 依赖、三插件 SDF 参数默认值、`WindLUT` JSON/VTI 字段、可选掩膜与热点吸附、`worlds/` 与 `models/` 目录职责、`GAZEBO_PLUGIN_PATH` / `GAZEBO_MODEL_PATH` 与示例启动命令；链到 [`docs/ops/Gazebo风场插件gazebo_wind_plugin与Demo详解.md`](docs/ops/Gazebo风场插件gazebo_wind_plugin与Demo详解.md) 作长文补充。
- **范围**：仅文档；未改插件源码或 CMake。`build/` 与 mesh 二进制归类说明，不逐文件枚举。

## 2026-05-09

### WindFieldPlugin：热点自动吸附到 LUT 建筑外格点
- **动机**：`(1450,1350,10)` 在 VTI `inside_building` 掩膜内，`hotspot_check` 与插值风速可为 0。
- **实现**：[`lut_reader/WindLUT.{hh,cc}`](gazebo_wind_plugin/lut_reader/WindLUT.hh) 增加 `cellIsOutdoor`、`snapHotspotNearestOutdoor`（固定 `z` 对应层 `iz`，在 XY 上搜最近室外格点；有掩膜优先用掩膜，否则用 `|U|≥min_wind`）；[`WindFieldPlugin.cc`](gazebo_wind_plugin/WindFieldPlugin.cc) 在 `hotspot_check` 前可选吸附。
- **SDF（默认向后兼容）**：`hotspot_snap_outdoor` 默认 `true`；`hotspot_snap_max_radius_m` 默认 `120`；`hotspot_snap_min_wind` 默认 `0.05`（无建筑掩膜时的回退判据）。旧 demo 热点若在室外格点上则不移动。
- **模型**：`iris_wind_quad_hires_demo` 的 `hotspot_*` 已改回 `(1450,1350,10)`，依赖自动吸附产生非零 `hotspot_check`。

### Gazebo hires：四旋翼 10× 可见性 + mesh 风箭头
- **机体**：[`iris_wind_quad_hires_demo/model.sdf`](gazebo_wind_plugin/models/iris_wind_quad_hires_demo/model.sdf) 机身/桨叶 mesh `scale=10`，桨叶局部 pose ×10，碰撞 `4.7×4.7×1.1`，[`TrailMarkerPlugin`](gazebo_wind_plugin/TrailMarkerPlugin.cc) `marker_radius=0.3`（尾迹球小于大机体）。
- **箭头资产**：[`scripts/generate_arrow_unit_stl.py`](scripts/generate_arrow_unit_stl.py) 生成 [`wind_arrow_glyph/meshes/arrow_unit.stl`](gazebo_wind_plugin/models/wind_arrow_glyph/meshes/arrow_unit.stl)（+X 长 1 m，三角网格杆+锥）；[`wind_arrow_glyph`](gazebo_wind_plugin/models/wind_arrow_glyph/) 含 `model.config` / `model.sdf`。
- **生成器**：[`scripts/generate_gazebo_wind_arrows.py`](scripts/generate_gazebo_wind_arrows.py) 的 `mesh` 模式为 **单 link + 多 visual**；默认 bbox **`--step 40`**、`--z-levels` 可多层；**`--mesh-len-*` / `--mesh-thick`** 加大箭长与粗细（远景可见性）。
- **hires 出生点**：`guangzhou_wind_hires_demo` 与 `iris_wind_quad_hires_demo` 目标为 **(1470,1350,80)**（高度由 12 m 调整为 80 m）；`buildings.stl` 在 (1450,1350,10) 附近包进楼体网格，东移 20 m、略抬高以离开建筑视觉体。
- **文档**：[`docs/ops/Gazebo风场插件gazebo_wind_plugin与Demo详解.md`](docs/ops/Gazebo风场插件gazebo_wind_plugin与Demo详解.md) §7.4 与目录索引已更新。

## 2026-05-07

### Gazebo Wind Field Plugin（WSL2 / Gazebo Classic）执行记录
- **目标**：在 Gazebo Classic 11 中实现 `ModelPlugin`，每个 physics step 按 LUT 三线性插值获得 `(u,v,w)` 并施加拖曳型风力：$F = 0.5\\rho C_D A |U_{rel}| U_{rel}\$，用于演示无人机进入文丘里加速区后的可观察偏移。
- **输入数据**：`data/wind_lut/20250903_1400/wind_lut.{json,vti}`（若 WSL2 端 VTK 不可用则回退补导出 `wind_lut.npz` + cnpy）。
- **执行门**：
  - 先运行 `scripts/check_gazebo_env.sh` 确认 Gazebo/VTK/CMake/GCC/OS。
  - 若 `libvtk` 开发库可用 → 优先走 VTI/VTK 读取；否则 → 走 NPZ/cnpy 读取。
- **工程位置**：将在仓库根新建 `gazebo_wind_plugin/`（CMake + 插件 + demo model/world）。
- **环境自检输出（WSL2）**：\n+  - `gazebo --version`: 11.10.2（注意：该命令在本机返回码可能非 0，但输出正常）\n+  - `pkg-config gazebo`: 11.10.2\n+  - `libvtk`: `libvtk9-dev` / `libvtk9.1` 已安装（VTK 9.1）\n+  - `cmake`: 3.22.1\n+  - `gcc`: 11.4.0\n+  - OS: Ubuntu 22.04.5 LTS（WSL2）\n+\n+- **关键发现**：仓库实际位于 Windows 挂载盘（`/mnt/e/WRF-OpenFOAM-Coupling`），从该路径读取 354MB 的 `wind_lut.vti` 会显著拖慢 Gazebo 启动。\n+  - 解决：将 LUT 复制到 WSL 根文件系统缓存目录 `~/wrf_openfoam_coupling_cache/wind_lut/20250903_1400/`，并让模型 SDF 指向缓存路径。

- **构建（WSL2）**：
  - `cmake -S gazebo_wind_plugin -B gazebo_wind_plugin/build`
  - `cmake --build gazebo_wind_plugin/build -j`
  - 产物：`gazebo_wind_plugin/build/libWindFieldPlugin.so`、`gazebo_wind_plugin/build/libHoverPidPlugin.so`

- **HoverPid + 刚体 demo（续）**：
  - 新增 [`gazebo_wind_plugin/HoverPidPlugin.hh`](gazebo_wind_plugin/HoverPidPlugin.hh) / [`HoverPidPlugin.cc`](gazebo_wind_plugin/HoverPidPlugin.cc)：世界系位置 PID，对 `base_link` 施力；默认目标 `(1420,-880,50)`，`Kp/Ki/Kd=8/0.1/4`，每 250 步打印 `hover_error=` 与 `hover_force=`。
  - [`WindFieldPlugin.cc`](gazebo_wind_plugin/WindFieldPlugin.cc)：LUT 加载后打印 `hotspot_check LUT(1420,-880,145)` 的 `wind=` 与 `|U|`（三线性插值；该网格点与 QC 文档热点 `(1416,-881)` 不同，`|U|` 数值不必等于 4.81 m/s）。
  - [`models/iris_wind_demo/model.sdf`](gazebo_wind_plugin/models/iris_wind_demo/model.sdf)：`base_link` 质量 1.5 kg，惯量 `ixx=iyy=0.0347`、`izz=0.0617`；碰撞/可视用 `0.28×0.28×0.08` box；同时挂载风场与 Hover PID 插件。
  - 运行后插件消息写入 `~/.gazebo/server-11345/default.log`（终端可能只见 OpenAL 告警）；需同时看到 `wind=` 与 `hover_error=`。`gzserver --iters 500` 在本机仍可能不自动退出，可用 `timeout` 结束；未见仿真崩溃。
  - **GUI 微扰可见性（2026-05-07）**：`guangzhou_wind.world` 增加 `model://sun` 与 `model://ground_plane`（需 `GAZEBO_MODEL_PATH` 含 `/usr/share/gazebo-11/models` 且 `GAZEBO_MODEL_DATABASE_URI=""`）；`iris_wind_demo` 在 `base_link` 上增加**仅视觉**的放大蓝 box + 白色高杆 `visual_mast`（碰撞仍为小 box，物理不变），便于在 Follow/远景下看到小幅位姿/位置变化。

- **验证（headless）**：
  - 关键环境：
    - `export GAZEBO_PLUGIN_PATH="$PWD/gazebo_wind_plugin/build:${GAZEBO_PLUGIN_PATH:-}"`
    - `export GAZEBO_MODEL_PATH="$PWD/gazebo_wind_plugin/models:${GAZEBO_MODEL_PATH:-}"`
    - `export GAZEBO_MODEL_DATABASE_URI=""`（避免联网拉模型库）
  - 运行：`gazebo gazebo_wind_plugin/worlds/guangzhou_wind.world --verbose`
  - 观察到日志：`[WindFieldPlugin] LUT loaded: dims=(501,501,101) ...`（证明 VTI + JSON 读取成功，插件成功 Load）
  - 备注：`gazebo --iters N` 在本机环境下未按预期自动退出（可用 `timeout` / 手动 kill 做“有限步”验证）。

### Gazebo GUI「静止平面 + 细圆柱」深度诊断与文档闭环

- **触发问题**：Gazebo GUI 中只能看到一个静止 `ground_plane` 与一根细圆柱，疑似“仿真没动 / 插件没加载”。
- **关键结论**：
  - 这符合当前 demo 资产设计：`iris_wind_demo` 不是完整无人机，而是极简风场探针；细圆柱是 `model.sdf` 中的 `visual_mast`（仅视觉）。
  - `WindFieldPlugin` 与 `HoverPidPlugin` 在日志中持续输出，证明链路在跑；稳态时 PID 的水平力几乎抵消风力，肉眼难以观察漂移。
  - 当前两插件均通过 `link->AddForce(...)` 施力，无偏心施力点/力矩 → 杆不会明显倾斜，进一步增强“看似静止”。
- **证据抓手（可复现）**：
  - 运行日志：`~/.gazebo/server-*/default.log` 中同时出现 `LUT loaded`、周期 `wind=`、周期 `hover_error=`。
  - 位姿采样：`gz model -m iris_wind_demo -p` 连续采样位姿，确认模型处于稳态锁定。
- **文档产出**：
  - 将原“对话式记录”重构为诊断闭环文档：[`docs/ops/搭建RBM仿真环境Gazebo-2.md`](docs/ops/搭建RBM仿真环境Gazebo-2.md)
  - 文档新增：现象复盘、证据链、根因分层、最小排障分支（症状→判断→动作）、三层演示模式与工程化改进清单。

### Gazebo 视觉 Demo（四旋翼 + 建筑 + 风箭头 + 尾迹 + 可见漂移）

- **目标**：从“日志正常但视觉为 0”升级到可录视频：四旋翼外观、建筑街谷参照、静态风矢量箭头、轨迹尾迹，同时保留 `WindFieldPlugin`/`HoverPidPlugin` 证据链。
- **新增/引入模型资源**：
  - `gazebo_wind_plugin/models/px4_iris_assets/`：稀疏克隆 `PX4/PX4-SITL_gazebo-classic` 的 Iris 资源（仅用于 mesh 复用）。
  - `gazebo_wind_plugin/models/iris_wind_quad/`：本地四旋翼外观模型（机身 `iris.stl` + 桨叶 DAE），并挂载风场、悬停 PID 与尾迹插件。
  - `gazebo_wind_plugin/models/guangzhou_buildings/`：将 `constant/triSurface/buildings.stl` 封装为静态 visual-only 模型。
  - `gazebo_wind_plugin/models/wind_arrows_hotspot/`：从 `wind_lut.npz` 采样生成的静态风箭头模型。
- **新增脚本**：
  - `scripts/generate_gazebo_wind_arrows.py`：读取 `data/wind_lut/.../wind_lut.npz`，采样热点附近并生成 `wind_arrows_hotspot` 的 `model.sdf`。
- **新增插件**：
  - `gazebo_wind_plugin/TrailMarkerPlugin.{hh,cc}`：每隔一段时间在轨迹上生成小球，并删除旧点；已加入 `CMakeLists.txt` 并成功编译生成 `libTrailMarkerPlugin.so`。
- **控制模式（可见漂移）**：
  - `HoverPidPlugin` 新增 SDF 参数 `enable_xy`（默认 true）；当 `enable_xy=false` 时仅控制 Z，XY 由风推动形成可见位移。
  - `iris_wind_quad/model.sdf` 已设置 `<enable_xy>false</enable_xy>` 用于 demo。
- **world 更新**：
  - `gazebo_wind_plugin/worlds/guangzhou_wind.world`：替换为 include `iris_wind_quad` 并新增 `guangzhou_buildings` 与 `wind_arrows_hotspot`。
- **headless 验证（证据）**：
  - `gzserver` 输出中出现：
    - `[WindFieldPlugin] LUT loaded ...`、`hotspot_check ...`
    - `[HoverPidPlugin] ... enable_xy=0`
    - `[TrailMarkerPlugin] ...`
    - 且 `pos=(...)` 随时间在 XY 方向明显变化，证明可见漂移模式生效。

### `render_3d_streamlines_v2`：流线密度 + 标注字号（对齐 `docs/image2.png` 审美，相对旧版 `figure1_streamlines_v2`）
- **问题**：默认盒状种子 `10×10×10`、标题字号 34 且随 `_cb_scale` 线性放大 → **流线极密、左上角标题过大易裁切**；色标刻度曾偏小。后又一度 **`4×4×4` + 大步长** → 画面偏稀、折线显直。
- **代码**：[`util/render_3d_streamlines_v2.py`](util/render_3d_streamlines_v2.py)
  - 默认种子 **`7×7×6`（294）**，管半径默认 **0.38 m**（略细，减轻叠加遮挡）。
  - **积分步长**：`--initial-step` 默认 **0.06**（原 0.12），`max_step_length` 仍随 `initial_step×2.2`，折线顶点更密 → **视觉上更柔顺**（耗时增加）。
  - 标题：默认 `--title-font-size 11`，`_text_scale = clip(_cb_scale**0.5, …)`，**取消**对 `_cb_scale` 的近线性放大，减轻 1280/4K 截图标题暴涨。
  - 色标：`title_font_size` / `label_font_size` 用 **独立于标题的 floor**（约 16–19 / 14–16 随 `_text_scale`），`n_labels=5`，条高度与 `position_y` 略抬。
  - 默认标题文案略缩短（第二行 `·` 连接），降低裁切概率。
- **运行**：`python -u util/render_3d_streamlines_v2.py --window-size 1920,1280`
- **本机一次输出（2026-05-06 调密+顺滑后）**：`n_seeds=294` → **`377` 条**流线、**11099** polyline 点；输出 [`results/microhazard/20250903_1400/figure1_streamlines_v2.png`](results/microhazard/20250903_1400/figure1_streamlines_v2.png)（`--initial-step 0.06`）。
- **微调**：更稀 → `--seed-nx/ny/nz` 再减或 `--seed-style plane`；更柔顺 → **`--initial-step 0.04`**；更密 → 再增大种子或 `--tube-radius-m 0.42`；路演近 `docs/image2.png` 无字风格 → `--title " "` 或改脚本支持关闭（当前可用极小字号兜底）。

### Wind LUT for Gazebo（22:00 LST 帧 → 3D 风速查找表，下一任务"Gazebo 风场插件"输入）
- **任务源头**：[`docs/ops/搭建RBM仿真环境Gazebo.md`](docs/ops/搭建RBM仿真环境Gazebo.md) 的"5月6日 — 从 72 个实验中选最佳帧，导出 (u,v,w) 为 3D 查找表（Python 脚本）"。
- **选帧**：`steady_experiments_finer_ABL/20250903_1400_two_boundaries_as_outlet`（UTC 14:00 = 当地 22:00 LST，夜间稳定 ABL 的文丘里加速帧）。
- **脚本**：[`util/export_wind_lut_3d.py`](util/export_wind_lut_3d.py)
  - 读场：`POpenFOAMReader` → 末态时间目录 `5000/` → `internalMesh` block → `cell_data_to_point_data`（与 [`util/render_3d_microhazard_pyvista.py`](util/render_3d_microhazard_pyvista.py) 同款链路；内部网格 6.6M 点 / 5.47M cell）。
  - 重采样：`pv.ImageData(dimensions=(501,501,101), spacing=(10,10,5), origin=(-2500,-2500,0))` → `grid.sample(internal)`（VTK `vtkProbeFilter`）。
  - 建筑掩膜：`select_interior_points`（PyVista 0.48 新 API，旧 `select_enclosed_points` 兜底；两者均用 `constant/triSurface/buildings.stl`）→ 1.11% 点位（281,398 / 25.35M）落在建筑内部，统一置 `U=0`（与 simpleFoam no-slip 墙语义一致，便于插件三线性插值在墙面平滑收敛到 0）。
  - 域外掩膜：`vtkValidPointMask` → 1.16% 点（293,534）位于 CFD 网格之外（建筑表面 sliver / 网格洞），置 `valid_mask=0` 供插件守门。
- **输出（数据三件套）**：`data/wind_lut/20250903_1400/`
  - [`wind_lut.vti`](data/wind_lut/20250903_1400/wind_lut.vti)（370.8 MB，VTK XML ImageData，C++ 端 `vtkXMLImageDataReader::SetFileName(...)` 一行读盘；含 `U` 3-vector + `inside_building` + `valid_mask`）
  - [`wind_lut.npz`](data/wind_lut/20250903_1400/wind_lut.npz)（276.2 MB；`U(501,501,101,3) float32`、`inside_building`、`valid_mask`、`x_coords`/`y_coords`/`z_coords`）
  - [`wind_lut.json`](data/wind_lut/20250903_1400/wind_lut.json)（`origin/spacing/dimensions/units/axes/array_layout/masks/interface_for_gazebo_plugin`）
- **QC（视觉验真）**：`results/wind_lut/20250903_1400/`
  - [`qc_slice_z100m.png`](results/wind_lut/20250903_1400/qc_slice_z100m.png) — 左：LUT 在 `z=100 m` 的 `|U_h|` `pcolormesh`（叠加 `inside_building=1` 等高线 0.5）；右：原 `postProcessing/100m.csv` 同高散点。两者图样一致 → 重采样无误。
  - [`qc_profile_hotspot.png`](results/wind_lut/20250903_1400/qc_profile_hotspot.png) — 在已知热点 `(1416,-881) m`（来自上方 Figure 1 章节）抽 `u/v/w/|U|` 随 z 廓线；本帧 `|U|` 最大 **4.81 m/s @ z=145 m**，与"夜间峡谷加速"叙事一致。
  - [`qc_summary.json`](results/wind_lut/20250903_1400/qc_summary.json) — `mean(|U|)≈3.17 m/s`、`p95≈5.34 m/s`、`p99≈5.80 m/s`、`max≈9.12 m/s`。
- **接口契约（下一任务 Gazebo 风场插件直接读 JSON）**：
  - **坐标系**：与 OpenFOAM 域同一 ENU 直角坐标，原点 = 5 km×5 km 核心几何中心；Gazebo world frame 一次平移即可对齐。
  - **查表**：插件每个 physics step 用无人机位置 `(x,y,z)` 在 LUT 中三线性插值得 `(u,v,w)`（`U_components = [u_east, v_north, w_up]`）。
  - **越界**：位置在 `[origin, origin+spacing*(dims-1)]` 之外时，LUT 不外推，由插件决定（取 0 / 取 WRF 自由流）。
  - **建筑内**：`inside_building=1` 处 `U=0`；插件可直接返回 0，或交给碰撞模型。
  - **力律**：`F = 0.5 * rho * C_D * A * |U_rel| * U_rel`（`U_rel = U_wind - U_drone`）。
- **运行（Windows + Anaconda Python 3.12 + PyVista 0.48）**：
  - 默认（约 5 min 5 GB 内存）：`python -u util/export_wind_lut_3d.py`
  - 可选裁剪：`--xrange -2000,2000 --yrange -2000,2000 --zrange 0,300 --dx 20 --dy 20 --dz 10` → 体积 ~1/16
  - 兜底：`--building-mask-method distance`（极端情况 `select_*_points` 报错时改用 `compute_implicit_distance < 0` 判内部）
  - 仅产 NPZ：`--no-vti`；仅产 VTI：`--no-npz`；调试 QC 单独跑：`--no-vti --no-npz`（仍会写 JSON + QC）。
- **限制 / 后续可改**：
  - `z_max=500 m` 截断 — 最高 603 m 的塔楼顶尖被切；插件实际飞行高度 < 200 m，影响可忽略；如需扩 z 上限，仅改 `--zrange 0,800`。
  - 单帧稳态 — 后续若需"时变风场"，把多个 `<YYYYMMDD_HHMM>` 算例分别导出 LUT，插件按 wall-clock 切表即可（仍可用本脚本）。

### Slide 1 展示图：09-03 14:00 UTC（当地 22:00 LST）微尺度 hazard + LiDAR 验证快照
- **时刻**：`2025-09-03 14:00 UTC` = `2025-09-03 22:00 LST`（UTC+8）；CFD 算例 `steady_experiments_finer_ABL/20250903_1400_two_boundaries_as_outlet`。
- **Figure 2（散点 + 廓线）**：[`util/plot_validation_scatter_profile.py`](util/plot_validation_scatter_profile.py)
  - **输入**：[`data/260409/processed/merged_lidar_simulation_final.csv`](data/260409/processed/merged_lidar_simulation_final.csv)（四站合并；`ws_cfd = sqrt(u_cfd^2+v_cfd^2)`）。
  - **输出**：[`results/wrf_openfoam/snapshot_20250903_1400_validation.png`](results/wrf_openfoam/snapshot_20250903_1400_validation.png)
  - **运行**：`python util/plot_validation_scatter_profile.py --datetime "2025-09-03 14:00:00" --station GAW111 --out results/wrf_openfoam/snapshot_20250903_1400_validation.png`
  - **本次统计（UTC 该时次、四站所有高度点，N=523）**：
    - CFD vs LiDAR：`R²≈0.024`，`RMSE≈0.985 m/s`，`MBE≈-0.298 m/s`
    - WRF vs LiDAR：`R²≈0.087`，`RMSE≈0.953 m/s`，`MBE≈+0.394 m/s`
- **Figure 1（热点定位 + 真 3D）**：[`util/render_3d_microhazard_pyvista.py`](util/render_3d_microhazard_pyvista.py)
  - **Stage A**：读取 `postProcessing/100m.csv`，在城市核心区 `|x|,|y|<1500 m` 内取 **P99** 水平风速阈值并聚类得热点中心。
  - **本次热点摘要**：`P99≈4.120 m/s`；候选点数 `3600`；热点中心 **`(x0,y0)≈(1416, -881) m`**；默认 zoom：`±200 m`（水平）、`z∈[0,300] m`。
  - **Stage B**：**PyVista**（`POpenFOAMReader` + **水平 `half_w` + 竖向 `min(zmax-m, analysis-z-max-m)`**（默认 **250 m** 峡谷层，见 [`docs/ops/改进RBM-3D可视化-3.md`](docs/ops/改进RBM-3D可视化-3.md)）+ 可选 **`|U|` 等值面**（阈值亦只在峡谷层体上取分位）+ **双水平切片**（主 `slice-z-m` + `z=20 m` 底图；切片先 **cell→point** 减轻马赛克）+ **种子 `5×5×6`、`±50 m`、`z` 顶自动约 `min(150,0.58·z_top)`** + 流线 **`max_length`/`max_time≈150`**）。色标 **`--clim-low-percentile`（默认 5）–`--clim-percentile`（默认 98）** 仅在峡谷 clip 上统计。建筑默认 **`#d3d3d3`、无棱边**；**`--building-wind`** 才采样着色。流线管 **`--tube-radius-m` 默认 0.9**、`U_mag` 上色 + **`set_active_scalars`**；**`--no-streamlines`** 可只做切片+等值面。相机约 **(−400,−400,250)** 尺度。建议先不加 **`--iso-percentiles`** 以免与流线抢视觉。
  - **路演默认 `--require-3d`（可写 `--require-3d` 显式）**：PyVista 导入/读场/渲染失败时 **直接非零退出**，**不**自动落回 matplotlib 2D。仅调试可用：`--no-require-3d`（失败时回退双面板）或 `--fallback-only`（强制 2D，**勿用于路演**）。
  - **输出**：[`results/microhazard/20250903_1400/figure1_streamlines_buildings.png`](results/microhazard/20250903_1400/figure1_streamlines_buildings.png)
  - **运行示例**：
    - `pip install pyvista`（需 VTK 栈；本机已用其跑通 `20250903_1400`）。
    - `python util/render_3d_microhazard_pyvista.py --iso-percentiles 95 --window-size 3840,2160`
    - 更大视野：`--half-width-m 450`（扩大水平裁剪与场景内容）；或保持裁剪仅拉远相机：`--camera-distance-factor 1.5`。
    - **3×3×3 参数网格（27 张 + JSON）**：[`util/batch_figure1_view_grid.py`](util/batch_figure1_view_grid.py) — `half_width_m ∈ {200,400,600}`、`camera_distance_factor ∈ {1.0,1.5,2.0}`、`analysis_z_max_m ∈ {200,280,360}`（`zmax_m` 固定 400 以使竖向顶≈`analysis_z_max_m`）。输出 `results/microhazard/20250903_1400/figure1_grid_hwXXXX_cdfXpY_azZZZZ.png` 与 [`figure1_grid_index.json`](results/microhazard/20250903_1400/figure1_grid_index.json)。运行：`python util/batch_figure1_view_grid.py`；仅列文件名：`--dry-run`。
    - 种子 / 管径：`--seed-box-m`、`--seed-z-max-m`、`--seed-nx/ny/nz`、`--tube-radius-m`；高危风核可加 **`--iso-percentiles 95`**。
    - 高分辨率默认提示见 `--window-size` 帮助文案。
  - **Plan B（仍为 3D，不经本脚本 2D）**：若 `POpenFOAMReader` 无法读 polyhedral/时间目录，在算例根执行 OpenFOAM **`foamToVTK -latestTime`**（或 ParaView 直接打开 case），用 **ParaView / `pvpython`**：`Clip` + `Stream Tracer` + STL + `Save Screenshot`。
- **备注**：流线调用已按新版 PyVista 使用 **`max_length`** 替代已弃用的 `max_time`。
