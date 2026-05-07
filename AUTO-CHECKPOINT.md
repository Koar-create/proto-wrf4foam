# AUTO-CHECKPOINT

> 本文件由Cursor智能体自动维护，用于记录本仓库内任务执行过程、关键决策与可复现命令。

## 2026-05-06

### Gazebo Wind Field Plugin（WSL2 / Gazebo Classic）执行记录
- **目标**：在 Gazebo Classic 11 中实现 `ModelPlugin`，每个 physics step 按 LUT 三线性插值获得 `(u,v,w)` 并施加拖曳型风力：\(F = 0.5\\rho C_D A |U_{rel}| U_{rel}\\)，用于演示无人机进入文丘里加速区后的可观察偏移。
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
  - 运行：`gzserver gazebo_wind_plugin/worlds/guangzhou_wind.world --verbose`
  - 观察到日志：`[WindFieldPlugin] LUT loaded: dims=(501,501,101) ...`（证明 VTI + JSON 读取成功，插件成功 Load）
  - 备注：`gzserver --iters N` 在本机环境下未按预期自动退出（可用 `timeout` / 手动 kill 做“有限步”验证）。

## 2026-05-07

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

### WSL2 转发 Gazebo GUI 帧率低（澄清 + 推荐运行方式）

#### 关键澄清

- 你目前的命令是：
  - `gazebo <world> --verbose`
- **这不是“只跑服务器”**：`gazebo` 会在 WSL2 里同时启动 **`gzserver`（物理/仿真）** + **`gzclient`（GUI）**，因此 Windows 桌面弹出的窗口实际上是 **WSLg 转发的 Linux GUI**。

#### 正确的“WSL2 只跑服务器”

在 WSL2 里用 `gzserver`（或 `gazebo -s`）启动 world，确保不再弹 GUI：

```bash
export GAZEBO_PLUGIN_PATH="$HOME/WRF-OpenFOAM-Coupling/gazebo_wind_plugin/build:${GAZEBO_PLUGIN_PATH:-}"
export GAZEBO_MODEL_PATH="$HOME/WRF-OpenFOAM-Coupling/gazebo_wind_plugin/models:/usr/share/gazebo-11/models:${GAZEBO_MODEL_PATH:-}"
export GAZEBO_MODEL_DATABASE_URI=""

gzserver "$HOME/WRF-OpenFOAM-Coupling/gazebo_wind_plugin/worlds/guangzhou_wind.world" --verbose
```

可选：若要显式指定 master：

```bash
export GAZEBO_MASTER_URI="http://0.0.0.0:11345"
export GAZEBO_IP="$(hostname -I | awk '{print $1}')"
```

#### “Windows 原生跑 GUI”（推荐提帧）

- 在 Windows 上安装 Gazebo Classic 11（原生 Windows 版本），然后在 Windows 侧启动 `gzclient` / `gazebo`（仅 GUI），通过 `GAZEBO_MASTER_URI` 连接到 WSL2 的 `gzserver`。
- WSL2 当前 IP 可用 `hostname -I` 获取（例：`172.19.233.80`）。
- Windows 侧环境变量示例（PowerShell）：

```powershell
$env:GAZEBO_MASTER_URI="http://172.19.233.80:11345"
gzclient
```

> 注：若 Windows 侧无法直连该 IP/端口，需要检查 Windows 防火墙，或用 `netsh interface portproxy` 将 `127.0.0.1:11345` 转发到 WSL2 IP。
- **`render_3d_streamlines_v2`（色标 + 对齐 image2 迭代）**：[`util/render_3d_streamlines_v2.py`](util/render_3d_streamlines_v2.py)
  - **TypeError**：`scalar_bar_args` 已去掉不兼容的 `title_font_family` / `label_font_family`。
  - **视觉迭代（相对 docs/image2）**：默认 **暗色背景** `#1e1e22`、建筑 `#5c5c62`、地面 `#3a3a40`；**热点盒状种子** **`7×7×6`**（`--seed-style box`，可调）；**双向积分** `both`、**细管** `tube-radius-m` 默认 **0.38**、`tube-sides` 32、`tube-opacity` 0.88；**`--initial-step` 默认 0.06**（顺滑折线）；**plane** 种子改为热点 **上游** `x` 而非 `xmin+5`；相机距离系数 **2.35×half_w**；标题/色标字号 **次线性**随分辨率缩放，色标单独提高可读性。
  - **`--light-theme`**：恢复浅灰背景 + 浅建筑（旧风格）。
  - **示例**：`python util/render_3d_streamlines_v2.py --window-size 3840,2160 --ground-plane` → [`figure1_streamlines_v2.png`](results/microhazard/20250903_1400/figure1_streamlines_v2.png)（默认种子与步长见上一节；早期 `10³` 种子曾达 **~1170** 条线量级）。

### 批量：100m 切片水平风速空间 95% 分位数（排除敏感性算例）
- **脚本**：[`util/compute_100m_spatial_p95_windspeed.py`](util/compute_100m_spatial_p95_windspeed.py)
- **输入**：`steady_experiments_finer_ABL/<case_id>/postProcessing/100m.csv`；**排除**目录名含 `fvOpt_sensitivity_run` 的算例（与 `docs/project/Global_Constraints.md` 中 39 个敏感性实验一致）。
- **计算**：水平风速 `sqrt(U:0^2 + U:1^2)`，对切片上全部点取 **第 95 百分位**；时间从 `<case_id>` 前缀 `YYYYMMDD_HHMM` 解析为 UTC。
- **默认输出**：`results/wrf_openfoam/steady_ABL_100m_spatial_p95_windspeed.csv`（列 `time_utc`, `wind_speed_p95_m_s`）。
- **运行示例**：
  - `python util/compute_100m_spatial_p95_windspeed.py`
  - `python util/compute_100m_spatial_p95_windspeed.py --root steady_experiments_finer_ABL --out results/wrf_openfoam/my_p95.csv`
  - 若需遇错即失败：`python util/compute_100m_spatial_p95_windspeed.py --strict`
- **说明**：`steady_experiments_finer_ABL/` 常被 `.gitignore` 忽略，需在本地存在算例树后再运行。

## 2026-05-03

### 英文 `README.md`（仓库说明）
- **输入**：用户要求为本项目撰写英文 `README.md`。
- **产出**：根目录新增 `README.md`，概述 WRF–OpenFOAM 耦合与城市 CFD、目录结构（`scripts/` / `util/` / `data/` / `analysis/` / `results/` / `surrogate_dataset/`）、surrogate 流水线（stage2 + task1–5）、观测合并与验证脚本入口、依赖说明与数据路径引用（指向 `.cursor/skills/project-layout-data-results-analysis/SKILL.md`）。
- **说明**：任务脚本在磁盘上位于 `scripts/`（与 `AUTO-CHECKPOINT` 中部分历史记录里的 `util/task*.py` 表述可能不一致时，以当前仓库实际路径为准）。

### `README.md` 补充：`steady_experiments_finer_ABL`、`docs`、`W_myExp03`
- **输入**：用户反馈首版 README 未突出三条最重要路径。
- **更新**：在 `README.md` 增加 **“The three roots that tie everything together”** 专节：分别说明 `steady_experiments_finer_ABL/`（稳态算例库、`boundaryData`、`processed_hdf5`、组织用 CSV）、`W_myExp03/`（WRF auxhist 与边界前处理侧车目录、常见硬编码路径）、`docs/`（`Global_Constraints`、methodology、ops、reference-candidate）；注明部分 clone 中 `docs/` 等可能被 `.gitignore` 排除但仍为工作流所需。表格下增加指向该节的交叉引用；Surrogate 与 WRF→CFD 小节用语与上述路径对齐。

### util 目录平台粗分类（排除 old_scripts）
- **输入**：遍历 `util/`（忽略 `util/old_scripts/`），将每个文件归为：1 仅 Windows、2 仅 Linux（此处按「POSIX/HOME 路径布局」理解）、3 其他。
- **产出**：`util/platform_classification.json`（含判定口径、`summary`、逐文件 `category` 与 `reason`）。

### `scripts/` 平台粗分类（排除 csv）
- **范围**：`scripts/` 下除 `.csv` 外的脚本（如 `.py`、`.sh`、`.bat`）。
- **产出**：`scripts/platform_classification.json`。

### `merge_lidar_data.py` 迁至 `scripts/` 与数据路径
- **移动**：`analysis/260409/merge_lidar_data.py` → `scripts/merge_lidar_data.py`（原路径文件已删除）。
- **默认路径**（均相对仓库根 `Path(__file__).resolve().parents[1]`，无盘符/HOME）：
  - CFD：`data/260409/raw/cfd/control`
  - WRF：`data/260409/raw/wrf/WRF_lidar_simulation_1h-rolling.csv`
  - LiDAR：`data/260409/raw/lidar/lidar_1h-rolling.csv`
  - 输出：`data/260409/processed/merged_lidar_simulation_final.csv`（写入前 `mkdir` 父目录）
- **运行**：`python scripts/merge_lidar_data.py`；可用 `--cfd-dir/--wrf-csv/--lidar-csv/--output` 覆盖。

## 2026-04-29

### 启动
- **输入**：读取 `待使用的指令1.md` 并按其任务拆分开始执行。
- **任务目标**：将 `steady_experiments_finer_ABL/` 下的OpenFOAM结果与入流条件、几何编码整合为 `surrogate_dataset/`，用于3D神经网络训练。

### 读入任务定义（`待使用的指令1.md` 摘要）
- **任务一**：从 `constant/triSurface/buildings.stl` 生成与训练网格对齐的几何编码（UDF截断200m + 建筑占位mask），输出 `surrogate_dataset/geometry/building_encoding_131x131x15.npy` 并产出QC图。
- **任务二**：扫描111个case，检查 `processed_hdf5/<case_id>.h5` 是否存在且>1MB，集中拷贝到 `surrogate_dataset/fields/<case_id>.h5` 并汇总缺失列表。
- **任务三**：解析各case `constant/boundaryData/{east,south}/...`，提取U_ref、wind_dir、k_max、k_500m、z_kmax与ABL分类，写入 `surrogate_dataset/inflow/<case_id>_inflow.json`。
- **任务四**：汇总生成 `surrogate_dataset/index.csv`，并按ABL分层抽样划分 train/val/test（80/10/10，且每类val/test至少1个）。

### 当前状态
- **进度**：已读取任务说明，待遍历项目目录并开始实现/执行任务二（集中汇总字段HDF5）作为第一步落地。

### 任务二落地（集中汇总 fields）
- **新增脚本**：`util/task2_collect_fields.py`
- **用途**：扫描 `steady_experiments_finer_ABL/<case_id>/processed_hdf5/<case_id>.h5`，过滤掉小于1MB或缺失的文件，并集中拷贝到 `surrogate_dataset/fields/`。
- **忽略规则**：跳过 `-fvOpt_sensitivity_run-002` 与 `-fvOpt_sensitivity_run-005`。
- **可复现运行**：
  - Dry-run：
    - `python util/task2_collect_fields.py --dry-run`
  - 实际拷贝：
    - `python util/task2_collect_fields.py`
  - 输出报告（JSON）：
    - 默认写入 `surrogate_dataset/fields/_task2_report.json`

- **执行记录**：
  - 运行：`python util/task2_collect_fields.py`
  - 结果：扫描匹配case **111** 个，拷贝有效HDF5 **111** 个，缺失/过小 **0** 个；跳过不匹配/被忽略目录 **7** 个。

### 任务三落地（提取入流条件向量）
- **新增脚本**：`util/task3_extract_inflow.py`
- **用途**：解析各case的 `constant/boundaryData/{east,south}/points` 与 `0/{U,k}`，按z高度求平均廓线，计算 `U_ref`、`wind_dir`、`k_max`、`k_500m`、`z_kmax` 与 `ABL_class`，输出到 `surrogate_dataset/inflow/<case_id>_inflow.json`。
- **可复现运行**：
  - Dry-run：
    - `python util/task3_extract_inflow.py --dry-run`
  - 实际生成JSON：
    - `python util/task3_extract_inflow.py`

- **执行记录**：
  - 运行：`python -u util/task3_extract_inflow.py`
  - 结果：选中case **111** 个，成功生成 **111** 个入流JSON；缺失 `boundaryData` 的case **0** 个。
  - 备注：后续发现风向需按气象学“来向”定义修正（\( \theta_{from}=(\theta_{to}+180)\\%360 \)，其中 \(\\theta_{to}=\\mathrm{atan2}(u,v)\)）。
  - 更新：增加 `LLJ_detected` 与 `LLJ_diag`（基于速度廓线的最大垂直切变阈值；阈值参考 `steady_experiments_finer_ABL/WRF Atmospheric Stability Data Organization.csv` 的 “Yes (Strong Shear)” 校准）。

### 任务一落地（建筑几何编码：UDF + 占位mask）
- **新增脚本**：`util/task1_building_encoding.py`
- **用途**：从 `constant/triSurface/buildings.stl` 生成训练网格对齐的几何编码：
  - 通道0：UDF（到最近三角面距离，截断到200m）
  - 通道1：mask（对每个(x,y)列自上而下射线取屋顶最高命中点 z_roof，若 z < z_roof 则占位）
- **可复现运行**：
  - `python util/task1_building_encoding.py`
  - 输出：`surrogate_dataset/geometry/building_encoding_131x131x15.npy` 与 `surrogate_dataset/geometry/building_encoding_qc.png`

### 任务四落地（生成主索引 index.csv）
- **新增脚本**：`util/task4_make_index.py`
- **用途**：读取 `surrogate_dataset/fields/*.h5` 与 `surrogate_dataset/inflow/*_inflow.json`，生成 `surrogate_dataset/index.csv`，并按 `ABL_class` 分层抽样划分 `train/val/test`（整体约80/10/10，且每类val/test至少各1个）。
- **可复现运行**：
  - `python -u util/task4_make_index.py`
- **执行记录**：
  - 输出：`surrogate_dataset/index.csv`
  - split统计：train=89, val=11, test=11（总计111）

### 数据集目录文档
- **新增**：`surrogate_dataset/README_产出清单.md`（罗列产出与复现命令）

### 方法论文档：ABL稳定度与LLJ判据（中文版本）
- **新增**：`docs/methodology/abl_stability_and_llj_detection_zh.md`
- **用途**：为“基于 boundaryData 的稳定度诊断与 LLJ 检测”提供可复用的中文学术写作片段，与英文版保持章节与公式一致，便于在报告/论文/说明文档中引用。

### WRF 稳定度组织表四面板图复现
- **新增脚本**：`util/plot_wrf_stability_organization_csv.py`
- **输入**：`steady_experiments_finer_ABL/WRF Atmospheric Stability Data Organization.csv`
- **输出**：默认 `steady_experiments_finer_ABL/wrf_atmospheric_stability_organization.png`
- **说明**：首图稳定度为 East/South 两边界序数（不稳定=0、中性=1、强稳定=2）融合：均值 <0.5 为不稳定，>1.5 为强稳定，否则为中性过渡；LLJ 白点为任一边界为 Yes 即打点；下方三图为双线（East 实线蓝、South 虚线橙），第三面板为对数轴。
- **可复现运行**：
  - `python util/plot_wrf_stability_organization_csv.py`
  - 指定路径：`python util/plot_wrf_stability_organization_csv.py --csv "steady_experiments_finer_ABL/WRF Atmospheric Stability Data Organization.csv" --out steady_experiments_finer_ABL/my_plot.png`

## 2026-04-30

### 项目文件整理（data/results/analysis）与知识卡片Skill
- **背景**：用户已将原先的 `260409/` 与 `260413-sensitivity_run_analysis/` 下数据与图件移动到统一结构：`data/`、`analysis/`、`results/`。
- **目录现状**：
  - 数据：`data/260409/{raw,processed}`、`data/260413/processed`
  - 分析：`analysis/260409`、`analysis/260413-sensitivity`
  - 结果：`results/{hovmoller,taylor_diagram,ws_composite_profile,ws_station_profile}/<batch>/`
- **新增Skill（项目级）**：`.cursor/skills/project-layout-data-results-analysis/SKILL.md`
  - **用途**：固化“新位置地图”（关键CSV/PNG/ipynb）与结果产出约定；当脚本默认路径仍指向旧目录时，建议显式传 `--csv/--out` 使用新路径。

### 论文/演示选择性解读：SOWFA 一向耦合（WRF→LES→OpenFOAM）对 RANS 实验的启发
- **输入**：
  - 你的研究约束：`docs/project/Global_Constraints.md`
  - 候选参考：`docs/reference-candidate/SOWFA.pdf`（NREL/PR-5000-61122, 2013-10, Churchfield 等）
- **任务目标**：从 SOWFA 多尺度耦合经验中，抽取对“WRF 驱动稳态 RANS 城市/复杂地形下风场评估”最有迁移价值的信息，并形成可执行改进点（边界条件、近地层、湍流输入、域/采样设计、误差诊断）。
- **关键摘录（将用于输出总结）**：
  - 一向耦合流程：运行 WRF 与 WRF-LES；把时间序列插值到 OpenFOAM 边界位置并初始化内场；OpenFOAM 以 WRF-LES 的初场与边界驱动继续发展。
  - 边界条件思想：侧边界对 \(U,T\) 等混合 Dirichlet/Neumann；压力多为 Neumann；地表可由“表面应力模型”与“地表热通量”驱动（强调与上游模型的一致性）。
  - 湍流“发展距离/时间”显著：该案例中高波数能量约需 **1.5 km** 才“填充”，并出现 **overshoot→衰减** 的演化。
  - 近地层失配警示：报告指出 OpenFOAM 域内近地层水平风速随下游距离快速下降，与 WRF-LES 不一致，原因不明——提示耦合链条中“近地层/地表参数化/入口湍流结构”可能是主要误差源。
  - 未来工作方向与可迁移问题：内嵌分辨率、稳定度差异、入流扰动方法、动态 SGS 是否缓解谱 overshoot。
