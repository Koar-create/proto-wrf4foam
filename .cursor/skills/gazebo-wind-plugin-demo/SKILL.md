---
name: gazebo-wind-plugin-demo
description: >-
  Knowledge card for the Gazebo Classic wind-field + hover PID demo in this repo:
  original goals, what is implemented now, and what is still unverified.
  Use when the user asks "what's done / what's next" for the wind-plugin demo video.
maintainer: Zhixian Yang <zyang248@connect.hkust-gz.edu.cn>
---

# Gazebo Wind Plugin Demo（Classic 11）知识卡片

**维护者**：Zhixian Yang（zyang248@connect.hkust-gz.edu.cn）

## 最初要实现的工作（目标定义）

- **目标**：在 Gazebo Classic 中开发 `ModelPlugin`，每个 physics step 从 CFD 导出的风场 LUT 做三线性插值，得到无人机当前位置风速 $U_{wind}$，计算相对风 $U_{rel}=U_{wind}-U_{drone}$，并施加风力：
  - $F = 0.5 \rho C_D A |U_{rel}| U_{rel}$
- **演示预期**：无人机在三轴 PID 悬停控制下，进入“文丘里加速区”后出现可观察偏移（适合录 demo 视频）。
- **输入数据契约**：`data/wind_lut/20250903_1400/wind_lut.json` 提供 `origin/spacing/dimensions/axes` 等元数据；数据文件优先走 VTI/VTK（或回退 NPZ/cnpy）。

## 目前已实现的工作（状态：已落地）

### 1) LUT 读取与插值（VTI/VTK 路线）

- **已实现**：读取 `wind_lut.json + wind_lut.vti`，加载 `U`（3 分量）与可选 `valid_mask/inside_building`，并支持三线性插值查询。
- **代码位置**：
  - `gazebo_wind_plugin/lut_reader/WindLUT.hh`
  - `gazebo_wind_plugin/lut_reader/WindLUT.cc`
- **说明**：为了避免从 Windows 挂载盘读取大文件导致启动慢，LUT 固定使用缓存路径：
  - `~/wrf_openfoam_coupling_cache/wind_lut/20250903_1400/wind_lut.{json,vti}`

### 2) 风场施力插件（WindFieldPlugin）

- **已实现**：Gazebo Classic `ModelPlugin`，对指定 `link_name`（默认 `base_link`）施加风力；支持 `world_to_lut_offset_{x,y,z}`；每 N 步打印一次 `wind=` 日志。
- **代码位置**：`gazebo_wind_plugin/WindFieldPlugin.cc`
- **热点校验日志**：插件 Load 后会打印 `hotspot_check LUT(1420,-880,145)` 的 `wind=` 与 `|U|`。

### 3) 三轴独立悬停 PID（HoverPidPlugin）

- **已实现**：对 `base_link` 进行世界坐标系位置 PID（x/y/z 独立），直接 `AddForce`；SDF 可设目标点与增益；每 250 步打印 `hover_error=` 与 `hover_force=`。
- **代码位置**：
  - `gazebo_wind_plugin/HoverPidPlugin.hh`
  - `gazebo_wind_plugin/HoverPidPlugin.cc`
- **默认参数**：`target=(1420,-880,50)`，`Kp/Ki/Kd=8/0.1/4`，`log_every_n=250`。

### 4) Demo world/model（用于 GUI 录屏）

- **world**：`gazebo_wind_plugin/worlds/guangzhou_wind.world`
  - 将模型放在热点附近（坐标量级为 km），并设置初始相机到模型附近。
  - 包含 `model://sun` 与 `model://ground_plane`（需要 `GAZEBO_MODEL_PATH` 包含 `/usr/share/gazebo-11/models`）。
- **model**：`gazebo_wind_plugin/models/iris_wind_demo/model.sdf`
  - 刚体质量 1.5 kg，惯量 `ixx=iyy=0.0347, izz=0.0617`，碰撞为小 box。
  - 视觉增强：大号蓝色 `visual_body_big` + 白色高杆 `visual_mast`（仅视觉，不改变物理），便于看出微扰。
  - 同时挂载 `libWindFieldPlugin.so` 与 `libHoverPidPlugin.so`。

### 5) 构建方式（CMake）

- **已实现**：`gazebo_wind_plugin/CMakeLists.txt` 同时构建
  - `libWindFieldPlugin.so`
  - `libHoverPidPlugin.so`

## 目前未能检验成功 / 仍不确定的内容（待验证）

- **“进入文丘里区产生可观察偏移”的视频级效果**：\n
  目前可确认插件运行与 PID 稳定，但是否达到“肉眼明显偏移、适合 demo 视频”的效果，仍需在 GUI 中通过调参/调高度/调目标点来验证（例如提高风速高度、降低 PID 刚度等）。
- **`gzserver --iters N` 在本机是否会按预期自动退出**：\n
  观察到部分运行场景下仍可能长时间不退出；建议用 `timeout` 做批量验证，GUI 录屏用 `gazebo` 直接跑。
- **热点校验的数值对齐**：\n
  QC 文档中峰值 `|U|≈4.81 m/s @ z=145m` 对应热点 `(1416,-881)` 的廓线峰值；当前日志探针点为 `(1420,-880,145)`，插值 `|U|` 不一定等于 4.81（属于“取点不同”而非 bug）。

## 运行要点（最少命令集）

```bash
export GAZEBO_PLUGIN_PATH=~/WRF-OpenFOAM-Coupling/gazebo_wind_plugin/build:$GAZEBO_PLUGIN_PATH
export GAZEBO_MODEL_PATH=~/WRF-OpenFOAM-Coupling/gazebo_wind_plugin/models:/usr/share/gazebo-11/models:$GAZEBO_MODEL_PATH
export GAZEBO_MODEL_DATABASE_URI=""

gazebo ~/WRF-OpenFOAM-Coupling/gazebo_wind_plugin/worlds/guangzhou_wind.world
```

