# PLUGIN_COMPATIBILITY 后续：桥接实现说明

> 日期：2026-06-24  
> 前置评估：[PLUGIN_COMPATIBILITY.md](PLUGIN_COMPATIBILITY.md)  
> 本窗口实现：HDF5 → 时变 LUT 转换器 + `WindFieldPlugin` 时序扩展

---

## 1. 总结

`synthetic_wind` 与 `gazebo_wind_plugin` 之间的**阻塞级接口缺陷（B1/B2/B3）**已通过「Python 转换器 + C++ 时序 LUT」桥接解决；次级缺陷 **S2/S3/S6** 在转换策略中处理；**S1/S4** 仍搁置。C++ 改动已写入仓库，**编译与 Gazebo 运行验证待在 WSL/Linux 完成**。

---

## 2. 缺陷逐项对照

| 编号 | 缺陷 | 状态 | 解决方式 |
| --- | --- | --- | --- |
| **B1** | 输出格式不匹配（HDF5 vs JSON+VTI） | **已解决** | 新增 [`export_timeseries_lut.py`](export_timeseries_lut.py)：读 `synthetic_<case>.h5`，写出 `wind_lut_timeseries.json` + `frame_%04d.vti` |
| **B2** | 竖向 z 非均匀，插件假设等距 | **已解决** | 转换时线性插值到等距 z（默认 z=0..200 m、`dz=5`）；z=0 前置零风再下插 |
| **B3** | 插件不支持时变 | **已解决** | 新增 `WindLUTSeries` + `WindFieldPlugin` 的 `<lut_manifest>` 路径；`OnUpdate` 用 `SimTime * time_scale` 做帧间线性插值 |
| **S1** | 水平 40 m 分辨率偏粗 | **搁置** | demo 叙事可接受；需重跑合成方可改善 |
| **S2** | 缺建筑掩膜 | **已解决** | 转换器用 `gazebo_wind_plugin/models/guangzhou_buildings/meshes/buildings.stl` 重算 `inside_building`/`valid_mask`，建筑内 `U` 清零 |
| **S3** | 单算例 ~3.2 GB，逐帧导出不现实 | **已解决** | 空间裁剪（hotspot 半宽 600 m → 30×31×41）+ 降帧（`--out-dt 2.0`，默认最多 300 帧）；本机验证导出 60 帧约 8 s |
| **S4** | `mean-interp` 慢且缺方差校准 | **搁置** | 湍流抖振方案不走 `--mean-interp` |
| **S5** | `grid_meta` bbox 矛盾 | **此前已修** | `config.py::grid_meta` 增加 `grid_bbox`（窗口 3 评估时修复） |
| **S6** | origin/z 起点与 config 不一致 | **已解决** | manifest 的 `origin`/`spacing`/`dimensions` 由**重采样+裁剪后**网格写入，不照搬 config |

---

## 3. 新增/修改文件

| 路径 | 作用 |
| --- | --- |
| [`analysis/synthetic_wind/export_timeseries_lut.py`](export_timeseries_lut.py) | HDF5 → 时变 LUT 转换器 |
| [`gazebo_wind_plugin/lut_reader/WindLUT.hh`](../gazebo_wind_plugin/lut_reader/WindLUT.hh) | 新增 `WindLUTSeries` |
| [`gazebo_wind_plugin/lut_reader/WindLUT.cc`](../gazebo_wind_plugin/lut_reader/WindLUT.cc) | 抽取 `querySpatialTrilinear`；实现 manifest 加载与 `query(x,y,z,t)` |
| [`gazebo_wind_plugin/WindFieldPlugin.cc`](../gazebo_wind_plugin/WindFieldPlugin.cc) | `<lut_manifest>` / `time_scale` / `time_loop`；向后兼容单帧 |
| [`gazebo_wind_plugin/models/iris_wind_quad/model.sdf`](../gazebo_wind_plugin/models/iris_wind_quad/model.sdf) | 注释化时变 LUT 示例 |
| [`gazebo_wind_plugin/docs/demo/02-wind-lut-and-sampling.md`](../gazebo_wind_plugin/docs/demo/02-wind-lut-and-sampling.md) | §2.7 时变 LUT 契约 |
| `data/wind_lut_timeseries/20250903_1400/` | 代表算例导出产物（本机验证，60 帧） |

---

## 4. Manifest JSON 字段（C++ 读取）

| 键 | 类型 | 含义 |
| --- | --- | --- |
| `origin` | `[x0,y0,z0]` | 裁剪后等距网格角点 |
| `spacing` | `[dx,dy,dz]` | 默认 `[40,40,5]` |
| `dimensions` | `[Nx,Ny,Nz]` | 裁剪+重采样后尺寸 |
| `dt_s` | 标量 | 帧间隔（秒） |
| `n_frames` | 整数 | 帧数 |
| `frame_pattern` | 字符串 | `printf` 模式，如 `frame_%04d.vti` |
| `loop` | `0`/`1` | 是否时间循环 |

其余字段（`case_id`、`crop_bbox`、`masks` 等）仅供 provenance，插件忽略。

**本机验证样例**（`20250903_1400`，60 帧）：

```json
{
  "origin": [840.0, -1480.0, 0.0],
  "spacing": [40.0, 40.0, 5.0],
  "dimensions": [30, 31, 41],
  "dt_s": 2.0,
  "n_frames": 60,
  "frame_pattern": "frame_%04d.vti",
  "loop": 1
}
```

---

## 5. 本机验证结果（Python 侧）

```bash
python analysis/synthetic_wind/export_timeseries_lut.py \
  --case 20250903_1400_two_boundaries_as_outlet \
  --max-frames 60 --out-dt 2.0
```

| 检查项 | 结果 |
| --- | --- |
| VTI `dimensions` 与 manifest 一致 | 通过（30×31×41） |
| hotspot (1420,-880,145) 室外 | `inside_building=0` |
| 帧间风速变化（湍流抖振） | frame0 vs frame1：\|ΔU\| ≈ 0.063 m/s |
| 建筑内风速 | max \|U\| = 0 |
| 残留 `*.h5.tmp` | 已删除 9 个 |

---

## 6. WSL 待办（C++ 编译与 demo）

```bash
cd gazebo_wind_plugin
cmake -S . -B build && cmake --build build -j
export GAZEBO_PLUGIN_PATH="$(pwd)/build:${GAZEBO_PLUGIN_PATH}"

# 将 LUT 复制到 WSL 原生盘，SDF 中启用：
# <lut_manifest>.../wind_lut_timeseries/20250903_1400/wind_lut_timeseries.json</lut_manifest>
# <time_scale>1</time_scale>
# <time_loop>true</time_loop>

gazebo worlds/guangzhou_wind.world
```

期望启动日志：

```
[WindFieldPlugin] LUT series loaded: dims=(30,31,41) ... frames=60 dt=2 loop=1
[WindFieldPlugin] hotspot_check LUT(...) wind=(...) |U|=... m/s
```

飞行中应观察到随 `SimTime` 变化的阻力抖振（0.5 Hz 有效更新率，帧间线性插值）。

完整 demo 可增大 `--max-frames 300`（约 10 min 湍流片段 @ 2 s 间隔）。

---

## 7. 架构示意

```
synthetic_<case>.h5  ──export_timeseries_lut.py──►  wind_lut_timeseries.json
        │                                              frame_0000.vti …
        │                                                    │
        └─ 非均匀 z, 131², 2 Hz ──重采样/裁剪/降帧/掩膜──►     ▼
                                                    WindLUTSeries::loadFromManifest
                                                    WindFieldPlugin::OnUpdate(t)
```

---

## 7.5 关键修复：LUT 数组布局转置 bug（2026-06-24，WSL 验证时发现）

> 在 WSL 编译/验证阶段发现并修复了一个**既有**（非本次桥接新引入）的严重正确性 bug。

**问题**：`WindLUT` / `WindLUTSeries` 的索引 `idx()` 与 `querySpatialTrilinear` 的 `gridIdx()` 使用 `(ix*Ny+iy)*Nz+iz`（**z 最快**），而 VTK ImageData（及 `export_*_lut.py` 写出的 `.vti`、`vtkXMLImageDataReader` 读回的内存）是 **x 最快** `ix + Nx*(iy + Ny*iz)`。二者对**非立方网格**不一致，使风场沿 z/x 轴转置错乱。

**为何此前没暴露**：原单帧 demo（pt1/pt2/hires）重度依赖 `wind_bias_*` / `force_scale` 人为来流，且从未定量校核过 `hotspot_check` 的 `|U|`，故转置未被察觉。

**实证（`frame_0000.vti`，30×31×41）**：
- 38128/38130 个格点的旧索引指向错误内存位置；
- 导出器在真实建筑格点已正确清零（`max|U|=0`），但旧 z-fastest 索引在真实建筑位置读到 93% 非零风（mean 2.69、max 7.54 m/s）；
- 飞行点 `(1420,-880,80)` 真值 `(-1.42,3.47,-0.79)`，旧索引读成 `(-0.89,1.35,0.04)`。

**修复**：将 `WindLUT.hh` 两处 `idx()`、`WindLUT.cc` 的 `gridIdx()` 及其调用点统一为 x-fastest。

**验证**：
- `cmake --build build -j` 通过（WSL，gazebo 11.10.2 + VTK 9.1）；
- 独立测试 `WindLUTSeries::loadFromManifest` 读入 60 帧；在恰好落在网格节点的 `query(1160,-720,60)` 返回 `(0.501,-0.061,0.494)`，与 Python x-fastest 真值**逐位一致**；帧间时间插值正确。

**影响（需后续处理）**：现有单帧 demo（pt1/pt2/hires）此前是针对**转置后的错误风场**调参的；修复后它们将读到几何正确的风，**行为可能改变，需重新验证/微调**。

---

## 8. 仍搁置项说明

- **S1**：若需街谷热点细节，需在热点区以 `dx=10` 重跑合成或局部精化 LUT。
- **S4**：跨小时 `mean-interp` 非湍流抖振 demo 关键路径。

---

## 9. 端到端冒烟（center-266，2026-06-25，WSL）

专用 world/model 已接入 `data/wind_lut_timeseries/20250903_1200_center-266/`（300 帧，hotspot_xy `[-266.6,-459.1]`，domain x∈[-840,320] y∈[-1040,120] z∈[0,200]）。

**新增文件**：
- `gazebo_wind_plugin/models/iris_wind_quad_timeseries/`（`<lut_manifest>` + `model.config`）
- `gazebo_wind_plugin/worlds/guangzhou_wind_timeseries.world`
- `scripts/run_gazebo_guangzhou_wind_timeseries.sh`（`cache` / `build` / `smoke` 等）

**运行**：`./scripts/run_gazebo_guangzhou_wind_timeseries.sh smoke`（40 s headless gzserver 11.10.2）

### 9.1 时变 LUT 冒烟

启动日志（摘录）：
```
[WindFieldPlugin] LUT series loaded: dims=(30,30,41) origin=(-840,-1040,0) spacing=(40,40,5) frames=300 dt=2 loop=1
[WindFieldPlugin] hotspot_snap_outdoor: (-266.6,-459.1,50) -> (-280,-400,50) within 120 m
[WindFieldPlugin] hotspot_check LUT(-280,-400,50) wind=(0.350217,-0.0170715,-0.137938) |U|=0.376789 m/s
```

**注意**：名义热点 `(-266.6,-459.1)` 落在 `inside_building` 掩膜内；`hotspot_check` 自动 snap 到最近室外格点 `(-280,-400,50)`。悬停 target 亦设于此室外点，否则飞行日志中 `wind=0,0,0`（掩膜清零）。

40 s 飞行中观察到 wind/force 随 SimTime 变化（帧 dt=2 s，帧间线性插值），例如：
```
pos=(-280,-400,49.47) wind=0.349588,-0.00470211,-0.128881 force=(0.016869,-0.000225792,0.094543)
...
pos=(-280,-400,48.52) wind=0.402155,-0.0217255,-0.11239 force=(0.00417207,-0.000225315,-0.00135968)
```
u 分量由 ~0.35 增至 ~0.40（约 20 帧跨度内持续变化），阻力 force 同步响应。

### 9.2 pt2 hover 复核（单帧 hires LUT，索引修复后）

`timeout 75s gzserver gazebo_wind_plugin/worlds/guangzhou_demo_pt2_hover.world --verbose`

```
[WindFieldPlugin] LUT loaded: dims=(501,501,101) origin=(950,850,0) spacing=(2,2,2)
[WindFieldPlugin] hotspot_check LUT(1470,1350,80) wind=(0.00103656,0.046925,0.52137) |U|=0.523478 m/s
[InspectionPathControllerPlugin] waypoints=21 barrier=1 ...
[InspectionPathControllerPlugin] wp=1/21 ... roll=-0.55deg pitch=-0.90deg
```

- 插件正常加载，`hotspot_check |U|≈0.52 m/s`（非零、合理）
- 75 s 内无 `[ContactWatcher] CRASH`
- roll/pitch 稳定在 ±4° 以内（后期 ~±1°），未见索引修复后的明显劣化
- 75 s 内仍停留在 wp=1（距首航点 ~8 m，收敛较慢属预期；未撞楼）

**结论**：x-fastest 索引修复后，时变 LUT 端到端链路可用；pt2 单帧 hires demo 行为正常，暂无需重调参。
