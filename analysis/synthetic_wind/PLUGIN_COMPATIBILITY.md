# synthetic_wind 与 gazebo_wind_plugin 适配性评估

> 维护：Zhixian Yang（zyang248@connect.hkust-gz.edu.cn）
> 评估日期：2026-06-24
> 结论：`synthetic_wind` 物理建模合理，**无概念性致命缺陷**，但当前输出**不能被 `WindFieldPlugin` 直接消费**。升级 demo 前需补一层「HDF5→LUT 转换 + 插件时变扩展」桥接。

本文档记录评估结论、已顺手修复项、以及搁置（需较大工作量）的缺陷与桥接方案，供后续升级 demo 时参考。

---

## 1. 两侧数据契约对照

| 维度 | `synthetic_wind` 输出 | `WindFieldPlugin` 期望（`WindLUT`） |
| --- | --- | --- |
| 文件格式 | HDF5 `synthetic_<case>.h5`（`U`,`time_s`,`coords_*`） | 一对 `wind_lut.json`(`origin/spacing/dimensions`) + `wind_lut.vti`(VTK ImageData) |
| 水平网格 | 131×131，`dx=dy=40 m`，范围 ±2600 m | 任意，但**等间距** |
| 竖向网格 | 15 层 **非均匀** `[5,10,20,40,60,80,100,150,200,300,400,500,600,700,800]` | **等间距** `spacing_z`（三线性 `fz=(z−z0)/dz`） |
| 时间 | **时变**，1201 帧 @ 0.5 s（2 Hz，10 min） | **单帧静态**，`Load()` 一次性载入 |
| 掩膜 | 无 | 可选 `inside_building` / `valid_mask` |
| 数组布局 | `(frame,3,ix,iy,iz)`（z 最内） | VTK x-fastest：`(ix*Ny+iy)*Nz+iz` |
| 坐标系 | OpenFOAM 域 ENU 中心系 | 同上（demo `world_to_lut_offset=0`） |

源码定位：插件契约见 `gazebo_wind_plugin/lut_reader/WindLUT.{hh,cc}`、`WindFieldPlugin.cc`、`gazebo_wind_plugin/docs/demo/02-wind-lut-and-sampling.md`；合成侧见 `config.py`、`synthesize.py`、`rfg.py`。

---

## 2. 阻塞级缺陷（不解决 demo 跑不起来）— 搁置，属「桥接」工作

### B1. 输出格式不匹配
插件只读 `JSON+VTI`，合成侧只产 HDF5，仓库无 HDF5→VTI 转换器（`write_openfoam.py` 只导 OpenFOAM 时间目录）。

### B2. 竖向网格非均匀（最严重）
`WindLUT::query()` 假设每轴等间距。合成 Z 层强非均匀，直接当 ImageData 喂会把高度严重错位（demo 飞 80 m 会被映射到错误层），且 z 从 5 m 起、无 z=0。
→ 转换时必须先**竖向重采样到等距 z 网格**（建议对齐原 demo LUT：z 0–500 m、`dz=5`，覆盖飞行高度即可）。

### B3. 插件不支持时变
`WindFieldPlugin` 在 `Load()` 载入单帧、`OnUpdate()` 每步只 query 同一张表。要用 2 Hz 湍流抖振，插件需扩展为「按 wall-clock 在多帧间切换/插值」。这是 `AUTO-CHECKPOINT.md` 早已记录的待办。

---

## 3. 次级缺陷处理结论

> 决策（2026-06-24）：demo 采用**秒级湍流抖振**方案（单一小时状态内 2 Hz 脉动让无人机持续颠簸）。据此分类处理。

| 编号 | 缺陷 | 处理 |
| --- | --- | --- |
| S1 | 水平分辨率 40 m（原 demo 10 m），抹平建筑街谷/文丘里热点 | **搁置**（需重跑合成，开销大） |
| S2 | 缺 `inside_building`/`valid_mask`；脉动在建筑内部也叠加非零风 | **搁置**（在转换器写 VTI 时由 `buildings.stl` 重新生成更自然） |
| S3 | 单算例 ~3.2 GB（1201 帧），逐帧导 VTI 不现实 | **搁置**（属转换器导出策略：降帧/空间裁剪/插件直读 HDF5） |
| S4 | 默认不做跨小时时变；`MeanFieldInterpolator.at_time` 每帧重建全场 PCHIP，慢且占内存 | **搁置**（湍流抖振方案不走 `--mean-interp`，非关键路径） |
| S5 | `grid_meta()` 的 `bbox_clean`（±5100/±5600/0–2100）与实际网格（±2600/z5–800）矛盾，写入每个增强 HDF5 的 meta | **已修**：见 §4 |
| S6 | 坐标系 frame 一致，但 `origin` 数值（-2600 vs -2500）、z 起点不同 | **搁置（设计须知）**：转换器写 JSON 的 `origin/spacing/dimensions` 必须按**重采样后的等距网格**重写，不可照搬 `config` |

### 物理实现层面的小观察（不影响接口，记录备查）
- 脉动空间滤波用**全域 median L** 的各向同性 σ（`rfg.py:_spatial_filter`），AR(1) 用**全域 median T**（`rfg.py:step`）——空间/时间相关被均一化，热点处局地湍流结构不精确（README 已声明非 LES，可接受）。
- `generate_series` 中 `_normalize_variance` 与之后的时间方差校准逻辑略重叠；`mean-interp` 路径缺等价的时间方差校准。
- 数组布局 `(frame,3,ix,iy,iz)` 与 VTK x-fastest 不同：用 PyVista 写 VTI 会自动处理，**手工 flatten 必须遵守 x 最快**，否则风场转置错乱（见 demo 文档 §2.3 警告）。

---

## 4. 本次已顺手修复

1. **清理 29 GB 残留临时文件**：删除 `data/synthetic_wind/fields/*.h5.tmp`（9 个，`synthesize.py` 中断写入产物；均已确认有对应完成版 `.h5`）。
2. **修 S5 元数据矛盾**（`config.py::grid_meta`）：
   - 新增 `grid_bbox`，真实反映采样网格范围（±2600 m、z 5–800、`z_uniform=False`），供下游/转换器使用；
   - 为原 `bbox_clean` 加注释，明确其语义为「完整 CFD 域范围（含 sponge），仅作 provenance，**不**描述采样网格」。
   - 注意：仅对**新写入**的增强 HDF5 生效；已生成文件的旧 meta 不变。

---

## 5. 升级 demo 的桥接清单（秒级湍流抖振方案）

按依赖顺序：

1. **新增转换器** `HDF5 →（竖向重采样到等距 z）→ 生成掩膜 → 写时变 LUT`【解 B1/B2/S2/S6】
   - 竖向：插值到等距 z（如 0–200 m、`dz=5`，覆盖飞行带即可，避免全 0–800）。
   - 掩膜：用 `gazebo_wind_plugin/models/guangzhou_buildings` 的 `buildings.stl` 重算 `inside_building`/`valid_mask`，并把建筑内脉动清零。
   - 输出：建议「一个等距空间网格 + 时间维」的紧凑容器（多帧），而非逐帧 VTI。
2. **扩展 `WindFieldPlugin`** 支持按 wall-clock 取帧/帧间线性插值【解 B3】。
3. **控制体量**【解 S3】：降帧（2 Hz→0.2~0.5 Hz 即可观感颠簸）、空间裁剪到 demo 飞行区域（hotspot 周边数百米）。
4. （可选）**分辨率取舍**【S1】：若叙事需要热点细节，仅在热点区以 `dx=10` 重跑或局部精化。

---

## 6. 复现/核查命令

```bash
# 合成产物清单
ls -la data/synthetic_wind/fields/

# 插件契约参考
sed -n '1,75p' gazebo_wind_plugin/lut_reader/WindLUT.cc   # query() 等距假设
cat data/wind_lut/20250903_1400/wind_lut.json              # 原 demo LUT 的 JSON 契约
```
