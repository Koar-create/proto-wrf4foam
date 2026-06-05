# AUTO-CHECKPOINT

> 本文件由Cursor智能体自动维护，用于记录本仓库内任务执行过程、关键决策与可复现命令。

## 2026-06-02 

### 新建 skill：literature-obs-sim-comparison-analysis

- **路径**：`.cursor/skills/literature-obs-sim-comparison-analysis/`（`SKILL.md` + `reference.md`）
- **参照**：`.cursor/skills/literature-trans-scale-analysis/` 的工作流与 JSONL 落盘格式
- **问题树**：
  - **Q1**：本研究是否有「观测–模拟对比」？
  - **Q2**（Q1=Yes）：观测空间类型 `single_layer` | `multi_layer` | `wind_tunnel`
  - **Q3**（Q1=No）：缺乏观测对比时，作者如何论证研究意义
- **台账**：`docs/reference-candidate/literature-obs-sim-comparison-ledger.jsonl`
- **文章来源（2026-06-02 更新）**：除 PDF 外，支持 `docs/reference-candidate/marker_out/<stem>/` 目录及其中 `.md`；`source` 可含 `source_format`、`markdown_path`、`marker_out_dir`
- **调用示例**：
  ```text
  根据 @.cursor/skills/literature-obs-sim-comparison-analysis/SKILL.md 分析 @docs/reference-candidate/marker_out/1-s2.0-S0360132318302671-main
  ```

### Marker → asr-whisperx-gpu128

- **不在 base 安装** cu128 / marker；base 仍为 `torch 2.12.0+cpu`。
- **asr-whisperx-gpu128** 已具备：`torch 2.8.0+cu128`，RTX 5060 Ti sm_120 冒烟通过。
- 在该环境 `pip install marker-pdf`（1.10.2 + surya-ocr 0.17.1），**未重装 torch**。
- `scripts/marker-env.ps1` + `scripts/marker.ps1`；输出 `docs/reference-candidate/marker_out`。
- 模型缓存仍在 `E:\WRF-OpenFOAM-Coupling\.cache\datalab\models`（`MODEL_CACHE_DIR`）。

#### 使用

```powershell
cd E:\WRF-OpenFOAM-Coupling
.\scripts\marker.ps1 "docs\reference-candidate\direction1-rans-cpratio\<paper>.pdf"
```

或：`conda activate asr-whisperx-gpu128` 后手动 `marker_single`（需自行设置 `MODEL_CACHE_DIR` / 读 `local.env`）。

### OpenDataLoader PDF → asr-whisperx-gpu128

- [opendataloader-pdf](https://github.com/opendataloader-project/opendataloader-pdf)：JVM 本地快速模式（默认不需 GPU）；输出 Markdown/JSON。
- 环境内：`pip install opendataloader-pdf` + `conda install openjdk=17`（`Library\bin\java.exe`）。
- 脚本：`scripts/opendataloader-env.ps1`、`scripts/opendataloader.ps1`；输出 `docs/reference-candidate/opendataloader_out/`。

```powershell
cd E:\WRF-OpenFOAM-Coupling
.\scripts\opendataloader.ps1 "docs\reference-candidate\<paper>.pdf"
```

## 2026-05-14

### 9 月 2 日白昼 CFD 偏差诊断包（`analysis/260514-sep02-daytime-cfd-diagnosis/`）
- **输入**：按计划落实 `docs/ops/diagnose-anomaly-at-day02-daytime.md` 诊断链；数据以 `data/260409/processed/merged_lidar_simulation_final.csv` 为主；OpenFOAM `boundaryData` / `log.simpleFoam` 通过 manifest 指向本机算例（仓库内默认不附带算例树）。
- **产出**：[`analysis/260514-sep02-daytime-cfd-diagnosis/README.md`](analysis/260514-sep02-daytime-cfd-diagnosis/README.md)；脚本 `scripts/metrics_sep02_daytime_vs_neighbors.py`、`profiles_sep02_daytime_composite.py`、`boundarydata_inflow_profiles_compare.py`、`boundarydata_qc_scan.py`、`parse_simplefoam_residuals.py`；共享解析 `scripts/_foam_boundary_io.py`；`example_cases_manifest.csv`（仅表头，需用户填 `case_path`）。
- **实现说明**：`metrics`/`profiles` 通过 `_SCRIPT_DIR.parents[2]` 定位仓库根并 `sys.path` 导入 `analysis/260409/print_metric.py`；Windows 控制台对脚本内关键 `print` 使用英文，避免 cp1252 编码错误。
- **已运行（本机）**：
  - `python analysis/260514-sep02-daytime-cfd-diagnosis/scripts/metrics_sep02_daytime_vs_neighbors.py --out-dir analysis/260514-sep02-daytime-cfd-diagnosis/results`
  - `python analysis/260514-sep02-daytime-cfd-diagnosis/scripts/profiles_sep02_daytime_composite.py --out-dir analysis/260514-sep02-daytime-cfd-diagnosis/results --utc-hours 3 4 5 6`
  - 空 manifest 运行 `parse_simplefoam_residuals.py` 预期退出码 1 并提示 `No valid case paths in manifest.`

### 设计并实现入流扰动层（WRF→OpenFOAM，写出 *_cartesian_perturbed.nc）
- **目标**：在不运行 CFD 的前提下，提供一个可回溯、物理可解释的“入流扰动层”，用于缓解现阶段 CFD 风速系统性偏低（k-ε 过耗散 + mapped-inlet 动量损失）的问题。
- **实现**：
  - 新增 `util/perturb_OF_inlet_data.py`
  - 支持旋钮：
    - A：log-law 比值对 (U,V) 近地层再充气（保持风向，z>zmax 不动）
    - B：按稳定度给 `TKE_PBL` 施加湍强下限（只抬不压）
    - C：可选写出 `eps_override` + `mixingLength_override`（Blackadar 混合长度）
    - D：可选 LLJ nose boost（默认关闭）
  - 输出：在原 `*_cartesian.nc` 旁写 `*_perturbed.nc`（默认文件名），并保留 `U_raw/V_raw/WS_raw/TKE_PBL_raw` 以便回归对比。
- **下一步**：为让 C 旋钮真正影响 `boundaryData/epsilon`，需要在 `util/construct_OF_boundary_arrays.py` 加一段可选读取 `eps_override` 的分支（保持向后兼容）。

### 冒烟验证（仅 NC 读写，不跑 CFD）
- **说明**：`util/construct_OF_boundary_arrays.py` 当前以 Linux `HOME` + 含 `:` 的文件名拼接输入 NC 路径，在 Windows 上无法直接喂入自定义文件路径做端到端运行（文件名包含冒号也无法落盘）。因此本次 smoke 采用“可编译 + 可被 xarray 正常读取 + `eps_override` 变量存在且可切片访问”的验证口径。
- **执行（本机）**：
  - `python -m py_compile util/perturb_OF_inlet_data.py util/construct_OF_boundary_arrays.py`
  - 用合成的小型 `*_cartesian.nc` 运行 `util/perturb_OF_inlet_data.py` 生成 `*_perturbed.nc`，并确认 `eps_override` 可被 `ds.isel(...).eps_override` 访问。

## 2026-05-12

### `Research_Status_202605.md` + 约束文档瘦身（WRF 叙述 / checkMesh 细表外移）
- **输入**：用户要求执行两项工作——（1）将 `analysis/260409/ws_composite_analysis.md` 中的研究结论整理为 `docs/project/Research_Status_YYYYMM.md`（避免对话体、区分已修订信念与当前结论）；（2）将 `Global_Constraints.md` 中适合独立维护的长叙述迁出。
- **产出**：
  - [`docs/project/Research_Status_202605.md`](docs/project/Research_Status_202605.md)：元数据、执行摘要、关键指标表、Superseded 表、开放检查项；链回 `ws_composite_analysis.md` 与 `print_metric.py`、Fig.4 输出 PNG。
  - [`docs/methodology/wrf_inflow_diagnosis_summary.md`](docs/methodology/wrf_inflow_diagnosis_summary.md)：原 `Global_Constraints` §5 英文叙述全文。
  - [`docs/methodology/mesh_quality_checkmesh_baseline.md`](docs/methodology/mesh_quality_checkmesh_baseline.md)：原 §8 逐数字 `checkMesh` 记录。
- **更新**：[`docs/project/Global_Constraints.md`](docs/project/Global_Constraints.md) — §5 改为短 bullet + 指向 methodology；§6 敏感性结论压缩为一句并链到 Research_Status；§8 改为摘要 + 指向 mesh 基线文档。

## 2026-05-11

### `Global_Constraints.md`：敏感性算例定位为内部讨论
- **输入**：用户说明 buoyancyDestruction 敏感性实验相对主配置无优势，不对外作展示性成果。
- **处理**：更新 [`docs/project/Global_Constraints.md`](docs/project/Global_Constraints.md) §6：区分对外主研究（72 例）与内部敏感性/消融（39 例）；写明不改进主实验、不当作审稿/对外平行 headline；保留路径命名说明（`*-fvOpt_sensitivity_run`）。

### 文献跨尺度分析台账 + 项目 Skill
- **输入**：按计划新增可追加 JSONL 台账与 Cursor Skill，用于多篇 PDF 的「中尺度–微尺度 coupling/offline nesting」结构化摘录与落盘。
- **产出**：
  - [`docs/reference-candidate/literature-trans-scale-ledger.jsonl`](docs/reference-candidate/literature-trans-scale-ledger.jsonl)：每行一条 JSON；首条为 Lin et al. 2021 WRF4PALM/GMD（DOI `10.5194/gmd-14-2503-2021`），字段含 `ledger_schema_version`、`content`（Q1/1A/1B/1C/自评）。
  - [`.cursor/skills/literature-trans-scale-analysis/SKILL.md`](.cursor/skills/literature-trans-scale-analysis/SKILL.md)：判据、JSON schema、`jsonl` 强制追加流程；`disable-model-invocation: true`，需用户 `@` 引用后执行。
  - [`.cursor/skills/literature-trans-scale-analysis/reference.md`](.cursor/skills/literature-trans-scale-analysis/reference.md)：示例记录与 Q1 为 false 的占位结构。
- **调用示例**：`根据 @.cursor/skills/literature-trans-scale-analysis/SKILL.md 分析 @docs/reference-candidate/<论文>.pdf`
- **说明**：`docs/reference-candidate/README.md` 不存在，未新增该说明文件（避免无请求扩 scope）。

## 2026-05-08

### 新增时间序列绘图脚本（单站点×单高度）
- **输入**：用户要求参考 `scripts/plot_hovmoller_lidar_wrf_cfd.py`，新增一个命令行脚本画时间序列，支持 `--obtid --height [--3h-rolling] --tz [utc/lst]`；从 `data/260409/processed/merged_lidar_simulation_final.csv` 里按站点选择“最接近高度”；输出保存到 `results/` 下新目录；文件名需体现 obtid/height/rolling/tz；CSV 默认 UTC，`lst` 时 x 轴展示 UTC+8；配色/线型参考 `results/颜色映射.md`。
- **产出**：新增 `scripts/plot_timeseries_lidar_wrf_cfd.py`
  - 默认输入：`data/260409/processed/merged_lidar_simulation_final.csv`
  - 默认输出目录：`results/timeseries_lidar_wrf_cfd/260409/`
  - 输出文件名编码：`obtid` + `h<height>` + `raw|roll3h` + `tz-utc|tz-lst`
- **可复现运行**：
  - `python scripts/plot_timeseries_lidar_wrf_cfd.py --obtid GAW103 --height 100 --tz utc`
  - `python scripts/plot_timeseries_lidar_wrf_cfd.py --obtid GAW103 --height 100 200 300 --3h-rolling --tz lst`
- **小修改**：LiDAR 观测线型改为黑色 `o-`（原为仅散点）。

## 2026-05-06

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
