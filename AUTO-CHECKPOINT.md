# AUTO-CHECKPOINT

> 本文件由Cursor智能体自动维护，用于记录本仓库内任务执行过程、关键决策与可复现命令。

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
