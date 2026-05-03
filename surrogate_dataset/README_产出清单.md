## surrogate_dataset 产出清单

本目录为“CFD Surrogate训练标准化数据集”的统一输出位置。

### 已产出内容

- **主索引**
  - `index.csv`
  - 含字段：`case_id, timestamp_utc, is_sensitivity, ABL_class, wind_dir_bin, U_ref, k_max, k_500m, z_kmax, h5_path, sdf_path, split`
  - `split` 采用按 `ABL_class` 分层抽样，整体约 80/10/10，并保证每个 `ABL_class` 在 `val` 与 `test` 中至少各 1 个样本（当前数据满足）。
  - 备注：`wind_dir` 为气象学风向（来向），`wind_dir_bin` 由其映射到 8 方位（N/NE/E/SE/S/SW/W/NW）。

- **CFD场（任务二集中汇总）**
  - `fields/<case_id>.h5`（共111个）
  - 每个HDF5内部结构（已存在的stage2结果被集中拷贝）：
    - `/U` `float32 (3, 131, 131, 15)`
    - `/k` `float32 (1, 131, 131, 15)`
    - `/coords_x` `float32 (131,)`
    - `/coords_y` `float32 (131,)`
    - `/coords_z` `float32 (15,)`
    - `attrs/meta`：JSON
  - 任务二汇总报告：`fields/_task2_report.json`

- **入流条件向量（任务三）**
  - `inflow/<case_id>_inflow.json`（共111个）
  - 含字段：`case_id, timestamp_utc, timestamp_local, is_sensitivity, U_ref, wind_dir, k_max, k_500m, z_kmax, ABL_class, ABL_class_name`，以及 `east/south` 的U、k廓线列表
  - 任务三运行报告：`inflow/_task3_report.json`
  - 风向定义：先算去向 $\theta_{\mathrm{to}}=\mathrm{atan2}(u,v)$（北=0顺时针），再转来向 $\theta_{\mathrm{from}}=(\theta_{\mathrm{to}}+180)\bmod 360$。
  - 时间约定：**case_id 中的时间戳按 UTC 记号**；`timestamp_utc` 直接由 case_id 解析，`timestamp_local` 为其换算到 **UTC+8**。
  - ABL 分类逻辑（优先级判别；按 `local_hour` 使用本地时区 UTC+8）：

    $$
    \mathrm{ABL\_class}=
    \begin{cases}
    3~(\mathrm{LLJ}), & k_{\max}>0.5 \ \wedge\ k_{500\mathrm{m}}<0.1 \ \wedge\ z_{k\max}<400 \\
    0~(\mathrm{Unstable}), & 11\le h_{\mathrm{local}}\le 16 \ \wedge\ k_{\max}>1.0 \\
    2~(\mathrm{Stable}), & k_{\max}<0.1 \\
    1~(\mathrm{Neutral}), & \text{otherwise}
    \end{cases}
    $$

    其中 $h_{\mathrm{local}}=\text{local\_hour}$，$k_{\max}=\max_z k(z)$，$k_{500\mathrm{m}}$ 为 500m 高度处的 $k$（两边界平均），$z_{k\max}=\arg\max_z k(z)$。

- **几何编码（任务一）**
  - `geometry/building_encoding_131x131x15.npy`
    - `shape = (2, 131, 131, 15)`, `float32`
    - 通道0：建筑UDF（到最近三角面距离，截断200m）
    - 通道1：建筑占位mask（基于屋顶射线求 `z_roof`，若 \(z < z_{roof}\) 则为1）
  - 质量检查图：`geometry/building_encoding_qc.png`

- **样本空间地图（指令2统计分析）**
  - `sample_space_map.png`
  - 2×2 子图（300 dpi）：ABL×风向bin热力/散点、`U_ref×k_max`（log）+凸包、时间轴稳定度序列、覆盖缺口诊断表格。

### 可复现生成命令（仓库根目录运行）

- **任务一（几何编码）**：`python util/task1_building_encoding.py`
- **任务二（fields集中汇总）**：`python util/task2_collect_fields.py`
- **任务三（入流提取）**：`python -u util/task3_extract_inflow.py`
- **任务四（主索引生成）**：`python -u util/task4_make_index.py`
- **指令2（样本空间地图）**：`python -u util/task5_sample_space_map.py`

