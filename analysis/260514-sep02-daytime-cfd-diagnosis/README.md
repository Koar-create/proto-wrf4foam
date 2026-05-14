# 2025-09-02 白昼 CFD 偏差诊断（可复现脚本包）

本目录落实 [`docs/ops/diagnose-anomaly-at-day02-daytime.md`](../../docs/ops/diagnose-anomaly-at-day02-daytime.md) 的诊断链，并与 [`docs/project/Global_Constraints.md`](../../docs/project/Global_Constraints.md)、[`analysis/260409/ws_composite_analysis.md`](../260409/ws_composite_analysis.md) 中的结论对照。

## 数据路径（仓库约定）

| 用途 | 默认路径 |
|------|-----------|
| LiDAR–WRF–CFD 合并表 | `data/260409/processed/merged_lidar_simulation_final.csv` |
| OpenFOAM 算例根目录（本机） | 需自备，例如 `steady_experiments_finer_ABL/<case_id>/` |

若 `data/` 未同步，脚本会报错退出并打印期望路径。

## 诊断顺序（与 ops 文档 Step 对应）

```text
Step 0  LiDAR 基线（必做，仅用 merged CSV）
        → scripts/metrics_sep02_daytime_vs_neighbors.py
        → scripts/profiles_sep02_daytime_composite.py
Step 1  boundaryData 廓线 + LLJ 指标
        → scripts/boundarydata_inflow_profiles_compare.py
Step 2  boundaryData 数值 QC
        → scripts/boundarydata_qc_scan.py
Step 3  simpleFoam 残差
        → scripts/parse_simplefoam_residuals.py
Step 4  场内切片 / ParaView（本 README 清单即可，不强制脚本）
```

## 运行示例

在项目根目录执行（Windows / Linux 均可）：

```bash
python analysis/260514-sep02-daytime-cfd-diagnosis/scripts/metrics_sep02_daytime_vs_neighbors.py --csv data/260409/processed/merged_lidar_simulation_final.csv --out-dir analysis/260514-sep02-daytime-cfd-diagnosis/results

python analysis/260514-sep02-daytime-cfd-diagnosis/scripts/profiles_sep02_daytime_composite.py --csv data/260409/processed/merged_lidar_simulation_final.csv --out-dir analysis/260514-sep02-daytime-cfd-diagnosis/results
```

`profiles_sep02_daytime_composite.py`：四站垂向 `Height` 不完全一致时，先将各 `(datetime, obtid)` 廓线插值到统一网格（默认 `--z-min 55`、`--z-max-interp 2000`、`--dz 5` m）再对站点 `nanmean`，避免绘图连线锯齿。需要更细垂向分辨率可调小 `--dz`。风速横轴默认 **xlim −0.2–4.2 m/s**、主刻度 **0–4**（可用 `--x-min` / `--x-max` / `--x-ticks` 覆盖）。**`--utc-hours` 为数据选取的 UTC 整点；图中子标题显示对应 LST（UTC+8）。**

```bash
python analysis/260514-sep02-daytime-cfd-diagnosis/scripts/boundarydata_inflow_profiles_compare.py --manifest analysis/260514-sep02-daytime-cfd-diagnosis/example_cases_manifest.csv --out-dir analysis/260514-sep02-daytime-cfd-diagnosis/results

python analysis/260514-sep02-daytime-cfd-diagnosis/scripts/boundarydata_inflow_profiles_compare.py --case-root steady_experiments_finer_ABL --case-glob "2025090*_03*_two_boundaries_as_outlet" --out-dir analysis/260514-sep02-daytime-cfd-diagnosis/results
```

`boundaryData` 实际布局（`steady_experiments_finer_ABL/.../constant/boundaryData`）：patch 目录名为**小写** `west` / `east` / `south` / `north`；每个 patch 含根目录 `points` 与时次子目录（多为 `0/`）下的 `U,k,epsilon`（`vectorAverageField` 等）。Step1 默认处理 **`east,south`**；可用 `--discover-patches` 自动发现含完整场的 patch 并仍优先 `east,south`。若 patch 下同时存在与算例名一致的 `YYYYMMDD_HHMM` 与 `0/`，脚本**优先匹配算例时间戳子目录**（`timeVaryingMappedFixedValue`）；否则用 `0/`。仍可用 `--time-subdir` 强制指定。

```bash
python analysis/260514-sep02-daytime-cfd-diagnosis/scripts/boundarydata_qc_scan.py --manifest analysis/260514-sep02-daytime-cfd-diagnosis/example_cases_manifest.csv --out-dir analysis/260514-sep02-daytime-cfd-diagnosis/results

python analysis/260514-sep02-daytime-cfd-diagnosis/scripts/parse_simplefoam_residuals.py --manifest analysis/260514-sep02-daytime-cfd-diagnosis/example_cases_manifest.csv --out-dir analysis/260514-sep02-daytime-cfd-diagnosis/results
```

`example_cases_manifest.csv`：`case_path` 为算例根目录；`patches` 可选，**须与磁盘目录一致的小写名**（默认 `east,south`）。若第二列误填成另一条算例路径，脚本会识别并回退为默认 `east,south`。

列说明：

- `case_path`：算例根目录（必填）
- `patches`：可选，逗号分隔（默认 `east,south`）

示例：

```text
E:/WRF-OpenFOAM-Coupling/steady_experiments_finer_ABL/20250901_0300_two_boundaries_as_outlet,"east,south"
E:/WRF-OpenFOAM-Coupling/steady_experiments_finer_ABL/20250902_0300_two_boundaries_as_outlet,"east,south"
E:/WRF-OpenFOAM-Coupling/steady_experiments_finer_ABL/20250903_0300_two_boundaries_as_outlet,"east,south"
```

## Step 4：场内结构检查清单（ParaView / postProcess）

1. **Z ≈ 50 m** 水平切片：三日同一 UTC 对比近地层风速型。
2. **Y ≈ 0**（珠江开敞带法向）垂直断面：检查侧向入流与河道条带相互作用。
3. **建筑密集区局部放大**：异常低速区 / 大分离区是否仅在某日出现。

不依赖本仓库内 VTK；算例后处理目录按你本地 OpenFOAM 习惯即可。

## 结论决策表（避免循环论证）

| 条件 | 推断 |
|------|------|
| Step 1：09-02 白昼 East/South 相对 01/03 出现明显 **风速鼻状** 且 **k 显著偏低** | 主因倾向 **WRF 入流物理状态**（LLJ 残余 / 强稳定层结），CFD 映射入流难以自行纠正。 |
| Step 1 廓线相近，但 Step 2 出现 **k&lt;0**、**epsilon 过小** 或 **同高度面内 std/mean 过大** | 主因倾向 **插值或写入 boundaryData 的数值问题**。 |
| Step 1–2 正常，Step 3 **末段残差偏大或 k/epsilon 发散趋势** | 主因倾向 **稳态 simpleFoam + k-ε 在该入流下的数值困难**。 |
| Step 1–3 均正常 | 进入 Step 4 **局地几何 + 分离流** 等场内因素（兜底）。 |

## 方法论文档（可选深入）

- [`docs/methodology/abl_stability_and_llj_detection.md`](../../docs/methodology/abl_stability_and_llj_detection.md)
- [`docs/methodology/wrf_inflow_diagnosis_summary.md`](../../docs/methodology/wrf_inflow_diagnosis_summary.md)
