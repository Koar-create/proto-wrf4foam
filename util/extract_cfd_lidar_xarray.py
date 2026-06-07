"""
extract_cfd_lidar_xarray_flat.py
=================================
平坦化版本（xarray 风格统一）：从 OpenFOAM (simpleFoam 稳态实验) 中提取
LiDAR 站点处的模拟风场数据，输出 CSV 文件。

用法:
    python extract_cfd_lidar_xarray_flat.py <case_path>
示例:
    python util/extract_cfd_lidar_xarray_flat.py steady_experiments_finer_ABL/20250903_1200
"""

import os
import re
import glob
import sys
import io
import json
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d

# 强制输出为 UTF-8 编码，防止 Windows 控制台编码错误
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ←  根据实际路径修改这里
# ─────────────────────────────────────────────────────────────────────────────
# 处理 Windows 和 Linux 路径兼容性
CFD_DIR = "steady_experiments_finer_ABL"
# OUT_DIR 在解析 case_path 后根据路径是否含 sensitivity 设定（见下方）

# 读取 LiDAR 站点高度信息
JSON_PATH = os.path.join("util", "lidar_station_info.json")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    station_info_dict = json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# 站点信息
# ─────────────────────────────────────────────────────────────────────────────
lidar_sites = pd.DataFrame({
    "obtid":         ["GAW103",     "GAW104",     "GAW105",     "GAW111"],
    "lon":           [113.331446,   113.326053,   113.316797,   113.322620],
    "lat":           [ 23.110176,    23.116321,    23.116133,    23.113718],
    "x_rel":         [ 975,          450,          -506,          75],
    "y_rel":         [-320,          350,           400,          30],
    "altitude_m":    [  7.2,          11.6,           6,          30.9],
    "altitude_m_cfd":[  1.6,           6,            0.4,         25.3],
})

print("\n" + "="*60)
print("  CFD LiDAR 提取（全探针高度剖面）")
print("="*60)

# =============================================================================
# 处理命令行参数
# =============================================================================
def print_usage():
    print("\n" + "!"*60)
    print("[参数缺失或输入错误]")
    print("用法: python util/extract_cfd_lidar_xarray_flat.py <case_path>")
    print("示例: python util/extract_cfd_lidar_xarray_flat.py steady_experiments_finer_ABL/20250903_1200")
    print("!"*60 + "\n")

if len(sys.argv) < 2:
    print_usage()
    sys.exit(1)

# 用户输入路径，例如: steady_experiments_finer_ABL/20250903_1200
input_path = sys.argv[1].strip()
if os.path.isabs(input_path):
    full = input_path
else:
    full = os.path.abspath(input_path)

target_entry = os.path.basename(full)

if not os.path.isdir(full):
    print(f"\n错误: 目标目录不存在 -> {full}")
    print_usage()
    sys.exit(1)

if "sensitivity" in full.lower():
    OUT_DIR = os.path.join("data", "260409", "raw", "cfd", "sensitivity")
else:
    OUT_DIR = os.path.join("data", "260409", "raw", "cfd", "control")
os.makedirs(OUT_DIR, exist_ok=True)

exps = []

# 尝试从目录名解析日期时间，例如 20250903_1200
# 兼容带后缀的情况，如 20250903_1200_two_boundaries_as_outlet
m = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})", target_entry)
if m:
    yr, mo, dy, hh, mm = m.groups()
    dt_str = f"{yr}-{mo}-{dy} {hh}:{mm}:00"
else:
    # 如果解析失败，使用固定默认值或从当前文件系统获取（这里保持脚本逻辑，给出警告）
    dt_str = "2025-01-01 00:00:00"
    print(f"警告: 无法从目录名 {target_entry} 解析日期，使用默认值: {dt_str}")

# 找最终时间步
timesteps = []
for e in os.listdir(full):
    if os.path.isdir(os.path.join(full, e)):
        try:
            timesteps.append(int(e))
        except ValueError:
            pass
last_ts = str(max(timesteps)) if timesteps else None

if last_ts is not None:
    # 判断是否提前收敛：若最终步不是 5000
    status = "normal" if last_ts == "5000" else "early_converge"
    exps.append((dt_str, full, last_ts, status))
else:
    print(f"未能找到时间步，请检查目录: {full}")
    sys.exit(1)

print(f"\n  发现 {len(exps)} 个 CFD 实验目录")

# =============================================================================
# 逐实验提取 CFD 数据（原 extract_cfd_all 函数展开）
# =============================================================================
cfd_records = []

for dt_str, case_dir, last_ts, status in exps:
    exp_name = os.path.basename(case_dir)

    if status == "crashed":
        print(f"  CFD [lidar] {dt_str} SKIPPED (crashed): {exp_name}")
        continue

    print(f"  CFD [lidar] {dt_str} ts={last_ts} ({status}): {exp_name} ...", end=" ")

    # ── 读取 cell centres（原 read_foam_cell_centres 函数展开）────────────────
    cell_coords = None

    for candidate in [
        os.path.join(case_dir, "constant", "cellCentres"),
        os.path.join(case_dir, "0", "C"),
    ]:
        if not os.path.exists(candidate):
            continue

        # 原 parse_foam_vector_field 函数逻辑
        with open(candidate, "r", errors="replace") as f:
            content = f.read()

        m_vec = re.search(
            r"internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*;",
            content, re.DOTALL
        )
        if not m_vec:
            m_uni = re.search(
                r"internalField\s+uniform\s+\(([^)]+)\)", content
            )
            if m_uni:
                vals = list(map(float, m_uni.group(1).split()))
                cell_coords = np.array(vals)
                break
            raise ValueError(f"无法解析 internalField: {candidate}")

        raw = m_vec.group(2).strip()
        pattern_vec = re.compile(r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)")
        vecs = pattern_vec.findall(raw)
        cell_coords = np.array([[float(a), float(b), float(c)] for a, b, c in vecs])
        break
    # ── parse_foam_vector_field 逻辑结束 ──────────────────────────────────────

    if cell_coords is None:
        # 方法2: 读取 VTK（如果有）
        vtk_dir = os.path.join(case_dir, "VTK")
        if os.path.isdir(vtk_dir):
            vtk_files = sorted(glob.glob(os.path.join(vtk_dir, "**", "*.vtk"), recursive=True))
            if vtk_files:
                import vtk
                from vtk.util.numpy_support import vtk_to_numpy
                reader = vtk.vtkUnstructuredGridReader()
                reader.SetFileName(vtk_files[-1])
                reader.Update()
                grid = reader.GetOutput()
                centres_filter = vtk.vtkCellCenters()
                centres_filter.SetInputData(grid)
                centres_filter.Update()
                pts = centres_filter.GetOutput().GetPoints()
                cell_coords = vtk_to_numpy(pts.GetData())

    if cell_coords is None:
        raise FileNotFoundError(
            f"无法找到 cell centres，请在 case 目录下运行:\n"
            f"  postProcess -func writeCellCentres -time 0\n"
            f"  (case: {case_dir})"
        )
    # ── read_foam_cell_centres 逻辑结束 ───────────────────────────────────────

    # ── 读取速度场 U（原 read_foam_U_field 函数展开）─────────────────────────
    ufile = os.path.join(case_dir, last_ts, "U")
    with open(ufile, "r", errors="replace") as f:
        content = f.read()

    m_vec = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*;",
        content, re.DOTALL
    )
    if not m_vec:
        m_uni = re.search(
            r"internalField\s+uniform\s+\(([^)]+)\)", content
        )
        if m_uni:
            vals = list(map(float, m_uni.group(1).split()))
            U_arr = np.array(vals)
        else:
            raise ValueError(f"无法解析 internalField: {ufile}")
    else:
        raw = m_vec.group(2).strip()
        pattern_vec = re.compile(r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)")
        vecs = pattern_vec.findall(raw)
        U_arr = np.array([[float(a), float(b), float(c)] for a, b, c in vecs])
    # ── read_foam_U_field 逻辑结束 ────────────────────────────────────────────

    if U_arr.ndim == 1:
        # uniform 场（极少情况）
        U_arr = np.tile(U_arr, (len(cell_coords), 1))

    # ── 读取标量场 k 和 epsilon ─────────────────────────
    def read_scalar_field(field_name):
        ffile = os.path.join(case_dir, last_ts, field_name)
        if not os.path.exists(ffile):
            return np.zeros(len(cell_coords))
        with open(ffile, "r", errors="replace") as f:
            content = f.read()
        m_scalar = re.search(
            r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*;",
            content, re.DOTALL
        )
        if not m_scalar:
            m_uni = re.search(
                r"internalField\s+uniform\s+([-\d.eE+]+)", content
            )
            if m_uni:
                val = float(m_uni.group(1))
                return np.full(len(cell_coords), val)
            else:
                return np.zeros(len(cell_coords))
        else:
            raw = m_scalar.group(2).strip()
            vals = raw.split()
            return np.array([float(v) for v in vals])
            
    k_arr = read_scalar_field("k")
    eps_arr = read_scalar_field("epsilon")

    # ── 逐站点提取（原 cfd_extract_site_from_field 函数展开）─────────────────
    for _, row in lidar_sites.iterrows():
        obtid       = row["obtid"]
        site_x      = float(row["x_rel"])
        site_y      = float(row["y_rel"])
        
        # 从 JSON 中获取该站点的探测高度层，并过滤掉 >= 2000 的部分
        raw_levels = np.array(station_info_dict[obtid]["levels"], dtype=float)
        probe_heights = raw_levels[raw_levels < 2000]

        cx = cell_coords[:, 0]
        cy = cell_coords[:, 1]
        cz = cell_coords[:, 2]

        # 水平距离
        dist_h = np.sqrt((cx - site_x)**2 + (cy - site_y)**2)

        # 自适应搜索半径（从 50m 逐步扩大，直到有足够多的 cell）
        N_min = 30
        R = 50.0
        while R <= 2000.0:
            mask = dist_h <= R
            if mask.sum() >= N_min:
                break
            R *= 1.5
        mask = dist_h <= R

        if mask.sum() < 3:
            # 实在找不到，用全局最近的 N_min 个
            idx_sorted = np.argsort(dist_h)[:N_min]
            mask = np.zeros(len(cx), dtype=bool)
            mask[idx_sorted] = True

        z_sel = cz[mask]
        U_sel = U_arr[mask, 0]
        V_sel = U_arr[mask, 1]
        W_sel = U_arr[mask, 2]
        k_sel = k_arr[mask]
        eps_sel = eps_arr[mask]

        # 反距离加权（IDW）
        dist_sel = dist_h[mask]
        w = 1.0 / (dist_sel + 1e-6)
        w /= w.sum()

        # 建立 z 分 bin（每 20m 一个 bin）
        z_min, z_max = cz.min(), cz.max()
        bin_edges    = np.arange(z_min - 10, z_max + 30, 20)
        z_bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # ── weighted_bin 内联（原内联函数，保持逻辑不变）─────────────────────
        # 对变量进行分 bin 加权平均
        num_u = np.zeros(len(z_bin_centres))
        den_u = np.zeros(len(z_bin_centres))
        num_v = np.zeros(len(z_bin_centres))
        den_v = np.zeros(len(z_bin_centres))
        num_w = np.zeros(len(z_bin_centres))
        den_w = np.zeros(len(z_bin_centres))
        num_k = np.zeros(len(z_bin_centres))
        den_k = np.zeros(len(z_bin_centres))
        num_eps = np.zeros(len(z_bin_centres))
        den_eps = np.zeros(len(z_bin_centres))

        for i_bin, zb in enumerate(z_bin_centres):
            in_bin = (z_sel >= bin_edges[i_bin]) & (z_sel < bin_edges[i_bin + 1])
            if in_bin.sum() == 0:
                continue
            w_bin       = w[in_bin]
            num_u[i_bin] = (w_bin * U_sel[in_bin]).sum()
            den_u[i_bin] = w_bin.sum()
            num_v[i_bin] = (w_bin * V_sel[in_bin]).sum()
            den_v[i_bin] = w_bin.sum()
            num_w[i_bin] = (w_bin * W_sel[in_bin]).sum()
            den_w[i_bin] = w_bin.sum()
            num_k[i_bin] = (w_bin * k_sel[in_bin]).sum()
            den_k[i_bin] = w_bin.sum()
            num_eps[i_bin] = (w_bin * eps_sel[in_bin]).sum()
            den_eps[i_bin] = w_bin.sum()

        valid_u = den_u > 0
        valid_v = den_v > 0
        valid_w = den_w > 0
        valid_k = den_k > 0
        valid_eps = den_eps > 0

        # 取分量共同有效的 bin（保证 z_prof 一致）
        valid = valid_u & valid_v & valid_w & valid_k & valid_eps
        z_prof = z_bin_centres[valid]
        U_prof = num_u[valid] / den_u[valid]
        V_prof = num_v[valid] / den_v[valid]
        W_prof = num_w[valid] / den_w[valid]
        k_prof = num_k[valid] / den_k[valid]
        eps_prof = num_eps[valid] / den_eps[valid]
        # ── weighted_bin 内联结束 ─────────────────────────────────────────────

        if len(z_prof) < 2:
            # 退化情况：用所有点直接排序
            order  = np.argsort(z_sel)
            z_prof = z_sel[order]
            U_prof = U_sel[order]
            V_prof = V_sel[order]
            W_prof = W_sel[order]
            k_prof = k_sel[order]
            eps_prof = eps_sel[order]

        # 去重（同一 z 值）
        _, ui   = np.unique(z_prof, return_index=True)
        z_prof  = z_prof[ui]
        U_prof  = U_prof[ui]
        V_prof  = V_prof[ui]
        W_prof  = W_prof[ui]
        k_prof  = k_prof[ui]
        eps_prof = eps_prof[ui]

        # ── safe_interp 内联（原内联函数，保持逻辑不变）──────────────────────
        f_u = interp1d(z_prof, U_prof, kind="linear", bounds_error=False,
                       fill_value=(U_prof[0], U_prof[-1]))
        f_v = interp1d(z_prof, V_prof, kind="linear", bounds_error=False,
                       fill_value=(V_prof[0], V_prof[-1]))
        f_w = interp1d(z_prof, W_prof, kind="linear", bounds_error=False,
                       fill_value=(W_prof[0], W_prof[-1]))
        f_k = interp1d(z_prof, k_prof, kind="linear", bounds_error=False,
                       fill_value=(k_prof[0], k_prof[-1]))
        f_eps = interp1d(z_prof, eps_prof, kind="linear", bounds_error=False,
                         fill_value=(eps_prof[0], eps_prof[-1]))

        u_out  = f_u(probe_heights)
        v_out  = f_v(probe_heights)
        w_out  = f_w(probe_heights)
        ws_out = np.sqrt(u_out**2 + v_out**2)
        k_out  = f_k(probe_heights)
        eps_out= f_eps(probe_heights)
        # ── safe_interp 内联结束 ──────────────────────────────────────────────

        # 逐高度层构建记录
        for i_idx, z in enumerate(probe_heights):
            rec = {
                "datetime":      dt_str,
                "exp_name":      exp_name,
                "cfd_status":    status,
                "obtid":         row["obtid"],
                "lon":           row["lon"],
                "lat":           row["lat"],
                "x_rel":         row["x_rel"],
                "y_rel":         row["y_rel"],
                "altitude_m_cfd":row["altitude_m_cfd"],
                "z_probe":       z,
                "U_cfd":         u_out[i_idx],
                "V_cfd":         v_out[i_idx],
                "W_cfd":         w_out[i_idx],
                "WS_cfd":        ws_out[i_idx],
                "k_cfd":         k_out[i_idx],
                "eps_cfd":       eps_out[i_idx],
            }
            cfd_records.append(rec)
    # ── cfd_extract_site_from_field 逻辑结束 ──────────────────────────────────

    print("OK")

# =============================================================================
# 整理并保存 CFD 数据
# 使用 xarray.Dataset.from_dataframe 做一次中间转换，验证数据完整性后再导出 CSV
# =============================================================================
df_cfd_lidar = pd.DataFrame(cfd_records)

if not df_cfd_lidar.empty:
    df_cfd_lidar["datetime"] = pd.to_datetime(df_cfd_lidar["datetime"])
    df_cfd_lidar = df_cfd_lidar.sort_values(
        ["datetime", "obtid", "z_probe"]
    ).reset_index(drop=True)

    # ── xarray 整合验证（xarray 风格统一）────────────────────────────────────
    # 将数值列转为 xarray.Dataset，方便后续扩展（如添加坐标属性、NetCDF 输出等）
    numeric_cols = ["z_probe", "U_cfd", "V_cfd", "W_cfd", "WS_cfd", "k_cfd", "eps_cfd"]
    ds_cfd = xr.Dataset.from_dataframe(df_cfd_lidar[numeric_cols])
    # 仅做验证打印，最终仍以 CSV 格式保存
    print(f"\n  xarray Dataset 概览: {dict(ds_cfd.dims)}")
    ds_cfd.close()
    # ── xarray 整合验证结束 ───────────────────────────────────────────────────

out_cfd = os.path.join(OUT_DIR, f"CFD_lidar_simulation_{target_entry}.csv")
df_cfd_lidar.to_csv(out_cfd, index=False)
print(f"\n  → 已保存: {out_cfd}  shape={df_cfd_lidar.shape}")

print("\n✅  CFD 提取完成！")
