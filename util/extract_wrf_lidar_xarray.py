"""
extract_wrf_lidar_xarray_flat.py
=================================
平坦化版本（xarray 版）：从 WRF 中提取 LiDAR 站点处的模拟风场数据。
"""

import os
import re
import sys
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import glob
from scipy.interpolate import interp1d

# 强制输出为 UTF-8 编码，防止 Windows 控制台编码错误
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ←  根据实际路径修改这里
# ─────────────────────────────────────────────────────────────────────────────

def repo_root() -> Path:
    """仓库根目录：由脚本位置推断，不依赖 $HOME。"""
    return Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    """优先使用环境变量 WRF_OPENFOAM_COUPLING_ROOT，否则回退到 repo_root。"""
    env_root = os.environ.get("WRF_OPENFOAM_COUPLING_ROOT")
    if env_root:
        return Path(env_root)
    return repo_root()


def resolve_existing_nc_path(path: str | os.PathLike[str]) -> str | None:
    """返回存在的 WRF NetCDF 路径；同时接受 ':' 与 Windows 上的 '%3A' 时间戳文件名。"""
    p = Path(path)
    if p.exists():
        return str(p)

    name = p.name
    candidates = []
    if ":" in name:
        candidates.append(p.with_name(name.replace(":", "%3A")))
        candidates.append(p.with_name(name.replace(":", "%3a")))
    if "%3A" in name or "%3a" in name:
        candidates.append(p.with_name(name.replace("%3A", ":").replace("%3a", ":")))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


PROJECT_ROOT = get_project_root()
OUT_DIR = PROJECT_ROOT / "data" / "260409" / "raw" / "wrf"
WRF_DATA_DIR = PROJECT_ROOT / "W_myExp03" / "auxhist2"

# 读取 LiDAR 站点高度信息
JSON_PATH = PROJECT_ROOT / "util" / "lidar_station_info.json"
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

# =============================================================================
# 开始执行：创建输出目录
# =============================================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print("  WRF LiDAR 提取（自动扫描日期，循环处理指定时刻）")
print("="*60)

# =============================================================================
# WRF 提取部分（xarray）
# =============================================================================

target_hours = np.arange(24)
wrf_records = []

# 自动扫描可用日期（通配符兼容 ':' 与 '%3A' 时间戳）
file_pattern = str(WRF_DATA_DIR / "auxhist2_d03_*-*-*_*_1h-rolling_cartesian.nc")
all_matching_files = glob.glob(file_pattern)

found_dates = set()
for f in all_matching_files:
    # 提取日期部分: YYYY-MM-DD
    m = re.search(r"auxhist2_d03_(\d{4}-\d{2}-\d{2})_", os.path.basename(f))
    if m:
        found_dates.add(m.group(1))

target_dates = sorted(list(found_dates))
print(f"  识别到日期范围: {target_dates}")

for d_str in target_dates:
    for h in target_hours:
        # 构造文件名（优先 ':'，若不存在则尝试 '%3A' 变体）
        f_h = f"{h:02d}:00:00"
        base_name = f"auxhist2_d03_{d_str}_{f_h}_1h-rolling_cartesian.nc"
        fpath = resolve_existing_nc_path(WRF_DATA_DIR / base_name)

        if fpath is None:
            # 如果某个时次不存在，静默跳过
            continue
            
        fname = os.path.basename(fpath)
        
        # 从文件名中解析日期时间
        m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}).*?(\d{2}).*?(\d{2})", fname)
        if m:
            datetime_str = f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}"
        else:
            datetime_str = f"{d_str} {h:02d}:00:00"
        
        print(f"  WRF [lidar] {datetime_str} ...")
        
        # ── 使用 xarray 打开 NetCDF 文件 ──────────────────────────────────────────────
        ds = xr.open_dataset(fpath, mask_and_scale=False)
        
        # ── 提取相对坐标轴（cartesian 产品无 XLONG/XLAT）──────────────────────────
        x_coords = ds["x_rel"].values.squeeze().astype(float)
        y_coords = ds["y_rel"].values.squeeze().astype(float)

        if x_coords.ndim != 1 or y_coords.ndim != 1:
            raise ValueError(
                f"x_rel/y_rel 必须是 1-D，当前 x_rel={x_coords.shape}, y_rel={y_coords.shape}"
            )
        
        # ── 提取垂直坐标 z ────────────────────────────────────────────────────────────
        z_arr = ds["z"].values.squeeze().astype(float)      # shape (nz,)
        
        # ── 提取风场变量 ──────────────────────────────────────────────────────────────
        varnames = ["U", "V", "W", "WS", "TKE_PBL"]
        wrf_vars = {}
        for vn in varnames:
            arr = ds[vn].values.squeeze().astype(float)
            if arr.ndim != 3:
                raise ValueError(f"{vn} 必须是 3-D（squeeze 后），当前 shape={arr.shape}")
            wrf_vars[vn] = arr
        
        ds.close()
        
        # ── 逐站点插值 ────────────────────────────────────────────────────────────────
        for _, row in lidar_sites.iterrows():
            obtid = row["obtid"]
            site_x = float(row["x_rel"])
            site_y = float(row["y_rel"])

            # 从 JSON 中获取该站点的探测高度层，并过滤掉 >= 2000 的部分
            raw_levels = np.array(station_info_dict[obtid]["levels"], dtype=float)
            target_z = raw_levels[raw_levels < 2000]

            # 最近邻：在相对坐标轴上找距离最近的格点
            i = int(np.argmin(np.abs(x_coords - site_x)))
            j = int(np.argmin(np.abs(y_coords - site_y)))
        
            # 提取该格点的垂直廓线
            col_data = {}
            for vn in varnames:
                col_data[vn] = wrf_vars[vn][:, j, i]
        
            # 垂直坐标
            h_col = z_arr.copy()
            
            if h_col.ndim != 1:
                raise ValueError(f"h_col 必须为 1-D，当前 shape={h_col.shape}")
    
            sort_idx = np.argsort(h_col)
            h_sorted = h_col[sort_idx]
        
            res = {}
            for vn in ["U", "V", "W", "WS", "TKE_PBL"]:
                v_col = np.asarray(col_data[vn], dtype=float)
                v_sorted = v_col[sort_idx]
                f = interp1d(
                    h_sorted, v_sorted,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(v_sorted[0], v_sorted[-1])
                )
                res[vn] = f(target_z)
            
            # 计算 k 和 epsilon
            k_wrf = np.maximum(res["TKE_PBL"], 1e-6)
            Cmu = 0.09
            kappa = 0.41
            L_CFD_min = 10.0
            mixing_length = np.clip(kappa * target_z, L_CFD_min, 100.0)
            eps_wrf = (Cmu**0.75) * (k_wrf**1.5) / mixing_length
            eps_wrf = np.maximum(eps_wrf, 1e-8)
            
            res["k"] = k_wrf
            res["epsilon"] = eps_wrf
        
            # 逐高度层构建记录
            for i_idx, z in enumerate(target_z):
                rec = {
                    "datetime":  pd.Timestamp(datetime_str),
                    "obtid":     row["obtid"],
                    "lon":       row["lon"],
                    "lat":       row["lat"],
                    "x_rel":     row["x_rel"],
                    "y_rel":     row["y_rel"],
                    "z_probe":   float(z),
                    "U_wrf":     res["U"][i_idx],
                    "V_wrf":     res["V"][i_idx],
                    "W_wrf":     res["W"][i_idx],
                    "WS_wrf":    res["WS"][i_idx],
                    "k_wrf":     res["k"][i_idx],
                    "eps_wrf":   res["epsilon"][i_idx],
                }
                wrf_records.append(rec)
        
        print(f"  OK {datetime_str}")

# =============================================================================
# 整理并保存 WRF 数据
# =============================================================================
df_wrf_lidar = pd.DataFrame(wrf_records)
if not df_wrf_lidar.empty:
    df_wrf_lidar["datetime"] = pd.to_datetime(df_wrf_lidar["datetime"])
    df_wrf_lidar = df_wrf_lidar.sort_values(
        ["datetime", "obtid", "z_probe"]
    ).reset_index(drop=True)

out_wrf = OUT_DIR / "WRF_lidar_simulation_1h-rolling.csv"
df_wrf_lidar.to_csv(out_wrf, index=False)
print(f"\n  → 已保存: {out_wrf}  shape={df_wrf_lidar.shape}")

print("\n✅  WRF 提取完成！")
