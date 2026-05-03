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
# 处理 Windows 和 Linux 路径兼容性
PROJECT_ROOT = os.path.join(os.environ.get('HOME'), 'WRF-OpenFOAM-Coupling')
OUT_DIR = os.path.join(PROJECT_ROOT, "260409")   # 输出目录
WRF_DATA_DIR = os.path.join(PROJECT_ROOT, "W_myExp03/auxhist2/horiz_raw_z_interp")

# 读取 LiDAR 站点高度信息
JSON_PATH = os.path.join(PROJECT_ROOT, "util", "lidar_station_info.json")
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
os.makedirs(OUT_DIR, exist_ok=True)

print("\n" + "="*60)
print("  WRF LiDAR 提取（自动扫描日期，循环处理指定时刻）")
print("="*60)

# =============================================================================
# WRF 提取部分（xarray）
# =============================================================================

target_hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
wrf_records = []

# 自动扫描可用日期
file_pattern = os.path.join(WRF_DATA_DIR, "auxhist2_d03_*-*-*_*:00:00_1h-rolling.nc")
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
        # 构造文件名，确保小时是两位数，并且冒号编码为 :
        f_h = f"{h:02d}:00:00"
        fpath = os.path.join(WRF_DATA_DIR, f"auxhist2_d03_{d_str}_{f_h}_1h-rolling.nc")
        
        if not os.path.exists(fpath):
            # 如果某个时次不存在，静默跳过或简单提示
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
        
        # ── 提取经纬度网格 ──────────────────────────────────────────────────────────
        lons = ds["XLONG"].values.squeeze().astype(float)
        lats = ds["XLAT"].values.squeeze().astype(float)
        
        if lons.ndim != 2 or lats.ndim != 2:
            raise ValueError(f"XLONG/XLAT 必须是 2-D，当前 XLONG={lons.shape}, XLAT={lats.shape}")
        
        # ── 提取垂直坐标 z ────────────────────────────────────────────────────────────
        z_arr = ds["z"].values.squeeze().astype(float)      # shape (nz,)
        
        # ── 提取风场变量 ──────────────────────────────────────────────────────────────
        varnames = ["U", "V", "W", "WS"]
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
            site_lon = float(row["lon"])
            site_lat = float(row["lat"])
            
            # 从 JSON 中获取该站点的探测高度层，并过滤掉 >= 2000 的部分
            raw_levels = np.array(station_info_dict[obtid]["levels"], dtype=float)
            target_z = raw_levels[raw_levels < 2000]
        
            # 最近邻：在经纬度平面上找距离最近的格点
            dist2 = (lons - site_lon) ** 2 + (lats - site_lat) ** 2
            j, i = np.unravel_index(np.nanargmin(dist2), dist2.shape)
        
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
            for vn in ["U", "V", "W", "WS"]:
                v_col = np.asarray(col_data[vn], dtype=float)
                v_sorted = v_col[sort_idx]
                f = interp1d(
                    h_sorted, v_sorted,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(v_sorted[0], v_sorted[-1])
                )
                res[vn] = f(target_z)
        
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

out_wrf = os.path.join(OUT_DIR, "WRF_lidar_simulation_1h-rolling.csv")
df_wrf_lidar.to_csv(out_wrf, index=False)
print(f"\n  → 已保存: {out_wrf}  shape={df_wrf_lidar.shape}")

print("\n✅  WRF 提取完成！")
