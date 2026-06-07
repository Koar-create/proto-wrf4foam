import zipfile
import io
import re
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 处理 Windows 控制台编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ==========================================
# 湍流估算常数（与 WRF 提取脚本保持一致）
# ==========================================
CMU = 0.09          # k-epsilon 模型常数 C_μ
KAPPA = 0.41        # von Kármán 常数
L_MIX_MIN = 10.0    # 混合长度下限 (m)
L_MIX_MAX = 100.0   # 混合长度上限 (m)

# ==========================================
# 1. 命令行参数解析
# ==========================================
_root = _repo_root()
parser = argparse.ArgumentParser(description="LiDAR DBS_096 数据预处理脚本")
parser.add_argument(
    "mode",
    nargs="?",
    default="transient",
    choices=["transient", "mean"],
    help="处理模式: transient (原始分钟级数据) 或 mean (1小时滑动平均)",
)
parser.add_argument(
    "--lidar-zip-dir",
    type=Path,
    default=_root / "data" / "lidar",
    help="各站点 LiDAR 压缩包所在目录（默认: 仓库根下 data/lidar，即 <st>.zip）",
)
parser.add_argument(
    "--out-dir",
    type=Path,
    default=_root / "data" / "260409" / "raw" / "lidar",
    help="输出 CSV 目录（默认: data/260409/raw/lidar）",
)
args = parser.parse_args()

# 读取 LiDAR 站点高度信息（用于 epsilon 估算中的高度 z）
_STATION_JSON = _root / "util" / "lidar_station_info.json"
with open(_STATION_JSON, "r", encoding="utf-8") as _f:
    _station_info = json.load(_f)

def read_nested_lidar_096(zip_path, target_date_str):
    """
    智能读取嵌套或直排的 LiDAR DBS_096 数据
    """
    station_id = Path(zip_path).stem
    y, m, d = target_date_str.split('/')
    date_compact = f"{y}{m}{d}"  # 转换为 20250903 格式
    
    data_rows = []
    
    # --- 内部解析器函数 ---
    def parse_bin_file(f_obj, filename):
        # 1. 从文件名中提取时间戳 (格式如: 20250903123000)
        time_match = re.search(rf"{date_compact}(\d{{6}})", filename)
        time_str = time_match.group(1) if time_match else "000000"
        timestamp = pd.to_datetime(f"{date_compact}{time_str}")
        
        # 2. 读取文本流
        text_stream = io.TextIOWrapper(f_obj, encoding='latin-1')
        for line in text_stream:
            line = line.strip()
            if not line: continue
            # 匹配五位数字开头的数据行
            if line[0].isdigit() and re.match(r'^\d{5}\s+[\d\./-]+', line):
                parts = line.split()
                # 提取前4列，处理缺失值符号 '/'
                row = [float(p) if '/' not in p else np.nan for p in parts[:4]]
                # 将站点ID、时间、与气象数据组合
                data_rows.append([station_id, timestamp] + row)

    # --- 穿透ZIP目录查找文件 ---
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            all_paths = z.namelist()
            
            # 可能的三种存储路径模式
            direct_prefix = f"{station_id}/DBS_096/{target_date_str}/"
            monthly_zip = f"{station_id}/DBS_096/{y}/{m}.zip"
            daily_zip = f"{station_id}/DBS_096/{y}/{m}/{d}.zip"
            
            file_count = 0
            
            # 场景 A: 直排文件 (如 GAW102, GAW110)
            for p in all_paths:
                if p.startswith(direct_prefix) and p.endswith('.BIN') and date_compact in p:
                    with z.open(p) as f:
                        parse_bin_file(f, p)
                        file_count += 1
                        
            # 场景 B: 嵌套的月度压缩包 (如 GAW103, GAW104, GAW105, GAW111)
            if monthly_zip in all_paths:
                nested_data = z.read(monthly_zip)
                with zipfile.ZipFile(io.BytesIO(nested_data)) as inner_z:
                    for p in inner_z.namelist():
                        if p.endswith('.BIN') and date_compact in p:
                            with inner_z.open(p) as f:
                                parse_bin_file(f, p)
                                file_count += 1
                                
            # 场景 C: 嵌套的日度压缩包 (备用兼容)
            if daily_zip in all_paths:
                nested_data = z.read(daily_zip)
                with zipfile.ZipFile(io.BytesIO(nested_data)) as inner_z:
                    for p in inner_z.namelist():
                        if p.endswith('.BIN'):
                            with inner_z.open(p) as f:
                                parse_bin_file(f, p)
                                file_count += 1
            
            print(f"✅ {station_id}: 成功读取 {file_count} 个分钟级文件。")

    except Exception as e:
        print(f"❌ {station_id}: 读取失败 -> {e}")
        
    return data_rows

# ==========================================
# 主执行流程：遍历6个站点，合并 DataFrame
# ==========================================
stations = ['GAW103', 'GAW104', 'GAW105', 'GAW111']  # ['GAW102', 'GAW103', 'GAW104', 'GAW105', 'GAW110', 'GAW111']
target_dates = ["2025/09/01", "2025/09/02", "2025/09/03", "2025/09/04", "2025/09/05", "2025/09/06"]

all_data = []

print(f"🚀 开始批量处理 LiDAR DBS_096 数据 ({target_dates[0]} ~ {target_dates[-1]})，模式: {args.mode}")
print(f"   输入目录: {args.lidar_zip_dir}")
print(f"   输出目录: {args.out_dir}")
for target_date in target_dates:
    for st in stations:
        zip_path = args.lidar_zip_dir / f"{st}.zip"
        if zip_path.is_file():
            rows = read_nested_lidar_096(str(zip_path), target_date)
            all_data.extend(rows)
        else:
            print(f"⚠️ 未找到文件: {zip_path}")

# 1. 创建全局 DataFrame
df_lidar_raw = pd.DataFrame(
    all_data, 
    columns=['obtid', 'datetime', 'Height', 'WindDir', 'WindSpd', 'VerticalSpd']
)

# 2. 矢量分解 (U / V 分量计算)
df_lidar_raw['rad'] = np.radians(df_lidar_raw['WindDir'])
df_lidar_raw['U'] = -df_lidar_raw['WindSpd'] * np.sin(df_lidar_raw['rad'])
df_lidar_raw['V'] = -df_lidar_raw['WindSpd'] * np.cos(df_lidar_raw['rad'])
df_lidar_raw = df_lidar_raw.drop(columns=['rad']) # 删掉中间变量

# 3. 基础清洗：强制将秒钟抹零，统一到分钟级，并处理重复观测
df_lidar_raw['datetime'] = df_lidar_raw['datetime'].dt.floor('min')

agg_dict = {
    'WindDir': 'mean',
    'WindSpd': 'mean',
    'VerticalSpd': 'mean',
    'U': 'mean',
    'V': 'mean'
}
df_lidar_raw = df_lidar_raw.groupby(['obtid', 'datetime', 'Height'], as_index=False).agg(agg_dict)

# ==========================================
# 分支处理逻辑
# ==========================================
if args.mode == "mean":
    print("⏳ 正在应用 1h 滑动平均 (min_periods=3) ...")
    # ── 根因修复说明 ──────────────────────────────────────────────────────
    # 问题：pandas rolling('1h') 的输出索引与输入索引一一对应（不插值）。
    # 若 LiDAR 仪器在某整点 T 恰好没有输出观测记录（仪器扫描周期偏差、
    # DBS 文件丢包等），则 rolling 输出中也不会有 T 这一行，后续
    # minute==0 的整点筛选自然找不到该时刻数据。
    # （已确认：GAW103 @ 00:00 UTC 和 GAW111 @ 08:00 UTC 即属此情况）
    #
    # 修复：在 rolling 之前先对每个 (站点, 高度) 组执行 resample('1min')，
    # 将稀疏的原始时间序列补齐为规则的 1min 网格。缺失的分钟填 NaN，
    # rolling 仍可在该整点产生有效均值（前提是窗口内有 ≥ min_periods 条非 NaN）。
    # ─────────────────────────────────────────────────────────────────────

    df_lidar_raw = df_lidar_raw.sort_values(by=['obtid', 'Height', 'datetime'])
    df_lidar_raw = df_lidar_raw.set_index('datetime')

    results = []
    groups = list(df_lidar_raw.groupby(['obtid', 'Height']))
    print(f"  处理 {len(groups)} 个 (站点, 高度) 组合...")
    for (obtid, height), group in groups:
        g = group[['U', 'V', 'VerticalSpd']].copy()
        # Step 1: resample 到规则 1min 网格（缺失分钟 → NaN）
        g_resampled = g.resample('1min').mean()
        
        # Step 2: 1h 中心滑动平均和方差（用于湍流估算）
        roller = g_resampled.rolling('1h', center=True, min_periods=3)
        g_rolled = roller.mean()
        g_var = roller.var(ddof=1)
        
        # 估算 k (TKE) = 0.5 * (var(U) + var(V) + var(W))
        k_est = 0.5 * (g_var['U'] + g_var['V'] + g_var['VerticalSpd'])
        # 确保 TKE 不小于一个极小值，避免计算 epsilon 时报错或无物理意义
        k_est = np.maximum(k_est, 1e-6)
        
        # 估算 epsilon
        mixing_length = np.clip(KAPPA * height, L_MIX_MIN, L_MIX_MAX)
        eps_est = (CMU**0.75) * (k_est**1.5) / mixing_length
        eps_est = np.maximum(eps_est, 1e-8)
        
        g_rolled['k'] = k_est
        g_rolled['epsilon'] = eps_est
        
        g_rolled['obtid']  = obtid
        g_rolled['Height'] = height
        results.append(g_rolled.reset_index())

    df_rolled = pd.concat(results, ignore_index=True)

    # 抽取整点数据（整点现在一定存在于 rolling 输出中）
    df_rolled = df_rolled[df_rolled['datetime'].dt.minute == 0].copy()

    # 反算标量风速风向
    df_rolled['WindSpd'] = np.sqrt(df_rolled['U']**2 + df_rolled['V']**2)
    # 气象学风向转换为 0-360 度，0度为正北
    df_rolled['WindDir'] = (270 - np.degrees(np.arctan2(df_rolled['V'], df_rolled['U']))) % 360

    df_final = df_rolled.sort_values(by=['obtid', 'datetime', 'Height']).reset_index(drop=True)
    out_name = "lidar_1h-rolling.csv"
else:
    # transient 模式
    df_lidar_raw['k'] = np.nan
    df_lidar_raw['epsilon'] = np.nan
    df_final = df_lidar_raw.sort_values(by=['obtid', 'datetime', 'Height']).reset_index(drop=True)
    out_name = "lidar_transient.csv"

print(f"\n🎉 全部处理完成！合并后的 DataFrame ({args.mode}) 预览:")
print(df_final.info())
print("\n", df_final.head())

args.out_dir.mkdir(parents=True, exist_ok=True)
out_path = args.out_dir / out_name
df_final.to_csv(out_path, index=False)
print(f"已写入: {out_path}")
