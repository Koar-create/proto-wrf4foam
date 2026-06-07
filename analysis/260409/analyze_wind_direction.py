import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# ─── 路径配置 ────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv"

OUT_CSV_CIRCULAR = ANALYSIS_DIR / "wind_direction_metrics_circular.csv"
OUT_CSV_VECTOR = ANALYSIS_DIR / "wind_direction_metrics_vector.csv"
OUT_CSV_VEER = ANALYSIS_DIR / "wind_direction_metrics_veer.csv"

HEIGHT_BINS = [0, 300, 1000, 2100]
LAYER_NAMES_EN = ["Low (52-300 m)", "Mid (300-1000 m)", "High (1000-2000 m)"]

def load_and_preprocess(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(str(path), parse_dates=['datetime'])
    
    # 派生基本变量
    df['ws_cfd'] = np.sqrt(df['u_cfd']**2 + df['v_cfd']**2)
    df['wd_obs'] = np.degrees(np.arctan2(-df['u_obs'], -df['v_obs'])) % 360
    df['wd_wrf'] = np.degrees(np.arctan2(-df['u_wrf'], -df['v_wrf'])) % 360
    df['wd_cfd'] = np.degrees(np.arctan2(-df['u_cfd'], -df['v_cfd'])) % 360
    df['layer'] = pd.cut(df['Height'], bins=HEIGHT_BINS, labels=LAYER_NAMES_EN)
    
    # 区分昼夜 LST
    df['lst'] = df['datetime'] + pd.Timedelta(hours=8)
    df['period'] = np.where((df['lst'].dt.hour >= 7) & (df['lst'].dt.hour <= 18), 'Daytime', 'Nighttime')
    
    # QC
    df['qc_ok'] = (df['ws_cfd'] <= 20.0) & ((df['ws_obs'] <= 30.0) | df['ws_obs'].isna())
    
    return df

# ═══════════════════════════════════════════════════════════════════
# 核心计算函数
# ═══════════════════════════════════════════════════════════════════

def circ_diff(m: np.ndarray, o: np.ndarray) -> np.ndarray:
    """循环差值，映射到 [-180, 180)"""
    return (m - o + 180) % 360 - 180

def circular_mbe(wd_m: np.ndarray, wd_o: np.ndarray) -> float:
    delta = circ_diff(wd_m, wd_o)
    return float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(delta))), np.mean(np.cos(np.radians(delta))))))

def circular_rmse(wd_m: np.ndarray, wd_o: np.ndarray) -> float:
    delta = circ_diff(wd_m, wd_o)
    return float(np.sqrt(np.mean(delta**2)))

def direction_accuracy(wd_m: np.ndarray, wd_o: np.ndarray, threshold: float = 22.5) -> float:
    delta = circ_diff(wd_m, wd_o)
    return float(np.mean(np.abs(delta) <= threshold)) * 100.0  # 转为百分比

def vector_rmse(u_m: np.ndarray, v_m: np.ndarray, u_o: np.ndarray, v_o: np.ndarray) -> float:
    return float(np.sqrt(np.mean((u_m - u_o)**2 + (v_m - v_o)**2)))

# ═══════════════════════════════════════════════════════════════════
# 路径分析执行
# ═══════════════════════════════════════════════════════════════════

def path1_circular_stats(df: pd.DataFrame) -> pd.DataFrame:
    """路径一：直接对方向角做循环统计"""
    sub = df[df['qc_ok'] & df['ws_obs'].notna()]
    records = []
    
    # 分层整体统计
    for layer in LAYER_NAMES_EN:
        grp = sub[sub['layer'] == layer]
        if len(grp) < 5: continue
        
        records.append({
            'Category': 'All Times',
            'Layer': layer,
            'N': len(grp),
            'WRF_Circ_MBE': circular_mbe(grp['wd_wrf'], grp['wd_obs']),
            'CFD_Circ_MBE': circular_mbe(grp['wd_cfd'], grp['wd_obs']),
            'WRF_Circ_RMSE': circular_rmse(grp['wd_wrf'], grp['wd_obs']),
            'CFD_Circ_RMSE': circular_rmse(grp['wd_cfd'], grp['wd_obs']),
            'WRF_Acc_22.5(%)': direction_accuracy(grp['wd_wrf'], grp['wd_obs'], 22.5),
            'CFD_Acc_22.5(%)': direction_accuracy(grp['wd_cfd'], grp['wd_obs'], 22.5),
        })
        
    # 分昼夜统计
    for period in ['Daytime', 'Nighttime']:
        for layer in LAYER_NAMES_EN:
            grp = sub[(sub['layer'] == layer) & (sub['period'] == period)]
            if len(grp) < 5: continue
            
            records.append({
                'Category': period,
                'Layer': layer,
                'N': len(grp),
                'WRF_Circ_MBE': circular_mbe(grp['wd_wrf'], grp['wd_obs']),
                'CFD_Circ_MBE': circular_mbe(grp['wd_cfd'], grp['wd_obs']),
                'WRF_Circ_RMSE': circular_rmse(grp['wd_wrf'], grp['wd_obs']),
                'CFD_Circ_RMSE': circular_rmse(grp['wd_cfd'], grp['wd_obs']),
                'WRF_Acc_22.5(%)': direction_accuracy(grp['wd_wrf'], grp['wd_obs'], 22.5),
                'CFD_Acc_22.5(%)': direction_accuracy(grp['wd_cfd'], grp['wd_obs'], 22.5),
            })
            
    return pd.DataFrame(records)

def path2_vector_stats(df: pd.DataFrame) -> pd.DataFrame:
    """路径二：风矢量 RMSE（综合风速与风向误差）"""
    sub = df[df['qc_ok'] & df['ws_obs'].notna()]
    records = []
    
    for layer in LAYER_NAMES_EN:
        grp = sub[sub['layer'] == layer]
        if len(grp) < 5: continue
        records.append({
            'Category': 'All Times',
            'Layer': layer,
            'N': len(grp),
            'WRF_Vector_RMSE': vector_rmse(grp['u_wrf'], grp['v_wrf'], grp['u_obs'], grp['v_obs']),
            'CFD_Vector_RMSE': vector_rmse(grp['u_cfd'], grp['v_cfd'], grp['u_obs'], grp['v_obs']),
        })
        
    for period in ['Daytime', 'Nighttime']:
        for layer in LAYER_NAMES_EN:
            grp = sub[(sub['layer'] == layer) & (sub['period'] == period)]
            if len(grp) < 5: continue
            records.append({
                'Category': period,
                'Layer': layer,
                'N': len(grp),
                'WRF_Vector_RMSE': vector_rmse(grp['u_wrf'], grp['v_wrf'], grp['u_obs'], grp['v_obs']),
                'CFD_Vector_RMSE': vector_rmse(grp['u_cfd'], grp['v_cfd'], grp['u_obs'], grp['v_obs']),
            })
            
    return pd.DataFrame(records)

def path3_veer_stats(df: pd.DataFrame) -> pd.DataFrame:
    """路径三：风向切变 (Veer) 比较"""
    sub = df[df['qc_ok'] & df['ws_obs'].notna()].copy()
    
    # 1. 对每个时次、站点、层计算平均矢量风
    grp = sub.groupby(['datetime', 'obtid', 'period', 'layer'], observed=True).agg({
        'u_obs': 'mean', 'v_obs': 'mean',
        'u_wrf': 'mean', 'v_wrf': 'mean',
        'u_cfd': 'mean', 'v_cfd': 'mean',
        'Height': 'mean'
    }).reset_index().dropna()
    
    # 将 U/V 转回角度
    for pfx in ['obs', 'wrf', 'cfd']:
        grp[f'wd_{pfx}'] = np.degrees(np.arctan2(-grp[f'u_{pfx}'], -grp[f'v_{pfx}'])) % 360
        
    # 2. 提取 Low 和 Mid 层并关联
    low = grp[grp['layer'] == LAYER_NAMES_EN[0]].set_index(['datetime', 'obtid', 'period'])
    mid = grp[grp['layer'] == LAYER_NAMES_EN[1]].set_index(['datetime', 'obtid', 'period'])
    
    # 交集
    common = low.index.intersection(mid.index)
    low = low.loc[common]
    mid = mid.loc[common]
    
    # 计算高度差 (以 100m 为单位)
    dz_100 = (mid['Height'] - low['Height']) / 100.0
    
    veer_df = pd.DataFrame(index=common)
    for pfx in ['obs', 'wrf', 'cfd']:
        # veer = delta_theta / delta_z
        d_theta = circ_diff(mid[f'wd_{pfx}'], low[f'wd_{pfx}'])
        veer_df[f'veer_{pfx}'] = d_theta / dz_100
        
    veer_df.reset_index(inplace=True)
    
    records = []
    
    # 辅助函数：计算 Veer MBE 和 RMSE
    def eval_veer(v_m, v_o):
        return np.mean(v_m - v_o), np.sqrt(np.mean((v_m - v_o)**2))
        
    # 整体
    wm, wr = eval_veer(veer_df['veer_wrf'], veer_df['veer_obs'])
    cm, cr = eval_veer(veer_df['veer_cfd'], veer_df['veer_obs'])
    records.append({
        'Category': 'All Times',
        'N_Profiles': len(veer_df),
        'Obs_Mean_Veer': veer_df['veer_obs'].mean(),
        'WRF_Veer_MBE': wm, 'CFD_Veer_MBE': cm,
        'WRF_Veer_RMSE': wr, 'CFD_Veer_RMSE': cr
    })
    
    # 分昼夜
    for period in ['Daytime', 'Nighttime']:
        p_df = veer_df[veer_df['period'] == period]
        if len(p_df) < 5: continue
        wm, wr = eval_veer(p_df['veer_wrf'], p_df['veer_obs'])
        cm, cr = eval_veer(p_df['veer_cfd'], p_df['veer_obs'])
        records.append({
            'Category': period,
            'N_Profiles': len(p_df),
            'Obs_Mean_Veer': p_df['veer_obs'].mean(),
            'WRF_Veer_MBE': wm, 'CFD_Veer_MBE': cm,
            'WRF_Veer_RMSE': wr, 'CFD_Veer_RMSE': cr
        })
        
    return pd.DataFrame(records)

def main():
    print("Loading data...")
    df = load_and_preprocess(DATA_PATH)
    
    print("\n--- Path 1: Circular Wind Direction Stats ---")
    df_circ = path1_circular_stats(df)
    print(df_circ.to_string(index=False, float_format="%.2f"))
    df_circ.to_csv(OUT_CSV_CIRCULAR, index=False)
    
    print("\n--- Path 2: Vector RMSE Stats ---")
    df_vec = path2_vector_stats(df)
    print(df_vec.to_string(index=False, float_format="%.3f"))
    df_vec.to_csv(OUT_CSV_VECTOR, index=False)
    
    print("\n--- Path 3: Wind Veer (Low to Mid) Stats (deg / 100m) ---")
    df_veer = path3_veer_stats(df)
    print(df_veer.to_string(index=False, float_format="%.3f"))
    df_veer.to_csv(OUT_CSV_VEER, index=False)
    
    print(f"\nOutputs saved to {ANALYSIS_DIR}")

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    main()
