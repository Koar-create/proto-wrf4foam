import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

# ─── 路径与阈值配置（避免 Hardcoding）────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv"   # 已处理数据

WS_MAX_OBS  = 30.0   # m/s: 观测物理上限（仪器异常阈值）
WS_MAX_CFD  = 20.0   # m/s: CFD 上限（>20 视为数值发散）

HEIGHT_BINS = [0, 300, 1000, 2100]                          # m：分层边界
LAYER_NAMES_EN = ["Low (52–300 m)",
                  "Mid (300–1000 m)",
                  "High (1000–2000 m)"]

'''
TIME_LABELS = {
    "2025-09-03 00:00:00": "0000 UTC",
    "2025-09-03 04:00:00": "0400 UTC",
    "2025-09-03 08:00:00": "0800 UTC",
    "2025-09-03 12:00:00": "1200 UTC",
    "2025-09-03 16:00:00": "1600 UTC",
    "2025-09-03 20:00:00": "2000 UTC"
}
'''
TIME_LABELS = {f"2025-09-{d:02d} {h:02d}:00:00": f"{d:02d}_{h:02d}00 UTC"
               for d in range(1, 4) for h in range(24)}

# ─── 色盲友好配色（IBM Color Blind Safe Palette 变体）────────────────────────
COLOR_OBS = "#1a1a2e"   # 深蓝黑 – LiDAR 观测
COLOR_WRF = "#e07b39"   # 橙色   – WRF 中尺度
COLOR_CFD = "#2196a5"   # 青蓝   – OpenFOAM CFD

def load_and_preprocess(path: str | Path) -> pd.DataFrame:
    """
    加载已对齐的三源数据 CSV，派生所有需要的物理量。

    派生量：
      - ws_cfd  : CFD 水平风速 = sqrt(u²+v²)
      - wd_obs/wrf/cfd : 气象风向（北=0°, 顺时针, 风从何方来）
      - layer   : 高度分层标签（低/中/高）
      - time_label : 人类可读时次标签
    """
    df = pd.read_csv(str(path), parse_dates=['datetime'])

    # 1. 派生 CFD 水平风速（原始只有 u_cfd, v_cfd）
    df['ws_cfd'] = np.sqrt(df['u_cfd']**2 + df['v_cfd']**2)

    # 2. 气象风向（arctan2 参数顺序：取反后为"从何方来"）
    df['wd_obs'] = np.degrees(np.arctan2(-df['u_obs'], -df['v_obs'])) % 360
    df['wd_wrf'] = np.degrees(np.arctan2(-df['u_wrf'], -df['v_wrf'])) % 360
    df['wd_cfd'] = np.degrees(np.arctan2(-df['u_cfd'], -df['v_cfd'])) % 360

    # 3. 高度分层
    df['layer'] = pd.cut(df['Height'], bins=HEIGHT_BINS, labels=LAYER_NAMES_EN)

    # 4. 可读时次标签
    df['time_label'] = df['datetime'].astype(str).map(TIME_LABELS)

    return df


def quality_control(df: pd.DataFrame,
                   ws_max_obs: float = WS_MAX_OBS,
                   ws_max_cfd: float = WS_MAX_CFD) -> pd.DataFrame:
    """
    数据质量控制（QC）。

    规则（按优先级）：
    ① obs NaN → 保留行但排除于统计（LiDAR 信噪比不足的缺测层，属有效缺测）
    ② ws_obs > ws_max_obs → 仪器异常，标记为无效
    ③ ws_cfd > ws_max_cfd → OpenFOAM 数值发散（RANS 在偏斜网格局部不收敛），标记整行无效

    注意：不对 WRF 执行额外 QC，WRF 数据本身经过诊断，物理范围合理。

    返回：原始 df 加 3 个布尔掩码列（qc_obs_ok, qc_cfd_ok, qc_ok）。
    """
    n0 = len(df)

    # CFD 发散标记
    cfd_ok  = df['ws_cfd'] <= ws_max_cfd
    # 观测物理上限（NaN 不触发该规则）
    obs_ok  = (df['ws_obs'] <= ws_max_obs) | df['ws_obs'].isna()

    df = df.copy()
    df['qc_obs_ok'] = obs_ok
    df['qc_cfd_ok'] = cfd_ok
    df['qc_ok']     = obs_ok & cfd_ok   # 行级别总掩码

    n_removed = (~df['qc_ok']).sum()
    n_nan_obs = df['ws_obs'].isna().sum()

    '''
    print(f"[QC] Rows (raw): {n0:,}")
    print(f"[QC] Rows removed (divergent/outlier): {n_removed:,} ({100*n_removed/n0:.1f}%)")
    print(f"[QC] LiDAR missing (NaN): {n_nan_obs:,} ({100*n_nan_obs/n0:.1f}%)")
    print(f"[QC] Usable paired rows (qc_ok & obs not NaN): "
          f"{(df['qc_ok'] & df['ws_obs'].notna()).sum():,}")

    # ── 诊断：发散行来自哪些时次/站点？────────────────────────────────────
    if n_removed > 0:
        print("\n[QC Diagnostic] Divergent rows by (time × site):")
        print(df[~df['qc_ok']].groupby(
            ['time_label','obtid'], observed=True)['ws_cfd']
              .agg(['count','max']).rename(columns={'count':'N_divergent','max':'ws_cfd_max'})
              .round(2).to_string())
    '''
    return df


# ═══════════════════════════════════════════════════════════════════
# 2.1 风速标量指标函数
# ═══════════════════════════════════════════════════════════════════

def mean_bias_error(model: np.ndarray, obs: np.ndarray) -> float:
    """平均偏差 MBE = mean(M - O)；正值=高估，负值=低估"""
    return float(np.nanmean(model - obs))


def rmse(model: np.ndarray, obs: np.ndarray) -> float:
    """均方根误差 RMSE"""
    return float(np.sqrt(np.nanmean((model - obs)**2)))


def index_of_agreement(model: np.ndarray, obs: np.ndarray) -> float:
    """
    Willmott (1981) 一致性指数 d ∈ [0,1]。
    公式：d = 1 - Σ(M-O)² / Σ(|M-Ō|+|O-Ō|)²
    d=1 为完美，d=0 为最差。
    """
    obs_mean    = np.nanmean(obs)
    numerator   = np.nansum((model - obs)**2)
    denominator = np.nansum((np.abs(model - obs_mean) + np.abs(obs - obs_mean))**2)
    if denominator == 0:
        return np.nan
    return float(1 - numerator / denominator)


def fractional_bias(model: np.ndarray, obs: np.ndarray) -> float:
    """
    分数偏差 FB = 2(M̄-Ō)/(M̄+Ō)，FB ∈ [-2,2]。
    |FB|<0.3 通常为可接受范围（大气模型评估惯例）。
    """
    m_bar = np.nanmean(model)
    o_bar = np.nanmean(obs)
    denom = m_bar + o_bar
    return float(2 * (m_bar - o_bar) / denom) if denom != 0 else np.nan


def skill_score(model: np.ndarray,
                obs: np.ndarray,
                baseline: np.ndarray) -> float:
    """
    技巧得分 SS = 1 - MSE(model) / MSE(baseline)。
    SS > 0  → model 优于 baseline（此处 baseline = WRF）
    SS = 0  → model 与 baseline 持平
    SS < 0  → model 劣于 baseline
    """
    mse_model    = np.nanmean((model - obs)**2)
    mse_baseline = np.nanmean((baseline - obs)**2)
    return float(1 - mse_model / mse_baseline) if mse_baseline != 0 else np.nan


# ═══════════════════════════════════════════════════════════════════
# 2.2 风向圆形统计函数（重点）
# ═══════════════════════════════════════════════════════════════════

def circular_error_stats(wd_model: np.ndarray,
                         wd_obs: np.ndarray) -> tuple:
    """
    计算风向（度）的圆形统计误差，正确处理 0°/360° 环绕问题。

    步骤：
    1. 计算角度差 Δθ = θ_M - θ_O，映射到 (-180°, 180°]
    2. 圆形均值误差：通过向量合成 arctan2(mean(sinΔ), mean(cosΔ))
    3. 圆形 RMSE：sqrt(mean(Δθ²))（已映射到(-180,180]后的欧氏距离近似）

    注意：此方法对小偏差（<60°）精度极高；对于大偏差（>90°）
    圆形 RMSE 在物理上意味着模型无法捕捉风向特征。

    Returns:
        circ_mean_err (°): 圆形均值误差（有符号）
        circ_rmse (°)    : 圆形均方根误差
    """
    wd_model = np.asarray(wd_model, dtype=float)
    wd_obs   = np.asarray(wd_obs,   dtype=float)

    # 有效数据掩码（任一为 NaN 则排除）
    valid = ~(np.isnan(wd_model) | np.isnan(wd_obs))
    if valid.sum() < 3:
        return np.nan, np.nan

    delta = wd_model[valid] - wd_obs[valid]
    # 环绕映射到 (-180, 180]
    delta = (delta + 180) % 360 - 180

    # 圆形均值误差（向量合成法，来自 Berens 2009）
    circ_mean_err = float(np.degrees(
        np.arctan2(np.mean(np.sin(np.radians(delta))),
                   np.mean(np.cos(np.radians(delta))))))

    circ_rmse = float(np.sqrt(np.mean(delta**2)))

    return circ_mean_err, circ_rmse


def compute_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 (站点 × 时次 × 高度层) 子集计算全套统计指标。

    筛选规则：qc_ok=True AND ws_obs 非 NaN（即有效观测配对）。
    返回：宽格式 DataFrame，每行一个 (站点, 时次, 层) 组合。
    """
    subset  = df[df['qc_ok'] & df['ws_obs'].notna()].copy()
    records = []

    for (site, time, layer), grp in subset.groupby(
            ['obtid','time_label','layer'], observed=True):
        if len(grp) < 5:   # 样本量不足时跳过，避免统计不稳定
            continue

        obs = grp['ws_obs'].values
        wrf = grp['ws_wrf'].values
        cfd = grp['ws_cfd'].values

        wrf_cme, wrf_crmse = circular_error_stats(grp['wd_wrf'].values,
                                                   grp['wd_obs'].values)
        cfd_cme, cfd_crmse = circular_error_stats(grp['wd_cfd'].values,
                                                   grp['wd_obs'].values)

        records.append({
            'Site'             : site,
            'Time'             : time,
            'Layer'            : layer,
            'N'                : len(grp),
            # ── WRF 风速 ──────────────────────────────────────────────
            'WRF_MBE'          : mean_bias_error(wrf, obs),
            'WRF_RMSE'         : rmse(wrf, obs),
            'WRF_IoA'          : index_of_agreement(wrf, obs),
            'WRF_FB'           : fractional_bias(wrf, obs),
            # ── CFD 风速 ──────────────────────────────────────────────
            'CFD_MBE'          : mean_bias_error(cfd, obs),
            'CFD_RMSE'         : rmse(cfd, obs),
            'CFD_IoA'          : index_of_agreement(cfd, obs),
            'CFD_FB'           : fractional_bias(cfd, obs),
            # ── 技巧得分（以 WRF 为基准） ────────────────────────────
            'CFD_SS_vs_WRF'    : skill_score(cfd, obs, wrf),
            # ── 风向圆形统计 ──────────────────────────────────────────
            'WRF_WD_CME_circ'  : wrf_cme,
            'WRF_WD_CRMSE_circ': wrf_crmse,
            'CFD_WD_CME_circ'  : cfd_cme,
            'CFD_WD_CRMSE_circ': cfd_crmse,
        })

    result    = pd.DataFrame(records)
    num_cols  = result.select_dtypes(include='number').columns
    result[num_cols] = result[num_cols].round(3)
    return result


def main() -> None:
    df_raw = load_and_preprocess(DATA_PATH)
    # print(f"Loaded data: {len(df_raw):,} rows × {df_raw.shape[1]} columns")
    # print(f"Sites: {sorted(df_raw['obtid'].unique())}")
    # print(f"Time labels: {sorted(df_raw['time_label'].unique())}")
    # print(f"Height range: {df_raw['Height'].min():.1f}–{df_raw['Height'].max():.1f} m")

    df = quality_control(df_raw)
    metrics = compute_metrics_table(df)

    # ── 展示格式优化 ──────────────────────────────────────────────────────────────
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 160)
    pd.set_option('display.float_format', '{:.3f}'.format)

    # print(f"\nMetrics table: {len(metrics)} rows (Site × Time × Layer)")
    # print(metrics.to_string(index=False))

    # ─── 分层聚合汇总（全站点、全时次）────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Layer-aggregated summary (all sites + all times)")
    print("=" * 70)

    sub_df = df[df['qc_ok'] & df['ws_obs'].notna()].copy()
    print("\n-- All times --")
    header = f"{'Layer':<22} {'N':>5}  {'WRF_MBE':>8} {'WRF_RMSE':>9} "
    header += f"{'WRF_IoA':>8}  {'CFD_MBE':>8} {'CFD_RMSE':>9} {'CFD_IoA':>8}  {'SS':>7}"
    print(header)
    print("-" * len(header))

    for layer in LAYER_NAMES_EN:
        g = sub_df[sub_df['layer'] == layer]
        if len(g) < 5:
            continue
        o, w, c = g['ws_obs'].values, g['ws_wrf'].values, g['ws_cfd'].values
        row = (f"{layer:<22} {len(g):>5}  "
               f"{mean_bias_error(w, o):>+8.3f} {rmse(w, o):>9.3f} "
               f"{index_of_agreement(w, o):>8.3f}  "
               f"{mean_bias_error(c, o):>+8.3f} {rmse(c, o):>9.3f} "
               f"{index_of_agreement(c, o):>8.3f}  "
               f"{skill_score(c, o, w):>+7.3f}")
        print(row)


if __name__ == "__main__":
    main()