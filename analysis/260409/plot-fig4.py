import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import argparse

# ─── 路径与阈值配置（避免 Hardcoding）────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv"   # 已处理数据
OUTPUT_DIR = REPO_ROOT / "outputs"

WS_MAX_OBS  = 30.0   # m/s: 观测物理上限（仪器异常阈值）
WS_MAX_CFD  = 20.0   # m/s: CFD 上限（>20 视为数值发散）

HEIGHT_BINS = [0, 300, 800, 2100]                          # m：分层边界
LAYER_NAMES_EN = ["Low (52–300 m)",
                  "Mid (300–800 m)",
                  "High (800–2000 m)"]

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

# ─── 全局 Matplotlib 学术样式设置 ───────────────────────────────────────────
def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        'font.family':       'DejaVu Serif',   # 学术字体，类 LaTeX 效果
        'font.size':         10,
        'axes.labelsize':    11,
        'axes.titlesize':    12,
        'axes.titleweight':  'bold',
        'axes.linewidth':    0.8,
        'axes.grid':         True,
        'grid.alpha':        0.25,
        'grid.linestyle':    '--',
        'xtick.direction':   'in',
        'ytick.direction':   'in',
        'xtick.top':         True,
        'ytick.right':       True,
        'legend.framealpha': 0.9,
        'legend.edgecolor':  '0.8',
        'figure.dpi':        120,     # 屏幕显示
        'savefig.dpi':       300,     # 论文保存（GMD 要求 ≥300 DPI）
        'savefig.bbox':      'tight',
    })

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
    return df


def _format_time_label_for_display(t_raw: str, tz: str) -> str:
    """
    t_raw: "YYYY-mm-dd HH:MM:SS" (assumed UTC, consistent with merged CSV)
    tz  : "utc" | "lst" (lst = UTC+8)
    """
    tz = tz.lower()
    dt = pd.Timestamp(t_raw)
    if tz == "lst":
        dt = dt + pd.Timedelta(hours=8)
        suffix = "LST"
    else:
        suffix = "UTC"
    return f"{dt.strftime('%d_%H00')} {suffix}"


def plot_profiles_with_errorbars(df: pd.DataFrame, out_dir: Path = OUTPUT_DIR, tz: str = "utc") -> None:
    """
    更新版：只可视化 WS 的全站 Composite。
    分6次生成图表（3天 × 每天2块 (前/后12小时) = 6张图）。
    布局为 3行 × 4列 (涵盖12小时)。
    """
    tz = tz.lower()
    if tz not in {"utc", "lst"}:
        raise ValueError("tz must be one of: utc, lst")

    height_bins = np.arange(0, 2150, 50)
    legend_handles = [
        Line2D([0], [0], marker='o', ms=5, color=COLOR_OBS, linestyle='none', label='LiDAR (obs)'),
        Line2D([0], [0], color=COLOR_WRF, lw=2, ls='--', label='WRF'),
        Line2D([0], [0], color=COLOR_CFD, lw=2, ls='-', label='OpenFOAM')
    ]
    
    for day in range(1, 4):
        for period_name, h_start in [('AM', 0), ('PM', 12)]:
            hours = [h_start + i for i in range(12)]
            
            fig, axes = plt.subplots(3, 4, figsize=(14, 12), sharex=True, sharey=True, 
                                     constrained_layout=True)
            
            for ax, h in zip(axes.flat, hours):
                t_raw = f"2025-09-{day:02d} {h:02d}:00:00"
                tl_utc = TIME_LABELS.get(t_raw, t_raw)  # 用于筛选数据（UTC）
                tl_disp = _format_time_label_for_display(t_raw, tz=tz)  # 用于显示（UTC/LST）
                
                sub = df[(df['time_label'] == tl_utc) & df['qc_ok']].copy()
                
                if sub.empty:
                    ax.set_title(tl_disp, fontweight='bold', pad=8)
                    continue
                    
                sub['H_bin'] = pd.cut(sub['Height'], bins=height_bins)
                
                sub_obs = sub.dropna(subset=['ws_obs']).copy()
                if not sub_obs.empty:
                    agg_obs = sub_obs.groupby('H_bin', observed=True).agg(
                        mean_ws=('ws_obs', 'mean'), std_ws=('ws_obs', 'std'), mean_h=('Height', 'mean')
                    ).dropna(subset=['mean_h']).reset_index()
                    ax.errorbar(agg_obs['mean_ws'], agg_obs['mean_h'], xerr=agg_obs['std_ws'].fillna(0),
                                fmt='o', ms=3, color=COLOR_OBS, alpha=0.85, elinewidth=0.8, capsize=2, zorder=5)
                
                agg_model = sub.groupby('H_bin', observed=True).agg(
                    mean_h=('Height', 'mean'), mean_wrf=('ws_wrf', 'mean'), mean_cfd=('ws_cfd', 'mean')
                ).dropna(subset=['mean_h']).reset_index()
                
                ax.plot(agg_model['mean_wrf'], agg_model['mean_h'], color=COLOR_WRF, lw=2.0, ls='--')
                ax.plot(agg_model['mean_cfd'], agg_model['mean_h'], color=COLOR_CFD, lw=2.0, ls='-')
                
                for h_line in [300, 800]: ax.axhline(h_line, color='0.6', lw=0.8, ls=':', zorder=0)
                    
                ax.set_title(tl_disp, fontweight='bold', pad=8)
                ax.set_xlim(left=0); ax.set_ylim(0, 1000)
                if ax in axes[-1, :]:
                    ax.set_xlabel(f'Wind Speed (m s$^{{-1}}$) [{tz.upper()}]', fontsize=11)

                    # 让每个子图底部的 xtick 文本也显式带上时区（满足论文图面可读性）
                    tick_locs = ax.get_xticks()
                    ax.set_xticks(tick_locs)
                    ax.set_xticklabels([f"{v:g}\n{tz.upper()}" for v in tick_locs])
                if ax in axes[:, 0]: ax.set_ylabel('Height (m a.g.l.)', fontsize=11)
            
            axes[0, 3].legend(handles=legend_handles, fontsize=10, loc='upper right', framealpha=0.9)
            fig.suptitle(f'Wind Speed Composite Profiles - 2025-09-{day:02d} {period_name}', 
                         fontsize=15, fontweight='bold')
            
            save_path = out_dir / f"fig4_ws_composite_09{day:02d}_{period_name}_tz-{tz}.png"
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            print(f"Saved: {save_path}")

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Fig4 wind speed composite profiles (12h panels).")
    p.add_argument("--tz", choices=["utc", "lst"], default="utc",
                   help="Timezone label for display: utc (default) or lst (UTC+8).")
    return p.parse_args()

def main() -> None:
    args = _parse_args()

    configure_matplotlib_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_and_preprocess(DATA_PATH)
    print(f"Loaded data: {len(df_raw):,} rows × {df_raw.shape[1]} columns")
    print(f"Sites: {sorted(df_raw['obtid'].unique())}")
    print(f"Time labels: {sorted(df_raw['time_label'].unique())}")
    print(f"Height range: {df_raw['Height'].min():.1f}–{df_raw['Height'].max():.1f} m")

    df = quality_control(df_raw)
    print("\nGenerating 6 WS composite figures...")
    plot_profiles_with_errorbars(df, OUTPUT_DIR, tz=args.tz)


if __name__ == "__main__":
    main()