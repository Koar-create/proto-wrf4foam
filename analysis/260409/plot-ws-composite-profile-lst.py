"""
plot-fig4-lst.py — 与 plot-fig4.py 同源，但按 **LST(本地标准时, UTC+8)** 重组每张图的 12 个子图。

分组规则（每天产出 2 张图，共 6 张）：
  AM 图：LST 07:00 – 18:00          （白天 12 小时）
  PM 图：LST 19:00 – 次日 06:00     （夜间 12 小时, 跨日）

每张图布局保持 3 行 × 4 列（r3c4），按时序从左到右、从上到下排列。
数据筛选仍按 UTC（与 CSV 中 datetime 列一致），仅展示标签按时区切换。

边界情况：09-01 AM 的第 1 格（r1c1）= LST 09-01 07:00 = UTC 08-31 23:00，
该 UTC 时刻不在实验/观测可用范围内（CSV 仅覆盖 2025-09-01..09-03），
因此显式调用 axis('off') 留空，避免误导。
"""

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
DATA_PATH = REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv"
OUTPUT_DIR = REPO_ROOT / "results/ws_composite_profile/260409"

WS_MAX_OBS = 30.0   # m/s
WS_MAX_CFD = 20.0   # m/s

HEIGHT_BINS = [0, 300, 800, 2100]
LAYER_NAMES_EN = ["Low (52–300 m)",
                  "Mid (300–800 m)",
                  "High (800–2000 m)"]

# CSV 中可用 UTC 时刻范围 → 用于过滤 LST 跨界到 08-31 / 09-04 的格子
TIME_LABELS = {f"2025-09-{d:02d} {h:02d}:00:00": f"{d:02d}_{h:02d}00 UTC"
               for d in range(1, 4) for h in range(24)}

# ─── 色盲友好配色（与 plot-fig4.py 保持一致）─────────────────────────────────
COLOR_OBS = "#1a1a2e"
COLOR_WRF = "#e07b39"
COLOR_CFD = "#2196a5"


def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        'font.family':       'DejaVu Serif',
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
        'figure.dpi':        120,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
    })


def load_and_preprocess(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(str(path), parse_dates=['datetime'])

    df['ws_cfd'] = np.sqrt(df['u_cfd']**2 + df['v_cfd']**2)

    df['wd_obs'] = np.degrees(np.arctan2(-df['u_obs'], -df['v_obs'])) % 360
    df['wd_wrf'] = np.degrees(np.arctan2(-df['u_wrf'], -df['v_wrf'])) % 360
    df['wd_cfd'] = np.degrees(np.arctan2(-df['u_cfd'], -df['v_cfd'])) % 360

    df['layer'] = pd.cut(df['Height'], bins=HEIGHT_BINS, labels=LAYER_NAMES_EN)

    df['time_label'] = df['datetime'].astype(str).map(TIME_LABELS)

    return df


def quality_control(df: pd.DataFrame,
                    ws_max_obs: float = WS_MAX_OBS,
                    ws_max_cfd: float = WS_MAX_CFD) -> pd.DataFrame:
    n0 = len(df)

    cfd_ok = df['ws_cfd'] <= ws_max_cfd
    obs_ok = (df['ws_obs'] <= ws_max_obs) | df['ws_obs'].isna()

    df = df.copy()
    df['qc_obs_ok'] = obs_ok
    df['qc_cfd_ok'] = cfd_ok
    df['qc_ok'] = obs_ok & cfd_ok

    n_removed = (~df['qc_ok']).sum()
    n_nan_obs = df['ws_obs'].isna().sum()

    print(f"[QC] Rows (raw): {n0:,}")
    print(f"[QC] Rows removed (divergent/outlier): {n_removed:,} ({100*n_removed/n0:.1f}%)")
    print(f"[QC] LiDAR missing (NaN): {n_nan_obs:,} ({100*n_nan_obs/n0:.1f}%)")
    print(f"[QC] Usable paired rows (qc_ok & obs not NaN): "
          f"{(df['qc_ok'] & df['ws_obs'].notna()).sum():,}")

    if n_removed > 0:
        print("\n[QC Diagnostic] Divergent rows by (time × site):")
        print(df[~df['qc_ok']].groupby(
            ['time_label', 'obtid'], observed=True)['ws_cfd']
              .agg(['count', 'max']).rename(columns={'count': 'N_divergent', 'max': 'ws_cfd_max'})
              .round(2).to_string())
    return df


# ─── LST↔UTC 工具函数 ───────────────────────────────────────────────────────
def _lst_slots_for(day: int, period: str) -> list[pd.Timestamp]:
    """
    返回某 LST 日某时段的 12 个 LST 时间戳。

    period == 'AM' → LST [day 07:00, day 08:00, …, day 18:00]    (12 个)
    period == 'PM' → LST [day 19:00, …, day 23:00, day+1 00:00, …, day+1 06:00]  (12 个)
    """
    if period == 'AM':
        return [pd.Timestamp(f"2025-09-{day:02d} {h:02d}:00:00") for h in range(7, 19)]
    # PM: 跨日
    same = [pd.Timestamp(f"2025-09-{day:02d} {h:02d}:00:00") for h in range(19, 24)]
    nxt = [pd.Timestamp(f"2025-09-{day+1:02d} {h:02d}:00:00") for h in range(0, 7)]
    return same + nxt


def _lst_to_utc(lst_ts: pd.Timestamp) -> pd.Timestamp:
    return lst_ts - pd.Timedelta(hours=8)


def _format_display_label(lst_ts: pd.Timestamp, tz: str) -> str:
    """根据 tz 选择子图标题显示的时区: 'lst' 或 'utc'。"""
    if tz == "lst":
        return f"{lst_ts.strftime('%d_%H00')} LST"
    utc_ts = _lst_to_utc(lst_ts)
    return f"{utc_ts.strftime('%d_%H00')} UTC"


# ─── 主绘图函数 ─────────────────────────────────────────────────────────────
def plot_profiles_with_errorbars(df: pd.DataFrame,
                                 out_dir: Path = OUTPUT_DIR,
                                 tz: str = "lst") -> None:
    """
    按 LST 日 × {AM, PM} 共 6 组、每组 12 子图(r3c4) 绘制风速复合廓线。
    数据仍以 UTC 时刻过滤 (与 CSV 列对齐)；子图标题按 tz 显示 LST 或 UTC。
    x 轴仅表示风速物理量，不在 xlabel/xtick 重复标注时区（避免误读为「LST 风速」）。
    """
    tz = tz.lower()
    if tz not in {"utc", "lst"}:
        raise ValueError("tz must be one of: utc, lst")

    height_bins = np.arange(0, 2150, 50)
    legend_handles = [
        Line2D([0], [0], marker='o', ms=5, color=COLOR_OBS, linestyle='none', label='LiDAR (obs)'),
        Line2D([0], [0], color=COLOR_WRF, lw=2, ls='--', label='WRF'),
        Line2D([0], [0], color=COLOR_CFD, lw=2, ls='-', label='OpenFOAM'),
    ]

    for day in range(1, 4):
        for period_name in ('AM', 'PM'):
            lst_slots = _lst_slots_for(day, period_name)

            fig, axes = plt.subplots(3, 4, figsize=(14, 12),
                                     sharex=True, sharey=True,
                                     constrained_layout=True)

            for ax, lst_ts in zip(axes.flat, lst_slots):
                utc_ts = _lst_to_utc(lst_ts)
                t_raw = utc_ts.strftime("%Y-%m-%d %H:%M:%S")
                tl_utc = TIME_LABELS.get(t_raw)

                # UTC 越界 (例如 08-31 23:00 或 09-04 xx:00) → 数据缺失, 留白
                if tl_utc is None:
                    ax.axis('off')
                    continue

                tl_disp = _format_display_label(lst_ts, tz=tz)

                sub = df[(df['time_label'] == tl_utc) & df['qc_ok']].copy()

                if sub.empty:
                    ax.set_title(tl_disp, fontweight='bold', pad=8)
                    continue

                sub['H_bin'] = pd.cut(sub['Height'], bins=height_bins)

                sub_obs = sub.dropna(subset=['ws_obs']).copy()
                if not sub_obs.empty:
                    agg_obs = sub_obs.groupby('H_bin', observed=True).agg(
                        mean_ws=('ws_obs', 'mean'),
                        std_ws=('ws_obs', 'std'),
                        mean_h=('Height', 'mean'),
                    ).dropna(subset=['mean_h']).reset_index()
                    ax.errorbar(agg_obs['mean_ws'], agg_obs['mean_h'],
                                xerr=agg_obs['std_ws'].fillna(0),
                                fmt='o', ms=3, color=COLOR_OBS, alpha=0.85,
                                elinewidth=0.8, capsize=2, zorder=5)

                agg_model = sub.groupby('H_bin', observed=True).agg(
                    mean_h=('Height', 'mean'),
                    mean_wrf=('ws_wrf', 'mean'),
                    mean_cfd=('ws_cfd', 'mean'),
                ).dropna(subset=['mean_h']).reset_index()

                ax.plot(agg_model['mean_wrf'], agg_model['mean_h'],
                        color=COLOR_WRF, lw=2.0, ls='--')
                ax.plot(agg_model['mean_cfd'], agg_model['mean_h'],
                        color=COLOR_CFD, lw=2.0, ls='-')

                for h_line in (300, 800):
                    ax.axhline(h_line, color='0.6', lw=0.8, ls=':', zorder=0)

                ax.set_title(tl_disp, fontweight='bold', pad=8)
                ax.set_xlim(left=0)
                ax.set_ylim(0, 1000)

                if ax in axes[-1, :]:
                    ax.set_xlabel(r'Wind Speed (m s$^{-1}$)', fontsize=11)
                if ax in axes[:, 0]:
                    ax.set_ylabel('Height (m a.g.l.)', fontsize=11)

            axes[0, 3].legend(handles=legend_handles, fontsize=10,
                              loc='upper right', framealpha=0.9)

            period_desc = "07-18 LST" if period_name == 'AM' else "19 LST – next 06 LST"
            fig.suptitle(
                f'Wind Speed Composite Profiles — LST-grouped 2025-09-{day:02d} {period_name} '
                f'({period_desc})',
                fontsize=15, fontweight='bold')

            save_path = out_dir / f"fig4_ws_composite_lstgrp_09{day:02d}_{period_name}_tz-{tz}.png"
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            print(f"Saved: {save_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Fig4 wind speed composite profiles, grouped by LST day "
                    "(AM = LST 07–18, PM = LST 19 → next 06).")
    p.add_argument("--tz", choices=["utc", "lst"], default="lst",
                   help="Subplot title time zone only: 'lst' (default) or 'utc'. "
                        "X-axis label is wind speed only (no LST/UTC suffix).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    configure_matplotlib_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_and_preprocess(DATA_PATH)
    print(f"Loaded data: {len(df_raw):,} rows × {df_raw.shape[1]} columns")
    print(f"Sites: {sorted(df_raw['obtid'].unique())}")
    print(f"Time labels: {sorted(df_raw['time_label'].dropna().unique())}")
    print(f"Height range: {df_raw['Height'].min():.1f}–{df_raw['Height'].max():.1f} m")

    df = quality_control(df_raw)
    print("\nGenerating 6 WS composite figures (LST-grouped)...")
    plot_profiles_with_errorbars(df, OUTPUT_DIR, tz=args.tz)


if __name__ == "__main__":
    main()
