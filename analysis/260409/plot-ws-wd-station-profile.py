"""按站点绘制 WS/WD 垂直廓线（4 小时 × 4 时次/图，UTC 分组）。"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── 路径与阈值 ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv"
OUTPUT_DIR = REPO_ROOT / "results/ws_station_profile/260409/by_hour"
TZ_TAG = "utc"  # 数据筛选与 CSV datetime 列一致（UTC）

WS_MAX_OBS = 30.0
WS_MAX_CFD = 20.0

METRIC_START = "2025-09-01 00:00:00"
METRIC_END = "2025-09-05 23:00:00"
METRIC_DATETIMES = pd.date_range(METRIC_START, METRIC_END, freq="h")
TIME_LABELS = {
    dt.strftime("%Y-%m-%d %H:%M:%S"): f"{dt.day:02d}_{dt.strftime('%H00')} UTC"
    for dt in METRIC_DATETIMES
}


def metric_utc_days() -> list[int]:
    return sorted({dt.day for dt in METRIC_DATETIMES})


def _has_any_hour(day: int, hours: list[int]) -> bool:
    keys = set(TIME_LABELS)
    return any(f"2025-09-{day:02d} {h:02d}:00:00" in keys for h in hours)

COLOR_OBS = "#1a1a2e"
COLOR_WRF = "#e07b39"
COLOR_CFD = "#2196a5"


def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def load_and_preprocess(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['datetime'])
    df['ws_cfd'] = np.sqrt(df['u_cfd']**2 + df['v_cfd']**2)
    df['wd_obs'] = np.degrees(np.arctan2(-df['u_obs'], -df['v_obs'])) % 360
    df['wd_wrf'] = np.degrees(np.arctan2(-df['u_wrf'], -df['v_wrf'])) % 360
    df['wd_cfd'] = np.degrees(np.arctan2(-df['u_cfd'], -df['v_cfd'])) % 360
    df['time_label'] = df['datetime'].astype(str).map(TIME_LABELS)
    return df


def quality_control(df: pd.DataFrame,
                    ws_max_obs: float = WS_MAX_OBS,
                    ws_max_cfd: float = WS_MAX_CFD) -> pd.DataFrame:
    out = df.copy()
    obs_ok = (out['ws_obs'] <= ws_max_obs) | out['ws_obs'].isna()
    cfd_ok = out['ws_cfd'] <= ws_max_cfd
    out['qc_ok'] = obs_ok & cfd_ok
    return out


def plot_ws_wd_station_profiles(df: pd.DataFrame, out_dir: Path, tz: str = TZ_TAG) -> None:
    """UTC 日历日 × 4 小时窗口；仅生成 CSV 中至少含 1 个时次的窗口。"""
    sites = sorted(df['obtid'].unique())
    n_sites = len(sites)
    height_bins = np.arange(0, 2150, 50)
    out_dir.mkdir(parents=True, exist_ok=True)

    for day in metric_utc_days():
        for h_start in range(0, 24, 4):
            hours = [h_start + i for i in range(4)]
            if not _has_any_hour(day, hours):
                continue
            fig, axes = plt.subplots(
                4, 2 * n_sites,
                figsize=(2.8 * 2 * n_sites, 3.2 * 4),
                constrained_layout=True,
            )

            for i_h, h in enumerate(hours):
                t_raw = f"2025-09-{day:02d} {h:02d}:00:00"
                tl = TIME_LABELS.get(t_raw, t_raw)
                col = 0
                for site in sites:
                    ax_ws = axes[i_h, col]
                    ax_wd = axes[i_h, col + 1]
                    sub = df[(df['obtid'] == site) & (df['time_label'] == tl) & df['qc_ok']].copy()

                    if sub.empty:
                        ax_ws.axis('off')
                        ax_wd.axis('off')
                        col += 2
                        continue

                    sub['H_bin'] = pd.cut(sub['Height'], bins=height_bins)
                    for prefix in ('obs', 'wrf', 'cfd'):
                        sub[f'u_{prefix}'] = -np.sin(np.radians(sub[f'wd_{prefix}']))
                        sub[f'v_{prefix}'] = -np.cos(np.radians(sub[f'wd_{prefix}']))

                    agg = sub.groupby('H_bin', observed=True).agg({
                        'ws_obs': 'mean', 'ws_wrf': 'mean', 'ws_cfd': 'mean',
                        'u_obs': 'mean', 'u_wrf': 'mean', 'u_cfd': 'mean',
                        'v_obs': 'mean', 'v_wrf': 'mean', 'v_cfd': 'mean',
                        'Height': 'mean',
                    }).dropna(subset=['Height']).reset_index()

                    for prefix in ('obs', 'wrf', 'cfd'):
                        agg[f'wd_{prefix}'] = (
                            np.degrees(np.arctan2(-agg[f'u_{prefix}'], -agg[f'v_{prefix}'])) + 360
                        ) % 360

                    std_map = sub.groupby('H_bin', observed=True)['ws_obs'].std().fillna(0)
                    ax_ws.errorbar(
                        agg['ws_obs'], agg['Height'],
                        xerr=agg['H_bin'].map(std_map).values,
                        fmt='o', ms=2.5, color=COLOR_OBS, alpha=0.85,
                        elinewidth=0.6, capsize=1.5,
                    )
                    ax_ws.plot(agg['ws_wrf'], agg['Height'], color=COLOR_WRF, lw=1.5, ls='--')
                    ax_ws.plot(agg['ws_cfd'], agg['Height'], color=COLOR_CFD, lw=1.8, ls='-')
                    ax_ws.set_ylim(0, 2100)
                    ax_ws.set_xlim(left=0)
                    ax_ws.set_title(
                        f"{site} | {tl}" if i_h == 0 else tl,
                        fontsize=9, fontweight='bold',
                    )
                    if i_h == 3:
                        ax_ws.set_xlabel('WS (m s$^{-1}$)', fontsize=9)
                    if col == 0:
                        ax_ws.set_ylabel('Height (m)', fontsize=9)
                    else:
                        ax_ws.set_yticklabels([])

                    ax_wd.scatter(
                        sub['wd_obs'], sub['Height'],
                        s=1, color=COLOR_OBS, alpha=0.15, edgecolors='none',
                    )
                    ax_wd.plot(agg['wd_obs'], agg['Height'], 'o', ms=2.5, color=COLOR_OBS, alpha=0.9, label='LiDAR')
                    ax_wd.plot(agg['wd_wrf'], agg['Height'], color=COLOR_WRF, lw=1.5, ls='--', label='WRF')
                    ax_wd.plot(agg['wd_cfd'], agg['Height'], color=COLOR_CFD, lw=1.8, ls='-', label='OpenFOAM')
                    ax_wd.set_ylim(0, 2100)
                    ax_wd.set_xlim(0, 360)
                    ax_wd.set_xticks([0, 90, 180, 270, 360])
                    if i_h == 3:
                        ax_wd.set_xticklabels(['N', 'E', 'S', 'W', 'N'], fontsize=8)
                    else:
                        ax_wd.set_xticklabels([])
                    ax_wd.set_yticklabels([])
                    if i_h == 3:
                        ax_wd.set_xlabel('WD (°)', fontsize=9)
                    if i_h == 0 and col == 2 * n_sites - 2:
                        ax_wd.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.05, 1.15))

                    col += 2

            fig.suptitle(
                f'Wind Speed & Direction Profiles — 2025-09-{day:02d} '
                f'({h_start:02d}00–{h_start + 3:02d}00 UTC)',
                fontsize=14, fontweight='bold',
            )
            save_path = out_dir / (
                f"fig1_ws_wd_station_09{day:02d}_{h_start:02d}00-{h_start + 3:02d}00_tz-{tz}.png"
            )
            fig.savefig(save_path, dpi=300)
            plt.close(fig)


def main() -> None:
    configure_matplotlib_style()
    df = quality_control(load_and_preprocess(DATA_PATH))
    plot_ws_wd_station_profiles(df, OUTPUT_DIR, tz=TZ_TAG)


if __name__ == "__main__":
    main()
