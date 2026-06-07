"""单时刻四站 composite 的 WS / WD 垂直廓线（左 WS | 右 WD，1×2 panel）。"""

import warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv"
OUTPUT_DIR = REPO_ROOT / "results/ws_wd_composite_profile/260409"

WS_MAX_OBS = 30.0
WS_MAX_CFD = 20.0
HEIGHT_BINS = np.arange(0, 2150, 50)

TIME_LABELS = {
    f"2025-09-{d:02d} {h:02d}:00:00": f"{d:02d}_{h:02d}00 UTC"
    for d in range(1, 4) for h in range(24)
}

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
    df['ws_cfd'] = np.sqrt(df['u_cfd'] ** 2 + df['v_cfd'] ** 2)
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


def _format_time_label_for_display(t_raw: str, tz: str) -> str:
    tz = tz.lower()
    dt = pd.Timestamp(t_raw)
    if tz == "lst":
        dt = dt + pd.Timedelta(hours=8)
        suffix = "LST"
    else:
        suffix = "UTC"
    return f"{dt.strftime('%Y-%m-%d %d_%H00')} {suffix}"


def _uv_from_wd(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    wd = out[f'wd_{prefix}']
    out[f'u_{prefix}'] = -np.sin(np.radians(wd))
    out[f'v_{prefix}'] = -np.cos(np.radians(wd))
    return out


def _wd_from_uv(agg: pd.DataFrame, prefix: str) -> pd.Series:
    return (np.degrees(np.arctan2(-agg[f'u_{prefix}'], -agg[f'v_{prefix}'])) + 360) % 360


def _aggregate_composite(sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """四站 pooled：按高度 bin 聚合 WS 与 WD（WD 经 u/v 分量平均）。"""
    sub = sub.copy()
    sub['H_bin'] = pd.cut(sub['Height'], bins=HEIGHT_BINS)

    sub_obs = sub.dropna(subset=['ws_obs']).copy()
    agg_obs = pd.DataFrame()
    if not sub_obs.empty:
        agg_obs = sub_obs.groupby('H_bin', observed=True).agg(
            mean_ws=('ws_obs', 'mean'),
            std_ws=('ws_obs', 'std'),
            mean_h=('Height', 'mean'),
        ).dropna(subset=['mean_h']).reset_index()

    for prefix in ('obs', 'wrf', 'cfd'):
        sub = _uv_from_wd(sub, prefix)

    agg_wd = sub.groupby('H_bin', observed=True).agg(
        mean_h=('Height', 'mean'),
        u_obs=('u_obs', 'mean'), v_obs=('v_obs', 'mean'),
        u_wrf=('u_wrf', 'mean'), v_wrf=('v_wrf', 'mean'),
        u_cfd=('u_cfd', 'mean'), v_cfd=('v_cfd', 'mean'),
        ws_wrf=('ws_wrf', 'mean'), ws_cfd=('ws_cfd', 'mean'),
    ).dropna(subset=['mean_h']).reset_index()

    for prefix in ('obs', 'wrf', 'cfd'):
        agg_wd[f'wd_{prefix}'] = _wd_from_uv(agg_wd, prefix)

    return sub, agg_obs, agg_wd


def plot_ws_wd_composite_profile(
    df: pd.DataFrame,
    datetime_utc: str,
    out_dir: Path,
    *,
    tz: str = "utc",
    zmax: float = 1000.0,
) -> Path:
    tz = tz.lower()
    if tz not in {"utc", "lst"}:
        raise ValueError("tz must be one of: utc, lst")

    tl_utc = TIME_LABELS.get(datetime_utc)
    if tl_utc is None:
        raise ValueError(
            f"datetime must be hourly UTC in 2025-09-01..03, got {datetime_utc!r}. "
            f"Example: 2025-09-03 15:00:00"
        )

    sub = df[(df['time_label'] == tl_utc) & df['qc_ok']].copy()
    sub_raw, agg_obs, agg_wd = _aggregate_composite(sub)

    fig, (ax_ws, ax_wd) = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
    tl_disp = _format_time_label_for_display(datetime_utc, tz=tz)
    n_sites = sub['obtid'].nunique() if not sub.empty else 0

    if not agg_obs.empty:
        ax_ws.errorbar(
            agg_obs['mean_ws'], agg_obs['mean_h'],
            xerr=agg_obs['std_ws'].fillna(0),
            fmt='o', ms=4, color=COLOR_OBS, alpha=0.85,
            elinewidth=0.8, capsize=2, zorder=5,
        )
    if not agg_wd.empty:
        ax_ws.plot(agg_wd['ws_wrf'], agg_wd['mean_h'], color=COLOR_WRF, lw=2.0, ls='--')
        ax_ws.plot(agg_wd['ws_cfd'], agg_wd['mean_h'], color=COLOR_CFD, lw=2.0, ls='-')

    for h_line in (300, 1000):
        ax_ws.axhline(h_line, color='0.6', lw=0.8, ls=':', zorder=0)
        ax_wd.axhline(h_line, color='0.6', lw=0.8, ls=':', zorder=0)

    ax_ws.set_ylim(0, zmax)
    ax_ws.set_xlim(left=0)
    ax_ws.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax_ws.set_xlabel(r'Wind Speed (m s$^{-1}$)', fontsize=11)
    ax_ws.set_ylabel('Height (m)', fontsize=11)
    ax_ws.set_title('Wind Speed (4-station composite)', fontweight='bold')

    if not sub_raw.empty:
        obs_pts = sub_raw.dropna(subset=['wd_obs', 'Height'])
        ax_wd.scatter(
            obs_pts['wd_obs'], obs_pts['Height'],
            s=2, color=COLOR_OBS, alpha=0.12, edgecolors='none', zorder=1,
        )
    if not agg_wd.empty:
        ax_wd.plot(agg_wd['wd_obs'], agg_wd['mean_h'], 'o', ms=4,
                   color=COLOR_OBS, alpha=0.9, label='LiDAR')
        ax_wd.plot(agg_wd['wd_wrf'], agg_wd['mean_h'],
                   color=COLOR_WRF, lw=2.0, ls='--', label='WRF')
        ax_wd.plot(agg_wd['wd_cfd'], agg_wd['mean_h'],
                   color=COLOR_CFD, lw=2.0, ls='-', label='OpenFOAM')

    ax_wd.set_ylim(0, zmax)
    ax_wd.set_xlim(0, 360)
    ax_wd.set_xticks([0, 90, 180, 270, 360])
    ax_wd.set_xticklabels(['N', 'E', 'S', 'W', 'N'], fontsize=9)
    ax_wd.set_xlabel('Wind Direction (°)', fontsize=11)
    ax_wd.set_yticklabels([])
    ax_wd.set_title('Wind Direction (4-station composite)', fontweight='bold')
    ax_wd.legend(loc='upper right', fontsize=10, framealpha=0.9)

    legend_handles = [
        Line2D([0], [0], marker='o', ms=5, color=COLOR_OBS, linestyle='none', label='LiDAR (obs)'),
        Line2D([0], [0], color=COLOR_WRF, lw=2, ls='--', label='WRF'),
        Line2D([0], [0], color=COLOR_CFD, lw=2, ls='-', label='OpenFOAM'),
    ]
    ax_ws.legend(handles=legend_handles, fontsize=10, loc='upper right', framealpha=0.9)

    dt_tag = pd.Timestamp(datetime_utc).strftime('%Y%m%d_%H%M')
    fig.suptitle(
        f'Composite WS & WD Profiles — {tl_disp}  (n={n_sites} sites)',
        fontsize=14, fontweight='bold',
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"ws_wd_composite_{dt_tag}_z{int(zmax)}m_tz-{tz}.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot single-time WS/WD composite profiles (4-station pooled, 1×2 panels).",
    )
    p.add_argument(
        "--datetime",
        required=True,
        help='UTC hour string, e.g. "2025-09-03 15:00:00" (must exist in merged CSV).',
    )
    p.add_argument("--tz", choices=["utc", "lst"], default="utc",
                   help="Title time zone: utc (default) or lst (UTC+8).")
    p.add_argument("--zmax", type=float, default=1000.0,
                   help="Upper limit of y-axis (height, m). Default: 1000.")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Output directory. Default: {OUTPUT_DIR}")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    configure_matplotlib_style()
    df = quality_control(load_and_preprocess(DATA_PATH))
    save_path = plot_ws_wd_composite_profile(
        df,
        args.datetime.strip(),
        args.out_dir,
        tz=args.tz,
        zmax=args.zmax,
    )
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
