"""单时刻三站点 WS/WD 垂直廓线（每站一张图，无额外子图）。"""
import argparse
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv"
OUTPUT_DIR = REPO_ROOT / "results/ws_wd_station_profile/260409/single_time"
WS_MAX_OBS = 30.0
WS_MAX_CFD = 20.0
HEIGHT_BINS = np.arange(0, 2150, 50)
DEFAULT_SITES = ("GAW103", "GAW104", "GAW111")
METRIC_START = "2025-09-01 00:00:00"
METRIC_END = "2025-09-05 23:00:00"
METRIC_DATETIMES = pd.date_range(METRIC_START, METRIC_END, freq="h")
TIME_LABELS = {
    dt.strftime("%Y-%m-%d %H:%M:%S"): f"{dt.day:02d}_{dt.strftime('%H00')} UTC"
    for dt in METRIC_DATETIMES
}
COLOR_OBS = "#1a1a2e"
COLOR_WRF = "#e07b39"
COLOR_CFD = "#2196a5"
def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
def load_and_preprocess(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["ws_cfd"] = np.sqrt(df["u_cfd"] ** 2 + df["v_cfd"] ** 2)
    df["wd_obs"] = np.degrees(np.arctan2(-df["u_obs"], -df["v_obs"])) % 360
    df["wd_wrf"] = np.degrees(np.arctan2(-df["u_wrf"], -df["v_wrf"])) % 360
    df["wd_cfd"] = np.degrees(np.arctan2(-df["u_cfd"], -df["v_cfd"])) % 360
    df["time_label"] = df["datetime"].astype(str).map(TIME_LABELS)
    return df
def quality_control(
    df: pd.DataFrame,
    ws_max_obs: float = WS_MAX_OBS,
    ws_max_cfd: float = WS_MAX_CFD,
) -> pd.DataFrame:
    out = df.copy()
    obs_ok = (out["ws_obs"] <= ws_max_obs) | out["ws_obs"].isna()
    cfd_ok = out["ws_cfd"] <= ws_max_cfd
    out["qc_ok"] = obs_ok & cfd_ok
    return out
def _format_time_label_for_display(t_raw: str, tz: str) -> str:
    dt = pd.Timestamp(t_raw)
    if tz.lower() == "lst":
        return f"{(dt + pd.Timedelta(hours=8)).strftime('%Y-%m-%d %H:%M')} (LST)"
    return f"{dt.strftime('%Y-%m-%d %H:%M')} (UTC)"
def _uv_from_wd(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    wd = out[f"wd_{prefix}"]
    out[f"u_{prefix}"] = -np.sin(np.radians(wd))
    out[f"v_{prefix}"] = -np.cos(np.radians(wd))
    return out
def _wd_from_uv(agg: pd.DataFrame, prefix: str) -> pd.Series:
    return (np.degrees(np.arctan2(-agg[f"u_{prefix}"], -agg[f"v_{prefix}"])) + 360) % 360
def _aggregate_station_profile(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()
    sub["H_bin"] = pd.cut(sub["Height"], bins=HEIGHT_BINS)
    for prefix in ("obs", "wrf", "cfd"):
        sub = _uv_from_wd(sub, prefix)
    agg = sub.groupby("H_bin", observed=True).agg(
        mean_h=("Height", "mean"),
        ws_obs=("ws_obs", "mean"),
        ws_wrf=("ws_wrf", "mean"),
        ws_cfd=("ws_cfd", "mean"),
        u_obs=("u_obs", "mean"),
        v_obs=("v_obs", "mean"),
        u_wrf=("u_wrf", "mean"),
        v_wrf=("v_wrf", "mean"),
        u_cfd=("u_cfd", "mean"),
        v_cfd=("v_cfd", "mean"),
    ).dropna(subset=["mean_h"]).reset_index()
    for prefix in ("obs", "wrf", "cfd"):
        agg[f"wd_{prefix}"] = _wd_from_uv(agg, prefix)
    return agg
def plot_station_single_profile(
    df: pd.DataFrame,
    datetime_utc: str,
    site: str,
    out_dir: Path,
    *,
    tz: str = "utc",
    zmax: float = 1000.0,
    show_title_legend: bool = True,
) -> Path:
    tz = tz.lower()
    if tz not in {"utc", "lst"}:
        raise ValueError("tz must be one of: utc, lst")
    tl_utc = TIME_LABELS.get(datetime_utc)
    if tl_utc is None:
        raise ValueError(
            f"datetime must be an hourly UTC time from {METRIC_START} to {METRIC_END}, "
            f"got {datetime_utc!r}."
        )
    sub = df[(df["obtid"] == site) & (df["time_label"] == tl_utc) & df["qc_ok"]].copy()
    agg = _aggregate_station_profile(sub) if not sub.empty else pd.DataFrame()
    if not agg.empty:
        agg = agg[(agg["mean_h"] >= 0) & (agg["mean_h"] <= zmax)].copy()
    fig, ax_ws = plt.subplots(figsize=(4.8, 8.4))
    fig.subplots_adjust(top=0.78 if show_title_legend else 0.92)
    if show_title_legend:
        tl_disp = _format_time_label_for_display(datetime_utc, tz)
        fig.suptitle(tl_disp, y=0.9, fontweight="bold")
    # Plot wind speed profiles (foreground)
    if not agg.empty:
        ax_ws.plot(agg["ws_obs"], agg["mean_h"], color=COLOR_OBS, lw=1.8, marker="o", ms=3.0, label="LiDAR WS")
        ax_ws.plot(agg["ws_wrf"], agg["mean_h"], color=COLOR_WRF, lw=1.8, ls="-", label="WRF WS")
        ax_ws.plot(agg["ws_cfd"], agg["mean_h"], color=COLOR_CFD, lw=2.0, ls="-", label="OpenFOAM WS")
        # Wind direction as rotated arrow markers (16-point compass)
        from matplotlib.path import Path as MPath
        from matplotlib.markers import MarkerStyle
        import matplotlib.transforms as mtransforms

        # Chevron arrow pointing up (north = 0°), will be rotated per bin
        _arrow_verts = [
            (0.0,  0.5),    # tip
            (-0.35, -0.4),  # left wing
            (0.0, -0.05),   # notch
            (0.35, -0.4),   # right wing
            (0.0,  0.5),    # close
        ]
        _arrow_codes = [
            MPath.MOVETO, MPath.LINETO, MPath.LINETO,
            MPath.LINETO, MPath.CLOSEPOLY,
        ]
        _ARROW = MPath(_arrow_verts, _arrow_codes)

        # Extend x-axis to make room for wind direction arrows on the right
        ax_ws.relim()
        ax_ws.autoscale_view(scalex=True, scaley=False)
        current_xlim = ax_ws.get_xlim()
        ax_ws.set_xlim(0, max(current_xlim[1], 1.0) * 1.45)

        trans = ax_ws.get_yaxis_transform()
        for wd_col, x_pos, col, title in [
            ("obs", 0.82, COLOR_OBS, "Obs"),
            ("wrf", 0.89, COLOR_WRF, "WRF"),
            ("cfd", 0.96, COLOR_CFD, "CFD"),
        ]:
            for _, row in agg.iterrows():
                wd_raw = row[f"wd_{wd_col}"]
                if pd.isna(wd_raw):
                    continue
                # Snap to 16-point compass (22.5° steps)
                wd_snap = round(wd_raw / 22.5) * 22.5 % 360
                # Rotate: meteorological WD is "from" direction (CW from N).
                # Arrow points downwind (TO direction = WD + 180).
                # matplotlib rotates CCW, so angle = -(WD + 180).
                rot = mtransforms.Affine2D().rotate_deg(-wd_snap - 180)
                marker = MarkerStyle(_ARROW, transform=rot)
                ax_ws.plot(
                    x_pos, row["mean_h"],
                    marker=marker, color=col, ms=9, lw=0,
                    transform=trans, clip_on=False,
                )
            # Header text for each column
            ax_ws.text(
                x_pos, 1.01, title,
                ha="left", va="bottom", fontsize=9, color=col,
                fontweight="bold", transform=ax_ws.transAxes,
                clip_on=False, rotation=45,
            )
    else:
        ax_ws.text(
            0.5,
            0.5,
            "No QC-passed data",
            transform=ax_ws.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
    for h_line in (300, 1000):
        ax_ws.axhline(h_line, color="0.6", lw=0.8, ls=":", zorder=0)
    ax_ws.set_ylim(0, zmax)
    if agg.empty:
        ax_ws.set_xlim(left=0)
    ax_ws.set_xlabel(r"Wind Speed (m s$^{-1}$)")
    ax_ws.set_ylabel("Height (m)")
    ax_ws.text(
        0.015,
        0.985,
        site,
        transform=ax_ws.transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )
    if show_title_legend:
        # Combine legend (only WS lines; WD shown via colour bar)
        ws_handles, ws_labels = ax_ws.get_legend_handles_labels()
        ax_ws.legend(
            ws_handles,
            ws_labels,
            loc="lower center",
            bbox_to_anchor=(0.45, 1.05),
            ncol=2,
            fontsize=9,
            framealpha=0.9,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    dt_tag = pd.Timestamp(datetime_utc).strftime("%Y%m%d_%H%M")
    save_path = out_dir / f"ws_wd_station_{site}_{dt_tag}_z{int(zmax)}m_tz-{tz}.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path
def plot_all_station_single_profiles(
    df: pd.DataFrame,
    datetime_utc: str,
    out_dir: Path,
    *,
    sites: tuple[str, ...] = DEFAULT_SITES,
    tz: str = "utc",
    zmax: float = 1000.0,
    show_title_legend: bool = True,
) -> list[Path]:
    return [
        plot_station_single_profile(
            df,
            datetime_utc,
            site,
            out_dir,
            tz=tz,
            zmax=zmax,
            show_title_legend=show_title_legend,
        )
        for site in sites
    ]
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot single-time WS/WD station profiles: one PNG per station, no subplots.",
    )
    p.add_argument(
        "--datetime",
        required=True,
        help='UTC hour string, e.g. "2025-09-03 15:00:00" (must exist in merged CSV).',
    )
    p.add_argument(
        "--tz",
        choices=["utc", "lst"],
        default="utc",
        help="Title time zone: utc (default) or lst (UTC+8).",
    )
    p.add_argument(
        "--zmax",
        type=float,
        default=1000.0,
        help="Upper limit of y-axis (height, m). Default: 1000.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory. Default: {OUTPUT_DIR}",
    )
    p.add_argument(
        "--sites",
        nargs="+",
        default=list(DEFAULT_SITES),
        help="Stations to plot. Default: GAW103 GAW104 GAW111.",
    )
    p.add_argument(
        "--no-title-legend",
        action="store_true",
        help="Do not show the time title or wind-speed legend.",
    )
    return p.parse_args()
def main() -> None:
    args = _parse_args()
    configure_matplotlib_style()
    df = quality_control(load_and_preprocess(DATA_PATH))
    save_paths = plot_all_station_single_profiles(
        df,
        args.datetime.strip(),
        args.out_dir,
        sites=tuple(args.sites),
        tz=args.tz,
        zmax=args.zmax,
        show_title_legend=not args.no_title_legend,
    )
    for save_path in save_paths:
        print(f"Saved: {save_path}")
if __name__ == "__main__":
    main()
