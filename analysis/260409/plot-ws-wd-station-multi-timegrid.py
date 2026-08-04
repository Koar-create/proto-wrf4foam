#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多站点多时段风速/风向垂直剖面网格可视化脚本（站点 × 6 小时）。
默认：分别输出风速折线网格与风向折线网格（同布局）；可选 --show-wd-arrows
在风速图右侧叠加风向箭头。子图默认行优先标号 (a)–(r)。图例置于图顶。
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.path import Path as MPath
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------- 配置与常量 ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv"
OUTPUT_DIR = REPO_ROOT / "results/ws_wd_station_profile/260409/multi_time_grid"

DEFAULT_SITES = ("GAW103", "GAW104", "GAW111")
ZMAX = 1000.0  # 固定的 zmax
TZ_DISPLAY = "LST"

COLOR_OBS = "#1a1a2e"
COLOR_WRF = "#e07b39"
COLOR_CFD = "#2196a5"

def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 24,
        "axes.labelsize": 26,
        "axes.titlesize": 28,
        "axes.titleweight": "bold",
        "axes.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

# ---------- 数据处理 ----------
def load_and_preprocess(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["ws_cfd"] = np.sqrt(df["u_cfd"] ** 2 + df["v_cfd"] ** 2)
    df["wd_obs"] = np.degrees(np.arctan2(-df["u_obs"], -df["v_obs"])) % 360
    df["wd_wrf"] = np.degrees(np.arctan2(-df["u_wrf"], -df["v_wrf"])) % 360
    df["wd_cfd"] = np.degrees(np.arctan2(-df["u_cfd"], -df["v_cfd"])) % 360
    df["time_label"] = df["datetime"].astype(str).map(TIME_LABELS)
    return df

def quality_control(df: pd.DataFrame, ws_max_obs: float = 30.0, ws_max_cfd: float = 20.0) -> pd.DataFrame:
    out = df.copy()
    obs_ok = (out["ws_obs"] <= ws_max_obs) | out["ws_obs"].isna()
    cfd_ok = out["ws_cfd"] <= ws_max_cfd
    out["qc_ok"] = obs_ok & cfd_ok
    return out

METRIC_START = "2025-09-01 00:00:00"
METRIC_END = "2025-09-05 23:00:00"
METRIC_DATETIMES = pd.date_range(METRIC_START, METRIC_END, freq="h")
TIME_LABELS = {
    dt.strftime("%Y-%m-%d %H:%M:%S"): f"{dt.day:02d}_{dt.strftime('%H00')} UTC"
    for dt in METRIC_DATETIMES
}

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
    sub["H_bin"] = pd.cut(sub["Height"], bins=np.arange(0, 2150, 50))
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

# ---------- 风向箭头样式 ----------
_arrow_verts = [
    (0.0, 0.5),
    (-0.35, -0.4),
    (0.0, -0.05),
    (0.35, -0.4),
    (0.0, 0.5),
]
_arrow_codes = [
    MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY,
]
_ARROW = MPath(_arrow_verts, _arrow_codes)

_MONTH_ABBR = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sept", "Oct", "Nov", "Dec",
)


def _format_date_label(ts: pd.Timestamp) -> str:
    return f"{_MONTH_ABBR[ts.month - 1]} {ts.day:02d}"


def _panel_letter(row: int, col: int, ncols: int = 6) -> str:
    """行优先子图标号：a, b, ...（3×6 → a–r）。"""
    return chr(ord("a") + row * ncols + col)


def _add_panel_label(ax, row: int, col: int, ncols: int = 6) -> None:
    ax.text(
        0.04, 0.97, f"({_panel_letter(row, col, ncols)})",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=20, fontweight="bold", zorder=10,
    )


def _ws_data_xmax(agg: pd.DataFrame) -> float:
    vals = np.concatenate([
        agg["ws_obs"].to_numpy(dtype=float),
        agg["ws_wrf"].to_numpy(dtype=float),
        agg["ws_cfd"].to_numpy(dtype=float),
    ])
    finite = vals[np.isfinite(vals)]
    return float(np.nanmax(finite)) if finite.size else 1.0


def _ws_tick_step(xmax: float) -> float:
    if xmax <= 8:
        return 1.0
    if xmax <= 16:
        return 2.0
    return 5.0


def _nice_ws_axis(data_xmax: float) -> tuple[float, float]:
    """按数据上取整 xmax，返回 (xmax, tick_step)，刻度为 1/2/5 等规整步长。"""
    x = max(float(data_xmax), 1e-3) * 1.05
    if x <= 8:
        xmax = max(float(np.ceil(x)), 1.0)
    elif x <= 16:
        xmax = float(np.ceil(x / 2.0) * 2.0)
    else:
        xmax = float(np.ceil(x / 5.0) * 5.0)
    return xmax, _ws_tick_step(xmax)


# 全图各有数据子图的 nice xmax，若 max/min <= 该比值则统一为最大值（3 与 4 统一；4 与 10 保持独立）
_XMAX_UNIFY_RATIO = 1.5


def _resolve_shared_ws_xmax(nice_xmaxs: list[float]) -> float | None:
    if not nice_xmaxs:
        return None
    lo, hi = min(nice_xmaxs), max(nice_xmaxs)
    if hi <= lo * _XMAX_UNIFY_RATIO:
        return hi
    return None


def _apply_ws_xlim(
    ax,
    data_xmax: float,
    *,
    show_wd_arrows: bool,
    agg: pd.DataFrame,
    row: int,
    forced_xmax: float | None = None,
) -> None:
    if show_wd_arrows:
        max_ws = max(forced_xmax if forced_xmax is not None else data_xmax, 10.0)
        _draw_wd_arrows(ax, agg, row, max_ws)
    else:
        if forced_xmax is not None:
            xmax, step = forced_xmax, _ws_tick_step(forced_xmax)
        else:
            xmax, step = _nice_ws_axis(data_xmax)
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_locator(MultipleLocator(step))


# ---------- 绘图核心函数 ----------
def _prepare_grid_window(start_dt: str) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    start = pd.Timestamp(start_dt)
    end = start + pd.Timedelta(hours=5)  # 包含起点的 6 个小时
    times = pd.date_range(start, end, freq="h")
    return start, times


def _finish_grid_figure(
    fig: plt.Figure,
    start: pd.Timestamp,
    legend_handles: list,
    legend_labels: list,
    xlabel: str,
    save_path: Path,
) -> Path:
    start_lst = start + pd.Timedelta(hours=8)
    fig.text(
        0.02, 0.98, _format_date_label(start_lst),
        ha="left", va="top", fontsize=32, fontweight="bold",
        transform=fig.transFigure,
    )
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=3,
            fontsize=32,
            framealpha=1.0,
        )
    fig.text(0.5, 0.02, xlabel, ha="center", va="center", fontsize=36, fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def _draw_wd_arrows(ax, agg: pd.DataFrame, row: int, max_ws: float) -> None:
    arrow_start = max_ws * 1.15
    ax.set_xlim(0, max_ws * 1.7)
    ax.axvspan(arrow_start, max_ws * 1.7, color="#f0f0f0", alpha=0.5, zorder=0)

    blend_trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for wd_col, x_pos, col_c, title in [
        ("obs", 0.76, COLOR_OBS, "Obs"),
        ("wrf", 0.86, COLOR_WRF, "WRF"),
        ("cfd", 0.96, COLOR_CFD, "CFD"),
    ]:
        for _, r_data in agg.iterrows():
            wd_raw = r_data[f"wd_{wd_col}"]
            if pd.isna(wd_raw):
                continue
            wd_snap = round(wd_raw / 22.5) * 22.5 % 360
            rot = mtransforms.Affine2D().rotate_deg(-wd_snap - 180)
            marker = MarkerStyle(_ARROW, transform=rot)
            ax.plot(
                x_pos,
                r_data["mean_h"],
                marker=marker,
                color=col_c,
                ms=16,
                lw=0,
                transform=blend_trans,
                clip_on=False,
            )
        if row == 0:
            ax.text(
                x_pos, 1.02, title,
                ha="left", va="bottom", rotation=45, fontsize=24, color=col_c,
                fontweight="bold",
                transform=mtransforms.blended_transform_factory(ax.transAxes, ax.transAxes),
                clip_on=False,
            )


def plot_multi_time_grid_ws(
    df: pd.DataFrame,
    sites: list[str],
    start_dt: str,
    show_wd_arrows: bool = False,
) -> Path:
    """风速廓线 m×n 网格；可选在右侧叠加风向箭头。"""
    start, times = _prepare_grid_window(start_dt)
    fig, axes = plt.subplots(len(sites), 6, figsize=(32, 18), sharex=False, sharey=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.15, top=0.90, bottom=0.08)

    # 先收集各子图数据与 nice xmax，再决定是否全图统一
    cells: dict[tuple[int, int], dict] = {}
    nice_xmaxs: list[float] = []
    for row, site in enumerate(sites):
        for col, cur_dt in enumerate(times):
            tl_utc = TIME_LABELS.get(cur_dt.strftime("%Y-%m-%d %H:%M:%S"))
            if tl_utc is None:
                cells[(row, col)] = {"kind": "oob"}
                continue
            sub = df[(df["obtid"] == site) & (df["time_label"] == tl_utc) & df["qc_ok"]].copy()
            agg = _aggregate_station_profile(sub) if not sub.empty else pd.DataFrame()
            if not agg.empty:
                agg = agg[(agg["mean_h"] >= 0) & (agg["mean_h"] <= ZMAX)].copy()
            if agg.empty:
                cells[(row, col)] = {"kind": "nodata"}
                continue
            data_xmax = _ws_data_xmax(agg)
            nice_xmax, _ = _nice_ws_axis(data_xmax)
            cells[(row, col)] = {
                "kind": "data",
                "agg": agg,
                "data_xmax": data_xmax,
                "nice_xmax": nice_xmax,
            }
            nice_xmaxs.append(nice_xmax)

    shared_xmax = None if show_wd_arrows else _resolve_shared_ws_xmax(nice_xmaxs)

    legend_handles: list = []
    legend_labels: list = []

    for row, site in enumerate(sites):
        for col, cur_dt in enumerate(times):
            ax = axes[row, col]
            cell = cells[(row, col)]

            cur_lst = cur_dt + pd.Timedelta(hours=8)
            if row == 0:
                ax.set_title(cur_lst.strftime("%H:%M LST"), fontsize=22, fontweight="bold", loc="center")
            _add_panel_label(ax, row, col)

            if cell["kind"] == "oob":
                ax.text(0.5, 0.5, "Out of bounds", ha="center", va="center", transform=ax.transAxes, fontsize=18)
            elif cell["kind"] == "nodata":
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=16, transform=ax.transAxes)
                ax.set_xlim(0, 15)
            else:
                agg = cell["agg"]
                l1, = ax.plot(agg["ws_obs"], agg["mean_h"], color=COLOR_OBS, lw=2.5, marker="o", ms=6.0, label="LiDAR")
                l2, = ax.plot(agg["ws_wrf"], agg["mean_h"], color=COLOR_WRF, lw=2.5, ls="-", label="WRF")
                l3, = ax.plot(agg["ws_cfd"], agg["mean_h"], color=COLOR_CFD, lw=3.0, ls="-", label="WRF-to-OpenFOAM")
                if not legend_handles:
                    legend_handles = [l1, l2, l3]
                    legend_labels = ["LiDAR", "WRF", "WRF-to-OpenFOAM"]

                forced = shared_xmax if shared_xmax is not None else (
                    None if show_wd_arrows else cell["nice_xmax"]
                )
                _apply_ws_xlim(
                    ax, cell["data_xmax"],
                    show_wd_arrows=show_wd_arrows, agg=agg, row=row,
                    forced_xmax=forced,
                )

            for h_line in (300, 1000):
                ax.axhline(h_line, color="0.6", lw=1.2, ls=":", zorder=0)
            ax.set_ylim(0, ZMAX)
            if col == 0:
                ax.set_ylabel(f"{site}\nHeight (m)", fontsize=32, fontweight="bold")

    dt_tag = start.strftime("%Y%m%d_%H%M")
    out_dir = OUTPUT_DIR / dt_tag
    if show_wd_arrows:
        # 风速 + 右侧风向箭头，文件名保留 ws_wd
        save_path = out_dir / f"ws_wd_grid_6h_all_stations_{dt_tag}_wdarrows.png"
    else:
        save_path = out_dir / f"ws_grid_6h_all_stations_{dt_tag}_nowdarrows.png"
    return _finish_grid_figure(
        fig, start, legend_handles, legend_labels,
        r"Wind Speed (m s$^{-1}$)", save_path,
    )


def plot_multi_time_grid_wd_lines(
    df: pd.DataFrame,
    sites: list[str],
    start_dt: str,
) -> Path:
    """风向廓线 m×n 折线网格（参考 station-profile / composite 的 WD 画法）。"""
    start, times = _prepare_grid_window(start_dt)
    fig, axes = plt.subplots(len(sites), 6, figsize=(32, 18), sharex=False, sharey=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.15, top=0.90, bottom=0.08)

    legend_handles: list = []
    legend_labels: list = []

    for row, site in enumerate(sites):
        for col, cur_dt in enumerate(times):
            ax = axes[row, col]

            cur_lst = cur_dt + pd.Timedelta(hours=8)
            if row == 0:
                ax.set_title(cur_lst.strftime("%H:%M LST"), fontsize=22, fontweight="bold", loc="center")
            _add_panel_label(ax, row, col)

            tl_utc = TIME_LABELS.get(cur_dt.strftime("%Y-%m-%d %H:%M:%S"))
            if tl_utc is None:
                ax.text(0.5, 0.5, "Out of bounds", ha="center", va="center", transform=ax.transAxes, fontsize=18)
                ax.set_ylim(0, ZMAX)
                if col == 0:
                    ax.set_ylabel(f"{site}\nHeight (m)", fontsize=32, fontweight="bold")
                continue

            sub = df[(df["obtid"] == site) & (df["time_label"] == tl_utc) & df["qc_ok"]].copy()
            sub_z = sub[(sub["Height"] >= 0) & (sub["Height"] <= ZMAX)].copy() if not sub.empty else sub
            agg = _aggregate_station_profile(sub) if not sub.empty else pd.DataFrame()
            if not agg.empty:
                agg = agg[(agg["mean_h"] >= 0) & (agg["mean_h"] <= ZMAX)].copy()

            if not agg.empty:
                obs_pts = sub_z.dropna(subset=["wd_obs", "Height"])
                if not obs_pts.empty:
                    ax.scatter(
                        obs_pts["wd_obs"], obs_pts["Height"],
                        s=8, color=COLOR_OBS, alpha=0.15, edgecolors="none", zorder=1,
                    )
                l1, = ax.plot(
                    agg["wd_obs"], agg["mean_h"], "o", ms=6.0,
                    color=COLOR_OBS, alpha=0.9, label="LiDAR",
                )
                l2, = ax.plot(
                    agg["wd_wrf"], agg["mean_h"],
                    color=COLOR_WRF, lw=2.5, ls="--", label="WRF",
                )
                l3, = ax.plot(
                    agg["wd_cfd"], agg["mean_h"],
                    color=COLOR_CFD, lw=3.0, ls="-", label="WRF-to-OpenFOAM",
                )
                if not legend_handles:
                    legend_handles = [l1, l2, l3]
                    legend_labels = ["LiDAR", "WRF", "WRF-to-OpenFOAM"]
            else:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=16, transform=ax.transAxes)

            for h_line in (300, 1000):
                ax.axhline(h_line, color="0.6", lw=1.2, ls=":", zorder=0)

            ax.set_xlim(0, 360)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set_xticklabels(["N", "E", "S", "W", "N"])
            ax.set_ylim(0, ZMAX)
            if col == 0:
                ax.set_ylabel(f"{site}\nHeight (m)", fontsize=32, fontweight="bold")

    dt_tag = start.strftime("%Y%m%d_%H%M")
    save_path = OUTPUT_DIR / dt_tag / f"wd_grid_6h_all_stations_{dt_tag}_wdlines.png"
    return _finish_grid_figure(
        fig, start, legend_handles, legend_labels,
        r"Wind Direction (°)", save_path,
    )


def plot_multi_time_grid_all_sites(
    df: pd.DataFrame,
    sites: list[str],
    start_dt: str,
    show_wd_arrows: bool = False,
) -> list[Path]:
    """
    默认（无矢量）：输出风速折线网格 + 风向折线网格。
    --show-wd-arrows：仅输出右侧带风向箭头的风速网格。
    """
    saved = [plot_multi_time_grid_ws(df, sites, start_dt, show_wd_arrows=show_wd_arrows)]
    if not show_wd_arrows:
        saved.append(plot_multi_time_grid_wd_lines(df, sites, start_dt))
    return saved


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot multi-station x 6-hour WS/WD profile grids. "
            "Outputs go under multi_time_grid/YYYYmmdd_HHMM/. "
            "Default: two figures - ws_grid_*_nowdarrows.png (wind speed) "
            "and wd_grid_*_wdlines.png (wind direction lines). "
            "With --show-wd-arrows: one combined figure "
            "ws_wd_grid_*_wdarrows.png (WS profiles + WD arrows on the right)."
        ),
    )
    p.add_argument(
        "--start",
        required=True,
        help="Start UTC time, e.g. '2025-09-01 11:00:00'. Will plot 6 hours from this time.",
    )
    p.add_argument(
        "--sites",
        nargs="+",
        default=list(DEFAULT_SITES),
        help="Station IDs (rows). Default: GAW103 GAW104 GAW111.",
    )
    p.add_argument(
        "--show-wd-arrows",
        action="store_true",
        default=False,
        help=(
            "Overlay wind-direction arrows on the right of WS panels (widens xmax). "
            "Default: off; instead save separate WS and WD line-profile grids."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    configure_matplotlib_style()
    df = quality_control(load_and_preprocess(DATA_PATH))
    saved = plot_multi_time_grid_all_sites(
        df, args.sites, args.start.strip(), show_wd_arrows=args.show_wd_arrows,
    )
    for path in saved:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
