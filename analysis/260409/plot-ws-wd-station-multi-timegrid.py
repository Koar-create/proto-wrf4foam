#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多站点多时段风速/风向垂直剖面网格可视化脚本（3行 x 6列）。
用于展示连续6个小时的三个站点的廓线变化。
图例统一放置在图的最上方。
风向箭头被放大并优化显示，字体显著增大，确保在演示文稿中清晰可见。
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.path import Path as MPath
from matplotlib.markers import MarkerStyle
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

# ---------- 绘图核心函数 ----------
def plot_multi_time_grid_all_sites(df: pd.DataFrame, sites: list[str], start_dt: str) -> Path:
    start = pd.Timestamp(start_dt)
    end = start + pd.Timedelta(hours=5)  # 包含起点的 6 个小时
    times = pd.date_range(start, end, freq="h")

    fig, axes = plt.subplots(len(sites), 6, figsize=(32, 18), sharex=False, sharey=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.15, top=0.90, bottom=0.08)

    legend_handles = []
    legend_labels = []

    for row, site in enumerate(sites):
        for col, cur_dt in enumerate(times):
            ax = axes[row, col]

            cur_lst = cur_dt + pd.Timedelta(hours=8)
            if row == 0:
                ax.set_title(cur_lst.strftime("%H:%M LST"), fontsize=22, fontweight="bold", loc="left")

            tl_utc = TIME_LABELS.get(cur_dt.strftime("%Y-%m-%d %H:%M:%S"))
            if tl_utc is None:
                ax.text(0.5, 0.5, "Out of bounds", ha="center", va="center", transform=ax.transAxes, fontsize=18)
                continue

            sub = df[(df["obtid"] == site) & (df["time_label"] == tl_utc) & df["qc_ok"]].copy()
            agg = _aggregate_station_profile(sub) if not sub.empty else pd.DataFrame()

            if not agg.empty:
                agg = agg[(agg["mean_h"] >= 0) & (agg["mean_h"] <= ZMAX)].copy()

            if not agg.empty:
                l1, = ax.plot(agg["ws_obs"], agg["mean_h"], color=COLOR_OBS, lw=2.5, marker="o", ms=6.0, label="LiDAR")
                l2, = ax.plot(agg["ws_wrf"], agg["mean_h"], color=COLOR_WRF, lw=2.5, ls="-", label="WRF")
                l3, = ax.plot(agg["ws_cfd"], agg["mean_h"], color=COLOR_CFD, lw=3.0, ls="-", label="OpenFOAM")

                if not legend_handles:
                    legend_handles = [l1, l2, l3]
                    legend_labels = ["LiDAR", "WRF", "OpenFOAM"]

                # 为了放下风向箭头，扩展X轴，并在右侧绘制一个淡灰色背景带
                ax.relim()
                ax.autoscale_view(scalex=True, scaley=False)
                curr_xlim = ax.get_xlim()
                max_ws = max(curr_xlim[1], 10.0)
                
                # 分配箭头区域：max_ws 的 1.1倍到 1.6倍区域
                arrow_start = max_ws * 1.15
                ax.set_xlim(0, max_ws * 1.7)

                # 绘制灰色背景作为“风向颜色条”区域底色
                ax.axvspan(arrow_start, max_ws * 1.7, color="#f0f0f0", alpha=0.5, zorder=0)

                x_obs = arrow_start + (max_ws * 0.1)
                x_wrf = arrow_start + (max_ws * 0.25)
                x_cfd = arrow_start + (max_ws * 0.4)

                trans = ax.get_yaxis_transform()
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
                            ms=16,  # 显著放大箭头
                            lw=0,
                            transform=blend_trans,
                            clip_on=False,
                        )
                    # 顶部标签
                    if row == 0:
                        ax.text(
                            x_pos, 1.02, title,
                            ha="left", va="bottom", rotation=45, fontsize=24, color=col_c,
                            fontweight="bold", transform=mtransforms.blended_transform_factory(ax.transAxes, ax.transAxes),
                            clip_on=False,
                        )

            else:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=16, transform=ax.transAxes)
                ax.set_xlim(0, 15)

            for h_line in (300, 1000):
                ax.axhline(h_line, color="0.6", lw=1.2, ls=":", zorder=0)

            ax.set_ylim(0, ZMAX)

            if col == 0:
                ax.set_ylabel(f"{site}\nHeight (m)", fontsize=32, fontweight="bold")

    # 统一图例
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

    # 底部统一的X轴标签
    fig.text(0.5, 0.02, r"Wind Speed (m s$^{-1}$)", ha="center", va="center", fontsize=36, fontweight="bold")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dt_tag = start.strftime("%Y%m%d_%H%M")
    save_path = OUTPUT_DIR / f"ws_wd_grid_6h_all_stations_{dt_tag}.png"
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot 3x6 grid for 6 hours WS/WD profiles for multiple stations.")
    p.add_argument(
        "--start",
        required=True,
        help="Start UTC time, e.g. '2025-09-01 11:00:00'. Will plot 6 hours from this time.",
    )
    p.add_argument(
        "--sites",
        nargs="+",
        default=list(DEFAULT_SITES),
        help="Station IDs to plot. They will be combined into a single 3x6 grid image.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    configure_matplotlib_style()
    df = quality_control(load_and_preprocess(DATA_PATH))
    saved = plot_multi_time_grid_all_sites(df, args.sites, args.start.strip())
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
