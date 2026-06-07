#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多站点多时段风速垂直剖面网格可视化脚本（3行 x 8列）。
用于展示9月1号-5号的8个LST时刻 (00:00/03:00/06:00/09:00/12:00/15:00/18:00/21:00 LST) 的五日（或四日）平均。
平均值通过error bar体现（参考fig4中的观测线）。
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.path import Path as MPath
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------- 配置与常量 ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv"
OUTPUT_DIR = REPO_ROOT / "results/ws_wd_station_profile/260409/multi_day_avg"

DEFAULT_SITES = ("GAW103", "GAW104", "GAW111")
ZMAX = 1000.0  # 固定的 zmax

COLOR_OBS = "#1a1a2e"
COLOR_WRF = "#e07b39"
COLOR_CFD = "#2196a5"

HEIGHT_BINS = np.arange(0, 2150, 50)

def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 24,
        "axes.labelsize": 26,
        "axes.titlesize": 26,
        "axes.titleweight": "bold",
        "axes.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

# ---------- 数据处理 ----------
METRIC_START = "2025-09-01 00:00:00"
METRIC_END = "2025-09-05 23:00:00"
METRIC_DATETIMES = pd.date_range(METRIC_START, METRIC_END, freq="h")
TIME_LABELS = {
    dt.strftime("%Y-%m-%d %H:%M:%S"): f"{dt.day:02d}_{dt.strftime('%H00')} UTC"
    for dt in METRIC_DATETIMES
}

def load_and_preprocess(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["ws_cfd"] = np.sqrt(df["u_cfd"] ** 2 + df["v_cfd"] ** 2)
    df["time_label"] = df["datetime"].astype(str).map(TIME_LABELS)
    return df

def quality_control(df: pd.DataFrame, ws_max_obs: float = 30.0, ws_max_cfd: float = 20.0) -> pd.DataFrame:
    out = df.copy()
    obs_ok = (out["ws_obs"] <= ws_max_obs) | out["ws_obs"].isna()
    cfd_ok = out["ws_cfd"] <= ws_max_cfd
    out["qc_ok"] = obs_ok & cfd_ok
    return out

def _utc_slots_for_lst_avg(lst_hour: int) -> list[pd.Timestamp]:
    utc_hour = (lst_hour - 8) % 24
    if lst_hour in [0, 3, 6]:
        # 4 days
        days = [1, 2, 3, 4]
    else:
        # 5 days
        days = [1, 2, 3, 4, 5]
    
    return [pd.Timestamp(f"2025-09-{day:02d} {utc_hour:02d}:00:00") for day in days]

def _wd_from_uv(agg: pd.DataFrame, prefix: str) -> pd.Series:
    return (np.degrees(np.arctan2(-agg[f"u_{prefix}"], -agg[f"v_{prefix}"])) + 360) % 360

_arrow_verts = [(0.0, 0.5), (-0.35, -0.4), (0.0, -0.05), (0.35, -0.4), (0.0, 0.5)]
_arrow_codes = [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY]
_ARROW = MPath(_arrow_verts, _arrow_codes)

# ---------- 绘图核心函数 ----------
def plot_multi_day_avg_grid(df: pd.DataFrame, sites: list[str], plot_diff: bool = False) -> Path:
    lst_hours = [0, 3, 6, 9, 12, 15, 18, 21]
    
    # 设置一个合适的画布大小以容纳8列 (根据原先10列为50x18, 这里可以给40x18)
    fig, axes = plt.subplots(len(sites), len(lst_hours), figsize=(40, 18), sharex=False, sharey=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.90, bottom=0.08)

    if plot_diff:
        legend_handles = [
            Line2D([0], [0], color=COLOR_WRF, lw=3.0, ls='--', label='WRF - Obs mean'),
            Line2D([0], [0], color=COLOR_CFD, lw=3.0, ls='-', label='OpenFOAM - Obs mean'),
        ]
    else:
        legend_handles = [
            Line2D([0], [0], marker='o', ms=10, color=COLOR_OBS, linestyle='none', label='LiDAR (obs) mean ± std'),
            Line2D([0], [0], color=COLOR_WRF, lw=3.0, ls='--', label='WRF mean'),
            Line2D([0], [0], color=COLOR_CFD, lw=3.0, ls='-', label='OpenFOAM mean'),
        ]
    
    for row, site in enumerate(sites):
        for col, lst_hour in enumerate(lst_hours):
            ax = axes[row, col]
            
            title = f"{lst_hour:02d}:00 LST"
            
            if row == 0:
                ax.set_title(title, fontsize=24, fontweight="bold", loc="center")
                
            utc_slots = _utc_slots_for_lst_avg(lst_hour)
            utc_labels = []
            for utc_ts in utc_slots:
                t_raw = utc_ts.strftime("%Y-%m-%d %H:%M:%S")
                if t_raw in TIME_LABELS:
                    utc_labels.append(TIME_LABELS[t_raw])
            
            sub = df[(df["obtid"] == site) & df["time_label"].isin(utc_labels) & df["qc_ok"]].copy()
            
            if sub.empty:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=20, transform=ax.transAxes)
                ax.set_xlim(0, 15)
            else:
                sub["H_bin"] = pd.cut(sub["Height"], bins=HEIGHT_BINS)
                
                sub_obs = sub.dropna(subset=["ws_obs"]).copy()
                agg_obs = pd.DataFrame()
                if not sub_obs.empty:
                    agg_obs = sub_obs.groupby("H_bin", observed=True).agg(
                        mean_ws=("ws_obs", "mean"),
                        std_ws=("ws_obs", "std"),
                        u_obs=("u_obs", "mean"),
                        v_obs=("v_obs", "mean"),
                        mean_h=("Height", "mean"),
                    ).dropna(subset=["mean_h"]).reset_index()
                    agg_obs = agg_obs[(agg_obs["mean_h"] >= 0) & (agg_obs["mean_h"] <= ZMAX)].copy()
                    agg_obs["wd_obs"] = _wd_from_uv(agg_obs, "obs")
                    if not plot_diff:
                        ax.errorbar(
                            agg_obs["mean_ws"], agg_obs["mean_h"],
                            xerr=agg_obs["std_ws"].fillna(0),
                            fmt="o", ms=6, color=COLOR_OBS, alpha=0.85,
                            elinewidth=2.0, capsize=3, zorder=5
                        )
                
                if plot_diff:
                    sub["val_wrf"] = sub["ws_wrf"] - sub["ws_obs"]
                    sub["val_cfd"] = sub["ws_cfd"] - sub["ws_obs"]
                else:
                    sub["val_wrf"] = sub["ws_wrf"]
                    sub["val_cfd"] = sub["ws_cfd"]

                agg_model = sub.groupby("H_bin", observed=True).agg(
                    mean_h=("Height", "mean"),
                    mean_wrf=("val_wrf", "mean"),
                    mean_cfd=("val_cfd", "mean"),
                    u_wrf=("u_wrf", "mean"),
                    v_wrf=("v_wrf", "mean"),
                    u_cfd=("u_cfd", "mean"),
                    v_cfd=("v_cfd", "mean"),
                ).dropna(subset=["mean_h"]).reset_index()
                agg_model = agg_model[(agg_model["mean_h"] >= 0) & (agg_model["mean_h"] <= ZMAX)].copy()
                agg_model["wd_wrf"] = _wd_from_uv(agg_model, "wrf")
                agg_model["wd_cfd"] = _wd_from_uv(agg_model, "cfd")
                ax.plot(agg_model["mean_wrf"], agg_model["mean_h"], color=COLOR_WRF, lw=3.0, ls="--")
                ax.plot(agg_model["mean_cfd"], agg_model["mean_h"], color=COLOR_CFD, lw=3.0, ls="-")
                
                if plot_diff:
                    agg_fill = agg_model.dropna(subset=["mean_wrf", "mean_cfd", "mean_h"]).sort_values("mean_h")
                    if len(agg_fill) >= 2:
                        # 插值到 1m 细密网格，消除稀疏数据点导致的漏判和不连贯
                        y_coarse = agg_fill["mean_h"].values
                        x1_coarse = agg_fill["mean_wrf"].values
                        x2_coarse = agg_fill["mean_cfd"].values
                        y_fine = np.arange(y_coarse.min(), y_coarse.max() + 1, 1.0)
                        x1_fine = np.interp(y_fine, y_coarse, x1_coarse)
                        x2_fine = np.interp(y_fine, y_coarse, x2_coarse)

                        same_side = (x1_fine * x2_fine >= 0)
                        diff_side = ~same_side
                        # 纯粹基于绝对偏差大小判断：谁离 0 更近谁更好
                        of_better = np.abs(x2_fine) <= np.abs(x1_fine)
                        wrf_better = ~of_better

                        # 同侧时：绿色=OF更好，红色=WRF更好
                        ax.fill_betweenx(y_fine, x1_fine, x2_fine, where=of_better & same_side,  facecolor='green', alpha=0.3, interpolate=True)
                        ax.fill_betweenx(y_fine, x1_fine, x2_fine, where=wrf_better & same_side, facecolor='red',   alpha=0.3, interpolate=True)
                        # 异侧时：用浅绿/浅红底 + 灰色叠加表达"异侧但仍可判优劣"
                        ax.fill_betweenx(y_fine, x1_fine, x2_fine, where=of_better & diff_side,  facecolor='green', alpha=0.15, interpolate=True)
                        ax.fill_betweenx(y_fine, x1_fine, x2_fine, where=wrf_better & diff_side, facecolor='red',   alpha=0.15, interpolate=True)
                        ax.fill_betweenx(y_fine, x1_fine, x2_fine, where=diff_side,              facecolor='gray',  alpha=0.15, interpolate=True)
                
                ax.relim()
                ax.autoscale_view(scalex=True, scaley=False)
                curr_xlim = ax.get_xlim()
                
                import matplotlib.ticker as ticker
                if plot_diff:
                    max_abs = max(abs(curr_xlim[0]), abs(curr_xlim[1]), 2.0)
                    arrow_start = max_abs * 1.15
                    ax.set_xlim(-max_abs * 1.1, max_abs * 1.7)
                    ax.axvline(0, color="gray", lw=1.5, ls="--", zorder=0)
                    ax.axvspan(arrow_start, max_abs * 1.7, color="#f0f0f0", alpha=0.5, zorder=0)
                else:
                    max_ws = max(curr_xlim[1], 6.0)
                    arrow_start = max_ws * 1.15
                    ax.set_xlim(0, max_ws * 1.7)
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                    ax.axvspan(arrow_start, max_ws * 1.7, color="#f0f0f0", alpha=0.5, zorder=0)

                blend_trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

                for wd_col, x_pos, col_c, title_label in [
                    ("obs", 0.76, COLOR_OBS, "Obs"),
                    ("wrf", 0.86, COLOR_WRF, "WRF"),
                    ("cfd", 0.96, COLOR_CFD, "CFD"),
                ]:
                    curr_agg = agg_obs if wd_col == "obs" else agg_model
                    if curr_agg.empty or "mean_h" not in curr_agg.columns:
                        continue
                    for _, r_data in curr_agg.iterrows():
                        wd_raw = r_data.get(f"wd_{wd_col}")
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

            ax.axhline(300, color="black", lw=2.0, ls="--", alpha=0.4, zorder=0)
            ax.axhline(1000, color="0.6", lw=1.2, ls=":", zorder=0)

            ax.set_ylim(0, ZMAX)

            if col == 0:
                ax.set_ylabel(f"{site}\nHeight (m)", fontsize=32, fontweight="bold")
                
    # 统一图例
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        fontsize=32,
        framealpha=1.0,
    )
    
    # 底部统一的X轴标签
    x_label = r"Wind Speed Difference (m s$^{-1}$)" if plot_diff else r"Wind Speed (m s$^{-1}$)"
    fig.text(0.5, 0.02, x_label, ha="center", va="center", fontsize=36, fontweight="bold")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_diff" if plot_diff else ""
    save_path = OUTPUT_DIR / f"ws_grid_3x8_8times_avg{suffix}.png"
    fig.savefig(save_path)
    plt.close(fig)
    return save_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multi-day average grid")
    parser.add_argument("--diff", action="store_true", help="Plot difference between model and observation")
    args = parser.parse_args()

    configure_matplotlib_style()
    df = quality_control(load_and_preprocess(DATA_PATH))
    saved = plot_multi_day_avg_grid(df, list(DEFAULT_SITES), plot_diff=args.diff)
    print(f"Saved: {saved}")

if __name__ == "__main__":
    main()
