#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 `steady_experiments_finer_ABL/WRF Atmospheric Stability Data Organization.csv`
复现四面板时间序列图（与既有读图说明一致）：
  1) 两边界稳定度按序数融合为一条色带 + LLJ 白点
  2) k_max：East 实线蓝 / South 虚线橙
  3) 500 m 以上平均 k：对数轴，双线
  4) Lt_max：双线

默认 CSV 路径为仓库内 steady_experiments_finer_ABL 下同名文件；可用 --csv / --out 覆盖。
"""

from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


REGIME_TO_CODE = {
    "Unstable / Convective": 0,
    "Neutral / Weakly Stable": 1,
    "Strongly Stable": 2,
}

CODE_TO_COLOR = {
    0: "#E74C3C",  # Unstable / Convective
    1: "#27AE60",  # Neutral / Weakly Stable
    2: "#8E44AD",  # Strongly Stable
}

EAST_COLOR = "#3B5998"
SOUTH_COLOR = "#D9822B"

# 默认 x 轴右端（UTC），避免日出/日落竖线把坐标撑到无数据区域
DEFAULT_X_RIGHT_UTC = pd.Timestamp("2025-09-04 06:00:00", tz=timezone.utc)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_csv() -> Path:
    return _repo_root() / "steady_experiments_finer_ABL" / "WRF Atmospheric Stability Data Organization.csv"


def _parse_record_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y%m%d_%H%M", utc=True)


def _llj_active(val) -> bool:
    s = str(val).strip().lower()
    return s.startswith("yes")


def _merged_regime_code(c0: int, c1: int) -> int:
    """两边界稳定度融合：极端差异取中间态，与读图“过渡带”一致。"""
    ms = 0.5 * (c0 + c1)
    if ms < 0.5:
        return 0
    if ms > 1.5:
        return 2
    return 1


def load_and_merge(df: pd.DataFrame) -> pd.DataFrame:
    col_record = "Record (UTC)"
    col_patch = "Boundary Patch"
    col_kmax = "k_max (m²/s²)"
    col_k500 = "Avg k > 500m (m²/s²)"
    col_lt = "Lt_max (m)"
    col_llj = "LLJ (Low-Level Jet)"
    col_regime = "Stability Regime"

    df = df.copy()
    df["dt"] = _parse_record_utc(df[col_record])

    east = df[df[col_patch] == "East"].sort_values("dt").reset_index(drop=True)
    south = df[df[col_patch] == "South"].sort_values("dt").reset_index(drop=True)
    if len(east) != len(south) or not np.array_equal(east["dt"].values, south["dt"].values):
        raise ValueError("East/South 行数或与时间戳不一致，请检查 CSV。")

    out = pd.DataFrame(
        {
            "dt": east["dt"],
            "k_max_east": east[col_kmax].astype(float).values,
            "k_max_south": south[col_kmax].astype(float).values,
            "k500_east": east[col_k500].astype(float).values,
            "k500_south": south[col_k500].astype(float).values,
            "lt_east": east[col_lt].astype(float).values,
            "lt_south": south[col_lt].astype(float).values,
            "llj_east": east[col_llj].map(_llj_active).values,
            "llj_south": south[col_llj].map(_llj_active).values,
        }
    )
    c_e = east[col_regime].astype(str).str.strip().map(REGIME_TO_CODE).astype(int).values
    c_s = south[col_regime].astype(str).str.strip().map(REGIME_TO_CODE).astype(int).values
    out["regime_merged"] = [_merged_regime_code(int(a), int(b)) for a, b in zip(c_e, c_s)]
    out["llj_any"] = out["llj_east"] | out["llj_south"]
    return out


def _add_sunrise_sunset_vlines(
    axes,
    t_series: pd.Series,
    xmax: pd.Timestamp | None = None,
) -> None:
    """与 `scripts/task5_sample_space_map.py` ax3 一致：UTC 日落 10:00、日出 22:00。
    若给定 xmax，则不绘制超过右边界的时间线（避免仅用于撑轴线的竖线）。"""
    tmin, tmax = t_series.min(), t_series.max()
    days = pd.date_range(tmin.floor("D"), tmax.ceil("D"), freq="D", tz="UTC")
    for ax in axes:
        for d in days:
            for hrs, color, ls in (
                (10, "#d62728", "--"),
                (22, "#1f77b4", "-."),
            ):
                t_line = d + pd.Timedelta(hours=hrs)
                if xmax is not None and t_line > xmax:
                    continue
                ax.axvline(
                    t_line,
                    color=color,
                    linestyle=ls,
                    linewidth=1.3,
                    alpha=0.75,
                    zorder=8,
                )


def plot_stability_figure(
    merged: pd.DataFrame,
    out_path: Path,
    dpi: int,
    x_right_utc: pd.Timestamp | None = None,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, constrained_layout=False)
    ax0, ax1, ax2, ax3 = axes

    # --- Panel 0: regime + LLJ ---
    ax0.set_facecolor("#F5F5F5")
    for _, row in merged.iterrows():
        t0 = row["dt"]
        t1 = t0 + pd.Timedelta(hours=1)
        code = int(row["regime_merged"])
        ax0.axvspan(
            t0,
            t1,
            ymin=0.0,
            ymax=1.0,
            facecolor=CODE_TO_COLOR[code],
            edgecolor="none",
            alpha=0.92,
        )
        if bool(row["llj_any"]):
            ax0.scatter(
                t0 + pd.Timedelta(minutes=30),
                0.5,
                s=36,
                c="white",
                zorder=5,
                edgecolors="#333333",
                linewidths=0.35,
            )
    ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.set_title(
        "Stability regime + LLJ (both boundaries averaged; dot = LLJ active)",
        fontsize=11,
    )

    # --- Panel 1–3: lines ---
    ax1.plot(merged["dt"], merged["k_max_east"], color=EAST_COLOR, lw=1.6, label="East")
    ax1.plot(
        merged["dt"],
        merged["k_max_south"],
        color=SOUTH_COLOR,
        lw=1.6,
        ls="--",
        label="South",
    )
    ax1.set_ylabel(r"$k_{\max}$ [m²/s²]")
    ax1.set_title("k_max [m²/s²] — turbulent kinetic energy peak", fontsize=11)
    ax1.set_ylim(0, 4)
    ax1.grid(True, axis="y", alpha=0.35)

    ax2.plot(merged["dt"], merged["k500_east"], color=EAST_COLOR, lw=1.6, label="East")
    ax2.plot(
        merged["dt"],
        merged["k500_south"],
        color=SOUTH_COLOR,
        lw=1.6,
        ls="--",
        label="South",
    )
    ax2.set_yscale("log")
    ax2.set_ylim(1e-4, 1.0)
    ax2.set_ylabel(r"Avg $k$ above 500 m [m²/s²]")
    ax2.set_title("Avg k above 500 m [m²/s²] — upper-layer TKE (log scale)", fontsize=11)
    ax2.grid(True, which="both", axis="y", alpha=0.35)

    ax3.plot(merged["dt"], merged["lt_east"], color=EAST_COLOR, lw=1.6, label="East")
    ax3.plot(
        merged["dt"],
        merged["lt_south"],
        color=SOUTH_COLOR,
        lw=1.6,
        ls="--",
        label="South",
    )
    ax3.set_ylabel(r"$L_{t,\max}$ [m]")
    ax3.set_title(r"$L_{t,\max}$ [m] — maximum mixing length", fontsize=11)
    ymin = float(np.nanmin([merged["lt_east"].min(), merged["lt_south"].min()]))
    ymax = float(np.nanmax([merged["lt_east"].max(), merged["lt_south"].max()]))
    pad = max(1.0, 0.02 * (ymax - ymin))
    ax3.set_ylim(ymin - pad, ymax + pad)
    ax3.grid(True, axis="y", alpha=0.35)

    x_left = merged["dt"].min()
    x_right = x_right_utc if x_right_utc is not None else DEFAULT_X_RIGHT_UTC

    # --- Sunrise/sunset guides (same as scripts/task5_sample_space_map.py ax3) ---
    _add_sunrise_sunset_vlines(axes, merged["dt"], xmax=x_right)

    for ax in axes:
        ax.set_xlim(x_left, x_right)

    # --- Shared x-axis (UTC): 6-hour ticks, label shows hour + date on two lines ---
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 6, 12, 18)))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    ax3.set_xlabel("Time (UTC)")

    regime_patches = [
        Patch(facecolor=CODE_TO_COLOR[0], edgecolor="none", label="Unstable / Convective"),
        Patch(facecolor=CODE_TO_COLOR[1], edgecolor="none", label="Neutral / Weakly Stable"),
        Patch(facecolor=CODE_TO_COLOR[2], edgecolor="none", label="Strongly Stable"),
    ]
    h_e, = ax1.plot([], [], color=EAST_COLOR, lw=1.8, label="East boundary")
    h_s, = ax1.plot([], [], color=SOUTH_COLOR, lw=1.8, ls="--", label="South boundary")
    sunset_proxy = Line2D(
        [0],
        [0],
        color="#d62728",
        ls="--",
        lw=1.3,
        alpha=0.75,
        label="Sunset (UTC 10:00)",
    )
    sunrise_proxy = Line2D(
        [0],
        [0],
        color="#1f77b4",
        ls="-.",
        lw=1.3,
        alpha=0.75,
        label="Sunrise (UTC 22:00)",
    )
    fig.legend(
        handles=regime_patches + [h_e, h_s, sunset_proxy, sunrise_proxy],
        loc="upper center",
        ncol=7,
        frameon=True,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.99),
    )

    fig.autofmt_xdate(rotation=0)
    fig.subplots_adjust(top=0.86, hspace=0.28, left=0.09, right=0.97, bottom=0.10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot WRF atmospheric stability organization CSV.")
    parser.add_argument("--csv", type=Path, default=None, help="Input CSV path")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: next to CSV as wrf_atmospheric_stability_organization.png)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--x-right",
        type=str,
        default=None,
        help="x-axis right limit (UTC), e.g. 2025-09-04T06:00:00Z (default: 2025-09-04 06:00 UTC)",
    )
    args = parser.parse_args()

    csv_path = args.csv or _default_csv()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    out_path = args.out or (csv_path.parent / "wrf_atmospheric_stability_organization.png")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    merged = load_and_merge(df)
    x_right = (
        pd.to_datetime(args.x_right, utc=True)
        if args.x_right
        else DEFAULT_X_RIGHT_UTC
    )
    plot_stability_figure(merged, out_path, args.dpi, x_right_utc=x_right)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
