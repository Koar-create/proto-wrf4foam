#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR turbulence intensity (TI) time series — **TKE-derived** (not classical σ/μ).

TI definition (this script only):

    TI = sqrt(2*k/3) / WindSpd

WindSpd below --ws-min is masked to NaN.

Methodology (repo-relative): docs/methodology/lidar_tke_derived_ti.md

Figures:
  - 1 station-mean (equal-weight across stations)
  - 1 per station
  - optional 2x2 panel (--panel-2x2): station mean + 3 stations, shared y

Nearest available measurement height to --height is used (often ~104 m for 100 m).

Usage:
  python util/plot_lidar_ti_tke_derived_timeseries.py
  python util/plot_lidar_ti_tke_derived_timeseries.py --panel-2x2
  python util/plot_lidar_ti_tke_derived_timeseries.py \\
    --csv data/260409/raw/lidar/lidar_1h-rolling.csv \\
    --out-dir results/timeseries_lidar/260409 \\
    --start 2025-09-01T00:00:00 --end 2025-09-13T23:00:00

Pair script (different TI): util/plot_lidar_ti_classical_10min_timeseries.py
"""

from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_CSV = _repo_root() / "data/260409/raw/lidar/lidar_1h-rolling.csv"
DEFAULT_OUT_DIR = _repo_root() / "results/timeseries_lidar/260409"
DEFAULT_STATIONS = ("GAW103", "GAW104", "GAW111")
DEFAULT_WS_MIN = 0.5  # m/s; WindSpd below this -> TI = NaN


def _fmt_height_tag(h: float) -> str:
    if float(h).is_integer():
        return f"{int(h)}m"
    s = f"{h:.1f}".rstrip("0").rstrip(".")
    return f"{s}m"


def _pick_nearest_height(heights: np.ndarray, h_req: float) -> float:
    heights = np.asarray(heights, dtype=float)
    if len(heights) == 0:
        raise ValueError("No valid Height values.")
    return float(heights[int(np.argmin(np.abs(heights - float(h_req))))])


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else _repo_root() / path


def ti_from_k_ws(
    k: pd.Series | np.ndarray,
    ws: pd.Series | np.ndarray,
    *,
    ws_min: float = DEFAULT_WS_MIN,
) -> pd.Series:
    """TKE-derived isotropic TI: sqrt(2*k/3) / WindSpd, NaN when ws < ws_min."""
    k_arr = pd.to_numeric(pd.Series(k), errors="coerce")
    ws_arr = pd.to_numeric(pd.Series(ws), errors="coerce")
    sigma = np.sqrt((2.0 / 3.0) * k_arr.clip(lower=0.0))
    ti = sigma / ws_arr
    ti = ti.where(ws_arr >= float(ws_min))
    ti = ti.where(np.isfinite(ti) & (ti >= 0.0))
    return ti


def _style_ax(
    ax,
    *,
    title: str,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    xlabel: bool = True,
    ylabel: bool = True,
) -> None:
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel("Time (UTC)")
    if ylabel:
        ax.set_ylabel("Turbulence intensity (-)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz=timezone.utc))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlim(t0, t1 + pd.Timedelta(hours=1))


def _draw_line(ax, times, ti, *, label: str) -> None:
    ax.plot(
        times,
        ti,
        linestyle="-",
        marker="o",
        markersize=2.8,
        linewidth=1.4,
        color="black",
        alpha=0.9,
        label=label,
    )


def _plot_series(
    times: pd.Series,
    ti: pd.Series,
    *,
    title: str,
    out_path: Path,
    label: str,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.8), constrained_layout=True)
    _draw_line(ax, times, ti, label=label)
    _style_ax(ax, title=title, t0=t0, t1=t1)
    ax.legend(loc="best", frameon=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


def _plot_panel_2x2(
    panels: list[tuple[str, pd.Series, pd.Series, str]],
    *,
    out_path: Path,
    h_tag: str,
    h_used_tag: str,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    dpi: int,
) -> None:
    """panels: list of (title, times, ti, legend_label), length 4, row-major."""
    if len(panels) != 4:
        raise ValueError(f"--panel-2x2 expects 4 panels, got {len(panels)}")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 8.0),
        sharey=True,
        constrained_layout=True,
    )
    for ax, (title, times, ti, label) in zip(axes.ravel(), panels):
        _draw_line(ax, times, ti, label=label)
        row = int(ax.get_subplotspec().rowspan.start)
        col = int(ax.get_subplotspec().colspan.start)
        _style_ax(
            ax,
            title=title,
            t0=t0,
            t1=t1,
            xlabel=(row == 1),
            ylabel=(col == 0),
        )
        ax.legend(loc="best", frameon=True)

    fig.suptitle(
        f"LiDAR TKE-derived TI | height={h_tag} (nearest {h_used_tag})",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "LiDAR TI time series, TKE-derived: TI = sqrt(2*k/3)/WindSpd "
            "(see docs/methodology/lidar_tke_derived_ti.md)."
        ),
        epilog=(
            "Classical 10-min TI (std/mean): "
            "util/plot_lidar_ti_classical_10min_timeseries.py"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input lidar CSV")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    ap.add_argument("--height", type=float, default=100.0, help="Requested height (m)")
    ap.add_argument(
        "--stations",
        nargs="+",
        default=list(DEFAULT_STATIONS),
        help="Station ids, e.g. --stations GAW103 GAW104 GAW111",
    )
    ap.add_argument(
        "--start",
        default="2025-09-01T00:00:00",
        help="Start time (UTC), inclusive",
    )
    ap.add_argument(
        "--end",
        default="2025-09-13T23:00:00",
        help="End time (UTC), inclusive",
    )
    ap.add_argument(
        "--ws-min",
        type=float,
        default=DEFAULT_WS_MIN,
        help=f"Min WindSpd (m/s) for valid TI (default {DEFAULT_WS_MIN})",
    )
    ap.add_argument("--dpi", type=int, default=160, help="Figure DPI")
    ap.add_argument(
        "--panel-2x2",
        action="store_true",
        default=False,
        help="Also save a 2x2 composite (station mean + stations) with shared y-ticks",
    )
    args = ap.parse_args()

    csv_path = _resolve(args.csv)
    out_dir = _resolve(args.out_dir)
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    t0 = pd.Timestamp(args.start, tz="UTC")
    t1 = pd.Timestamp(args.end, tz="UTC")
    stations = [str(s) for s in args.stations]
    h_req = float(args.height)
    ws_min = float(args.ws_min)

    df = pd.read_csv(csv_path, usecols=["datetime", "obtid", "Height", "WindSpd", "k"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["obtid"] = df["obtid"].astype(str)
    df["Height"] = pd.to_numeric(df["Height"], errors="coerce")
    df["WindSpd"] = pd.to_numeric(df["WindSpd"], errors="coerce")
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df = df[df["obtid"].isin(stations)].copy()
    df = df[(df["datetime"] >= t0) & (df["datetime"] <= t1)].copy()
    if df.empty:
        raise SystemExit("No rows after station/time filter.")

    h_used = _pick_nearest_height(df["Height"].dropna().unique(), h_req)
    df_h = df[np.isclose(df["Height"], h_used)].copy()
    if df_h.empty:
        raise SystemExit(f"No rows at nearest Height={h_used}")

    df_h["TI"] = ti_from_k_ws(df_h["k"], df_h["WindSpd"], ws_min=ws_min)
    n_valid = int(df_h["TI"].notna().sum())
    n_masked = int(len(df_h) - n_valid)

    print(f"Requested height={h_req} m -> nearest available Height={h_used} m")
    print(
        f"Rows={len(df_h)}; stations={sorted(df_h['obtid'].unique())}; "
        f"span={df_h['datetime'].min()} -> {df_h['datetime'].max()}"
    )
    print(
        f"TI=sqrt(2*k/3)/WindSpd; ws_min={ws_min} m/s; "
        f"valid={n_valid}, masked/NaN={n_masked}"
    )

    h_tag = _fmt_height_tag(h_req)
    h_used_tag = _fmt_height_tag(h_used)

    station_series: list[tuple[str, pd.Series, pd.Series]] = []
    for obtid in stations:
        sub = df_h[df_h["obtid"] == obtid].sort_values("datetime")
        if sub.empty:
            print(f"Skip {obtid}: no data at Height={h_used}")
            continue
        station_series.append((obtid, sub["datetime"], sub["TI"]))
        out = out_dir / f"ts_ti_lidar_obtid-{obtid}_h{h_tag}_raw_tz-utc.png"
        _plot_series(
            sub["datetime"],
            sub["TI"],
            title=(
                f"LiDAR TKE-derived TI | obtid={obtid} | "
                f"height={h_tag} (nearest {h_used_tag})"
            ),
            out_path=out,
            label=obtid,
            t0=t0,
            t1=t1,
            dpi=int(args.dpi),
        )

    pivot = (
        df_h.pivot_table(index="datetime", columns="obtid", values="TI", aggfunc="mean")
        .reindex(columns=stations)
        .sort_index()
    )
    mean_ti = pivot.mean(axis=1, skipna=True)
    mean_times = mean_ti.index.to_series()
    mean_label = "Station mean (" + "/".join(stations) + ")"
    out_mean = out_dir / f"ts_ti_lidar_station-mean_h{h_tag}_raw_tz-utc.png"
    _plot_series(
        mean_times,
        mean_ti,
        title=(
            f"LiDAR TKE-derived TI | station mean | "
            f"height={h_tag} (nearest {h_used_tag})"
        ),
        out_path=out_mean,
        label=mean_label,
        t0=t0,
        t1=t1,
        dpi=int(args.dpi),
    )
    n_st = pivot.notna().sum(axis=1)
    ti_finite = mean_ti.dropna()
    if ti_finite.empty:
        print("Station-mean: no valid TI values after ws_min mask.")
    else:
        print(
            f"Station-mean: n_hours={len(mean_ti)}, "
            f"all_stations={int((n_st == len(stations)).sum())}, "
            f"TI=[{ti_finite.min():.3f}, {ti_finite.max():.3f}]"
        )

    if args.panel_2x2:
        if len(station_series) != 3:
            raise SystemExit(
                f"--panel-2x2 needs exactly 3 stations with data, got {len(station_series)}"
            )
        panels = [
            ("Station mean", mean_times, mean_ti, mean_label),
            (
                station_series[0][0],
                station_series[0][1],
                station_series[0][2],
                station_series[0][0],
            ),
            (
                station_series[1][0],
                station_series[1][1],
                station_series[1][2],
                station_series[1][0],
            ),
            (
                station_series[2][0],
                station_series[2][1],
                station_series[2][2],
                station_series[2][0],
            ),
        ]
        out_panel = out_dir / f"ts_ti_lidar_2x2_h{h_tag}_raw_tz-utc.png"
        _plot_panel_2x2(
            panels,
            out_path=out_panel,
            h_tag=h_tag,
            h_used_tag=h_used_tag,
            t0=t0,
            t1=t1,
            dpi=int(args.dpi),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
