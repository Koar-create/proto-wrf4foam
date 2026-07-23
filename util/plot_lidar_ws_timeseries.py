#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot LiDAR hourly wind-speed time series at a requested height:
  - 1 station-mean figure (equal-weight mean across stations)
  - 1 figure per station
  - optional: 2x2 panel composite (station mean + 3 stations), sharey

No exact Height==100 m in lidar_1h-rolling.csv; nearest available level is used
(typically 104 m), matching scripts/plot_timeseries_lidar_wrf_cfd.py.

Usage:
  python util/plot_lidar_ws_timeseries.py
  python util/plot_lidar_ws_timeseries.py --panel-2x2
  python util/plot_lidar_ws_timeseries.py \\
    --csv data/260409/raw/lidar/lidar_1h-rolling.csv \\
    --out-dir results/timeseries_lidar/260409 \\
    --start 2025-09-01T00:00:00 --end 2025-09-13T23:00:00
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
        ax.set_ylabel("Wind speed (m/s)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz=timezone.utc))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlim(t0, t1 + pd.Timedelta(hours=1))


def _draw_line(ax, times, ws, *, label: str) -> None:
    ax.plot(
        times,
        ws,
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
    ws: pd.Series,
    *,
    title: str,
    out_path: Path,
    label: str,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.8), constrained_layout=True)
    _draw_line(ax, times, ws, label=label)
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
    """panels: list of (title, times, ws, legend_label), length 4, row-major."""
    if len(panels) != 4:
        raise ValueError(f"--panel-2x2 expects 4 panels, got {len(panels)}")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 8.0),
        sharey=True,
        constrained_layout=True,
    )
    for ax, (title, times, ws, label) in zip(axes.ravel(), panels):
        _draw_line(ax, times, ws, label=label)
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
        f"LiDAR wind speed | height={h_tag} (nearest {h_used_tag})",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot LiDAR hourly WS time series (station-mean + per-station)."
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

    df = pd.read_csv(csv_path, usecols=["datetime", "obtid", "Height", "WindSpd"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["obtid"] = df["obtid"].astype(str)
    df["Height"] = pd.to_numeric(df["Height"], errors="coerce")
    df["WindSpd"] = pd.to_numeric(df["WindSpd"], errors="coerce")
    df = df[df["obtid"].isin(stations)].copy()
    df = df[(df["datetime"] >= t0) & (df["datetime"] <= t1)].copy()
    if df.empty:
        raise SystemExit("No rows after station/time filter.")

    h_used = _pick_nearest_height(df["Height"].dropna().unique(), h_req)
    df_h = df[np.isclose(df["Height"], h_used)].copy()
    if df_h.empty:
        raise SystemExit(f"No rows at nearest Height={h_used}")

    print(f"Requested height={h_req} m -> nearest available Height={h_used} m")
    print(
        f"Rows={len(df_h)}; stations={sorted(df_h['obtid'].unique())}; "
        f"span={df_h['datetime'].min()} -> {df_h['datetime'].max()}"
    )

    h_tag = _fmt_height_tag(h_req)
    h_used_tag = _fmt_height_tag(h_used)

    station_series: list[tuple[str, pd.Series, pd.Series]] = []
    for obtid in stations:
        sub = df_h[df_h["obtid"] == obtid].sort_values("datetime")
        if sub.empty:
            print(f"Skip {obtid}: no data at Height={h_used}")
            continue
        station_series.append((obtid, sub["datetime"], sub["WindSpd"]))
        out = out_dir / f"ts_ws_lidar_obtid-{obtid}_h{h_tag}_raw_tz-utc.png"
        _plot_series(
            sub["datetime"],
            sub["WindSpd"],
            title=(
                f"LiDAR wind speed | obtid={obtid} | "
                f"height={h_tag} (nearest {h_used_tag})"
            ),
            out_path=out,
            label=obtid,
            t0=t0,
            t1=t1,
            dpi=int(args.dpi),
        )

    pivot = (
        df_h.pivot_table(index="datetime", columns="obtid", values="WindSpd", aggfunc="mean")
        .reindex(columns=stations)
        .sort_index()
    )
    mean_ws = pivot.mean(axis=1, skipna=True)
    mean_times = mean_ws.index.to_series()
    mean_label = "Station mean (" + "/".join(stations) + ")"
    out_mean = out_dir / f"ts_ws_lidar_station-mean_h{h_tag}_raw_tz-utc.png"
    _plot_series(
        mean_times,
        mean_ws,
        title=(
            f"LiDAR wind speed | station mean | "
            f"height={h_tag} (nearest {h_used_tag})"
        ),
        out_path=out_mean,
        label=mean_label,
        t0=t0,
        t1=t1,
        dpi=int(args.dpi),
    )
    n_st = pivot.notna().sum(axis=1)
    print(
        f"Station-mean: n_hours={len(mean_ws)}, "
        f"all_stations={int((n_st == len(stations)).sum())}, "
        f"ws=[{mean_ws.min():.2f}, {mean_ws.max():.2f}] m/s"
    )

    if args.panel_2x2:
        if len(station_series) != 3:
            raise SystemExit(
                f"--panel-2x2 needs exactly 3 stations with data, got {len(station_series)}"
            )
        # Layout (row-major): mean | st0 / st1 | st2
        panels = [
            ("Station mean", mean_times, mean_ws, mean_label),
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
        out_panel = out_dir / f"ts_ws_lidar_2x2_h{h_tag}_raw_tz-utc.png"
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
