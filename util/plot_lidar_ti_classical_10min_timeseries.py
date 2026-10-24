#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR turbulence intensity (TI) time series — **classical 10-minute σ/μ**
(not TKE-derived from k).

TI definition (this script only):

    TI = std(WindSpd) / mean(WindSpd)

on non-overlapping 10-minute bins; bins drop if mean WindSpd < --ws-min or
count < --min-periods.

Methodology (repo-relative): docs/methodology/lidar_classical_ti_10min.md

Figures:
  - 1 station-mean (equal-weight across stations)
  - 1 per station
  - optional 2x2 panel (--panel-2x2): station mean + 3 stations, shared y

Nearest available measurement height to --height is used (often ~104 m for 100 m).

Usage:
  python util/plot_lidar_ti_classical_10min_timeseries.py
  python util/plot_lidar_ti_classical_10min_timeseries.py --panel-2x2
  python util/plot_lidar_ti_classical_10min_timeseries.py \\
    --csv data/260409/raw/lidar/lidar_transient.csv \\
    --out-dir results/timeseries_lidar/260409 \\
    --start 2025-09-01T00:00:00 --end 2025-09-13T23:50:00

Pair script (different TI): util/plot_lidar_ti_tke_derived_timeseries.py
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


DEFAULT_CSV = _repo_root() / "data/260409/raw/lidar/lidar_transient.csv"
DEFAULT_OUT_DIR = _repo_root() / "results/timeseries_lidar/260409"
DEFAULT_STATIONS = ("GAW103", "GAW104", "GAW111")
DEFAULT_WS_MIN = 0.5  # m/s
# Floor for sparsest station cadence (~2 min at GAW103 → ~5 pts / 10 min).
DEFAULT_MIN_PERIODS = 3  # min samples inside a 10-min bin
DEFAULT_YMAX = 0.8
DEFAULT_CHUNKSIZE = 500_000


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


def _load_filtered_transient(
    csv_path: Path,
    *,
    stations: list[str],
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    """Chunked load of datetime/obtid/Height/WindSpd with station+time filter."""
    usecols = ["datetime", "obtid", "Height", "WindSpd"]
    station_set = set(stations)
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=int(chunksize)):
        chunk["obtid"] = chunk["obtid"].astype(str)
        chunk = chunk[chunk["obtid"].isin(station_set)]
        if chunk.empty:
            continue
        chunk["datetime"] = pd.to_datetime(chunk["datetime"], utc=True)
        chunk = chunk[(chunk["datetime"] >= t0) & (chunk["datetime"] <= t1)]
        if chunk.empty:
            continue
        chunk["Height"] = pd.to_numeric(chunk["Height"], errors="coerce")
        chunk["WindSpd"] = pd.to_numeric(chunk["WindSpd"], errors="coerce")
        parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=usecols)
    return pd.concat(parts, ignore_index=True)


def ti_10min_from_ws(
    times: pd.Series,
    ws: pd.Series,
    *,
    ws_min: float = DEFAULT_WS_MIN,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> pd.Series:
    """
    Non-overlapping 10-min classical TI = std(WindSpd) / mean(WindSpd).

    Returns a Series indexed by bin start (UTC), NaN where invalid.
    """
    s = pd.Series(
        pd.to_numeric(ws, errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(times, utc=True)),
        dtype=float,
    ).sort_index()
    s = s.groupby(level=0).mean()

    mean_ws = s.resample("10min").mean()
    std_ws = s.resample("10min").std(ddof=1)
    count = s.resample("10min").count()

    ti = std_ws / mean_ws
    ti = ti.where(mean_ws >= float(ws_min))
    ti = ti.where(count >= int(min_periods))
    ti = ti.where(np.isfinite(ti) & (ti >= 0.0))
    ti.name = "TI"
    return ti


def _style_ax(
    ax,
    *,
    title: str,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    xlabel: bool = True,
    ylabel: bool = True,
    ymax: float | None = None,
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
    ax.set_xlim(t0, t1 + pd.Timedelta(minutes=10))
    if ymax is not None:
        ax.set_ylim(0.0, float(ymax))


def _draw_line(ax, times, ti, *, label: str) -> None:
    ax.plot(
        times,
        ti,
        linestyle="-",
        marker="o",
        markersize=2.0,
        linewidth=1.1,
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
    ymax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.8), constrained_layout=True)
    _draw_line(ax, times, ti, label=label)
    _style_ax(ax, title=title, t0=t0, t1=t1, ymax=ymax)
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
    ymax: float | None = None,
) -> None:
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
            ymax=ymax,
        )
        ax.legend(loc="best", frameon=True)

    fig.suptitle(
        f"LiDAR classical TI (10 min) | height={h_tag} (nearest {h_used_tag})",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "LiDAR TI time series, classical 10-min: TI = std(WindSpd)/mean(WindSpd) "
            "(see docs/methodology/lidar_classical_ti_10min.md)."
        ),
        epilog=(
            "TKE-derived TI (sqrt(2k/3)/WindSpd): "
            "util/plot_lidar_ti_tke_derived_timeseries.py"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input lidar transient CSV")
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
        default="2025-09-13T23:50:00",
        help="End time (UTC), inclusive (10-min bin starts)",
    )
    ap.add_argument(
        "--ws-min",
        type=float,
        default=DEFAULT_WS_MIN,
        help=f"Min bin-mean WindSpd (m/s) for valid TI (default {DEFAULT_WS_MIN})",
    )
    ap.add_argument(
        "--min-periods",
        type=int,
        default=DEFAULT_MIN_PERIODS,
        help=(
            f"Min samples per 10-min bin (default {DEFAULT_MIN_PERIODS}; "
            "set from sparsest-station cadence, e.g. GAW103 ~2 min)"
        ),
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
        help="CSV read chunk size",
    )
    ap.add_argument("--dpi", type=int, default=160, help="Figure DPI")
    ap.add_argument(
        "--panel-2x2",
        action="store_true",
        default=False,
        help="Also save a 2x2 composite (station mean + stations) with shared y-ticks",
    )
    ap.add_argument(
        "--ymax",
        type=float,
        default=DEFAULT_YMAX,
        help=f"Y-axis upper limit (default {DEFAULT_YMAX}; <=0 disables)",
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
    min_periods = int(args.min_periods)

    print(f"Loading {csv_path} (chunked filter)...")
    df = _load_filtered_transient(
        csv_path,
        stations=stations,
        t0=t0,
        t1=t1,
        chunksize=int(args.chunksize),
    )
    if df.empty:
        raise SystemExit("No rows after station/time filter.")

    h_used = _pick_nearest_height(df["Height"].dropna().unique(), h_req)
    df_h = df[np.isclose(df["Height"], h_used)].copy()
    if df_h.empty:
        raise SystemExit(f"No rows at nearest Height={h_used}")

    print(f"Requested height={h_req} m -> nearest available Height={h_used} m")
    print(
        f"Minute rows={len(df_h)}; stations={sorted(df_h['obtid'].unique())}; "
        f"span={df_h['datetime'].min()} -> {df_h['datetime'].max()}"
    )
    print(
        f"TI=std(WindSpd)/mean(WindSpd) over non-overlapping 10-min bins; "
        f"ws_min={ws_min} m/s; min_periods={min_periods}"
    )

    h_tag = _fmt_height_tag(h_req)
    h_used_tag = _fmt_height_tag(h_used)
    ymax_arg = float(args.ymax)
    ymax = None if ymax_arg <= 0 else ymax_arg

    station_series: list[tuple[str, pd.Series, pd.Series]] = []
    ti_frames: list[pd.Series] = []

    for obtid in stations:
        sub = df_h[df_h["obtid"] == obtid]
        if sub.empty:
            print(f"Skip {obtid}: no data at Height={h_used}")
            continue
        ti = ti_10min_from_ws(
            sub["datetime"],
            sub["WindSpd"],
            ws_min=ws_min,
            min_periods=min_periods,
        )
        ti = ti[(ti.index >= t0) & (ti.index <= t1)]
        n_valid = int(ti.notna().sum())
        print(f"  {obtid}: 10-min bins={len(ti)}, valid TI={n_valid}")
        ti_frames.append(ti.rename(obtid))
        times = ti.index.to_series()
        station_series.append((obtid, times, ti))
        out = out_dir / f"ts_ti10_lidar_obtid-{obtid}_h{h_tag}_raw_tz-utc.png"
        _plot_series(
            times,
            ti,
            title=(
                f"LiDAR classical TI (10 min) | obtid={obtid} | "
                f"height={h_tag} (nearest {h_used_tag})"
            ),
            out_path=out,
            label=obtid,
            t0=t0,
            t1=t1,
            dpi=int(args.dpi),
            ymax=ymax,
        )

    if not ti_frames:
        raise SystemExit("No station TI series produced.")

    pivot = pd.concat(ti_frames, axis=1).reindex(columns=stations).sort_index()
    mean_ti = pivot.mean(axis=1, skipna=True)
    mean_times = mean_ti.index.to_series()
    mean_label = "Station mean (" + "/".join(stations) + ")"
    out_mean = out_dir / f"ts_ti10_lidar_station-mean_h{h_tag}_raw_tz-utc.png"
    _plot_series(
        mean_times,
        mean_ti,
        title=(
            f"LiDAR classical TI (10 min) | station mean | "
            f"height={h_tag} (nearest {h_used_tag})"
        ),
        out_path=out_mean,
        label=mean_label,
        t0=t0,
        t1=t1,
        dpi=int(args.dpi),
        ymax=ymax,
    )
    n_st = pivot.notna().sum(axis=1)
    ti_finite = mean_ti.dropna()
    if ti_finite.empty:
        print("Station-mean: no valid TI values after masks.")
    else:
        print(
            f"Station-mean: n_bins={len(mean_ti)}, "
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
        out_panel = out_dir / f"ts_ti10_lidar_2x2_h{h_tag}_raw_tz-utc.png"
        _plot_panel_2x2(
            panels,
            out_path=out_panel,
            h_tag=h_tag,
            h_used_tag=h_used_tag,
            t0=t0,
            t1=t1,
            dpi=int(args.dpi),
            ymax=ymax,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
