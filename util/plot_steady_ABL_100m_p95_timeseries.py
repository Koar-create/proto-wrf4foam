#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Line plot of per-case spatial 95th-percentile horizontal wind speed at 100 m
(from compute_100m_spatial_p95_windspeed.py output).

Usage:
  python util/plot_steady_ABL_100m_p95_timeseries.py
  python util/plot_steady_ABL_100m_p95_timeseries.py --csv results/wrf_openfoam/foo.csv --out results/wrf_openfoam/foo.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_CSV = (
    _repo_root()
    / "results"
    / "wrf_openfoam"
    / "steady_ABL_100m_spatial_p95_windspeed.csv"
)
DEFAULT_OUT = DEFAULT_CSV.with_suffix(".png")


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot 100m spatial p95 wind speed time series.")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output PNG path")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    args = ap.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else _repo_root() / args.csv
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "time_utc" not in df.columns or "wind_speed_p95_m_s" not in df.columns:
        raise SystemExit("Expected columns: time_utc, wind_speed_p95_m_s")

    t = pd.to_datetime(df["time_utc"], utc=True)
    y = df["wind_speed_p95_m_s"].astype(float)

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
        }
    )

    fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
    ax.plot(t, y, color="#2C3E50", lw=1.4, marker="o", ms=3, mfc="white", mew=0.8)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(r"Wind speed, 95th pct. (m s$^{-1}$)")
    ax.set_title("100 m horizontal slice: spatial 95th-percentile wind speed (non-sensitivity cases)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%Hh"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.grid(True, alpha=0.35, ls="--", lw=0.7)

    out_path = args.out if args.out.is_absolute() else _repo_root() / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
