#!/usr/bin/env python3
"""Aggregate the distance-to-building sweep into manuscript-sized numbers.

Reads the sweep outputs written by diag_distance_all.py.

    python analysis/260923/diag_distance_summary.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(_REPO, "results", "wrf_openfoam", "local_wind_ratio")
BANDS = ["0-25", "25-50", "50-100", "100-200", "200-400", "400-1e+09"]
NEAR, FAR = "0-25", "400-1e+09"


def regime_of(ts: str) -> str:
    t = pd.Timestamp(ts, tz="UTC")
    return "typhoon" if pd.Timestamp("2025-09-07 03:00", tz="UTC") <= t <= pd.Timestamp(
        "2025-09-08 19:00", tz="UTC") else "regular"


def main() -> None:
    hours = pd.read_csv(os.path.join(DATA_DIR, "distance_hours.csv"))
    bands = pd.read_csv(os.path.join(DATA_DIR, "distance_bands.csv"))
    hours["regime"] = hours.time_utc.map(regime_of)
    bands["regime"] = bands.time_utc.map(regime_of)

    print("=== coverage ===")
    print(hours.groupby(["regime", "height_m"]).agg(
        hours=("case", "size"), with_wrf=("has_wrf", "sum")).to_string())

    print()
    print("=== median over hours of mean Lambda' by distance band ===")
    tab = bands.pivot_table(index=["regime", "height_m"], columns="band_m",
                            values="mean_lp", aggfunc="median").round(3)
    print(tab.reindex(columns=[b for b in BANDS if b in tab.columns]).to_string())

    print()
    print("=== same, WRF-referenced Lambda (hours with WRF only) ===")
    tab_r = bands[bands.has_wrf].pivot_table(index=["regime", "height_m"], columns="band_m",
                                             values="mean_R", aggfunc="median").round(3)
    print(tab_r.reindex(columns=[b for b in BANDS if b in tab_r.columns]).to_string())

    print()
    print("=== spread across hours of the near and far bands ===")
    q = bands[bands.band_m.isin([NEAR, FAR])].pivot_table(
        index=["regime", "height_m", "band_m"], values=["mean_lp", "mean_R"],
        aggfunc=[lambda x: np.nanpercentile(x, 10), "median", lambda x: np.nanpercentile(x, 90)]).round(3)
    print(q.to_string())

    print()
    print("=== fast-patch area within 50 m of a building ===")
    print(hours.groupby(["regime", "height_m"]).agg(
        fast_within_50m=("fast_within_50m_frac", "median"),
        fluid_within_50m=("fluid_within_50m_frac", "median"),
        fast_km2=("fast_km2", "median"),
        zones=("n_zones", "median")).round(3).to_string())

    print()
    print("=== how often does each hour reproduce the ordering (near < far)? ===")
    pivot = bands.pivot_table(index=["case", "height_m"], columns="band_m", values="mean_lp")
    ok = (pivot[NEAR] < pivot[FAR])
    frame = ok.reset_index(name="ordered").merge(
        hours[["case", "height_m", "regime"]], on=["case", "height_m"])
    print(frame.groupby(["regime", "height_m"]).ordered.agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()
