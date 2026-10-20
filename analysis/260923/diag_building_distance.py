#!/usr/bin/env python3
"""Is the faster air next to buildings, or in the gaps between them?

Read-only diagnostic for manuscript Key Finding 2. Reuses the baseline field
definition from analysis/260923/local_wind_ratio.py and bins interior fluid
cells by their distance to the nearest building solid at the same slice
height. Reports Lambda' (domain-referenced) and Lambda (WRF-referenced).

    python analysis/260923/diag_building_distance.py [case ...]

Writes results/wrf_openfoam/local_wind_ratio/building_distance.csv.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "analysis", "260923"))
import local_wind_ratio as L  # noqa: E402

OUT_DIR = os.path.join(_REPO, "results", "wrf_openfoam", "local_wind_ratio")
BANDS = ((0, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 1e9))


def analyse(name: str, height: int, masks: dict) -> tuple[pd.DataFrame, dict]:
    case_dir = os.path.join(_REPO, "steady_experiments_finer_ABL", name)
    grid = L.load_cfd_grid(os.path.join(case_dir, "postProcessing", f"{height}m.csv"))
    solid = masks[height] & grid["sampled"]
    fluid = grid["sampled"] & ~masks[height] & L.interior_mask()
    v = grid["wind_speed"]
    v_wrf = L.wrf_speed_on_grid(case_dir, height)
    usable = fluid & np.isfinite(v) & np.isfinite(v_wrf) & (v_wrf >= L.WRF_SPEED_FLOOR)
    mean = float(np.mean(v[fluid & np.isfinite(v)]))
    lp = np.full(v.shape, np.nan)
    lp[usable] = v[usable] / mean
    ratio = np.full(v.shape, np.nan)
    ratio[usable] = v[usable] / v_wrf[usable]
    smoothed = L.smooth_fluid(lp, usable, L.BASE_SMOOTH_M)
    kept, zones = L.label_zones(
        smoothed, L.BASE_THRESHOLD, L.BASE_MIN_AREA_M2, above=True, companion=ratio,
    )

    dist_m = ndimage.distance_transform_edt(~solid) * L.DX
    cell_area_km2 = L.CELL_AREA / 1e6
    fast = kept & usable
    rows = []
    for lo, hi in BANDS:
        sel = usable & (dist_m >= lo) & (dist_m < hi)
        n = int(np.count_nonzero(sel))
        rows.append({
            "case": name, "height_m": height, "band_m": f"{lo:g}-{hi:g}",
            "cells": n,
            "fluid_km2": n * cell_area_km2,
            "mean_lp": float(np.nanmean(smoothed[sel])) if n else np.nan,
            "p90_lp": float(np.nanpercentile(smoothed[sel], 90)) if n else np.nan,
            "mean_R": float(np.nanmean(ratio[sel])) if n else np.nan,
            "p90_R": float(np.nanpercentile(ratio[sel], 90)) if n else np.nan,
            "fast_km2": float(np.count_nonzero(fast & sel) * cell_area_km2),
            "fast_frac": float(np.count_nonzero(fast & sel) / n) if n else np.nan,
        })
    total_fast = float(np.count_nonzero(fast) * cell_area_km2)
    info = {
        "case": name, "height_m": height,
        "fast_km2": total_fast,
        "fast_within_50m_of_building": (
            float(np.count_nonzero(fast & (dist_m < 50.0)) * cell_area_km2 / total_fast)
            if total_fast else np.nan
        ),
        "fluid_within_50m_of_building": float(np.count_nonzero(usable & (dist_m < 50.0)) / np.count_nonzero(usable)),
        "n_zones": len(zones),
    }
    return pd.DataFrame(rows), info


def main() -> None:
    names = sys.argv[1:] or [
        "20250901_1000_two_boundaries_as_outlet",
        "20250908_0000_two_boundaries_as_outlet",
    ]
    heights = tuple(int(h) for h in os.environ.get("LP_HEIGHTS", "").split(",") if h.strip()) or (30,)
    masks = L.load_masks(L.MASK_NPZ)
    frames, infos = [], []
    for height in heights:
        for name in names:
            try:
                frame, info = analyse(name, height, masks)
            except FileNotFoundError as exc:
                print(f"SKIP {name} z={height} m: {exc}")
                continue
            frames.append(frame)
            infos.append(info)
    print("=== where the fast patches sit ===")
    print(pd.DataFrame(infos).round(3).to_string(index=False))
    frame = pd.concat(frames, ignore_index=True)
    print()
    print("=== Lambda' and Lambda by distance to the nearest building ===")
    print(frame.pivot_table(index="band_m", columns=["height_m", "case"],
                            values="mean_lp").round(3).to_string())
    print()
    print("=== same, WRF-referenced Lambda ===")
    print(frame.pivot_table(index="band_m", columns=["height_m", "case"],
                            values="mean_R").round(3).to_string())
    print()
    print("=== fraction of fluid area in each band that is a fast patch ===")
    print(frame.pivot_table(index="band_m", columns=["height_m", "case"],
                            values="fast_frac").round(3).to_string())
    print()
    print("=== fluid area per band (km2), 30 m ===")
    print(frame[frame.height_m == 30].pivot_table(index="band_m", columns="case",
                                                  values="fluid_km2").round(3).to_string())
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "building_distance.csv")
    frame.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
