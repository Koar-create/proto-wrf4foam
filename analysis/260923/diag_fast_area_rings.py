#!/usr/bin/env python3
"""Where does the interior fast area sit, relative to the lateral boundary?

Read-only diagnostic. Reuses the baseline field definition from
analysis/260923/local_wind_ratio.py (interior 500 m, 25 m smoothing,
Lambda' > 1.2) and bins the fast area by distance to the
/-2500 m lateral boundary. A boundary-driven artifact shows up as a
concentrated band in the outermost rings; a physical structure is spread
across the interior.

    python analysis/260923/diag_fast_area_rings.py 20250901_1000_two_boundaries_as_outlet ...

Writes results/wrf_openfoam/local_wind_ratio/fast_area_rings.csv.
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
RING_M = 250.0


def ring_profile(name: str, height: int, masks: dict, axis: np.ndarray) -> tuple[pd.DataFrame, dict]:
    case_dir = os.path.join(_REPO, "steady_experiments_finer_ABL", name)
    grid = L.load_cfd_grid(os.path.join(case_dir, "postProcessing", f"{height}m.csv"))
    fluid = grid["sampled"] & ~masks[height] & L.interior_mask()
    v = grid["wind_speed"]
    v_wrf = L.wrf_speed_on_grid(case_dir, height)
    usable = fluid & np.isfinite(v) & np.isfinite(v_wrf) & (v_wrf >= L.WRF_SPEED_FLOOR)
    mean = float(np.mean(v[fluid & np.isfinite(v)]))
    ratio_p = np.full(v.shape, np.nan)
    ratio_p[usable] = v[usable] / mean
    smoothed = L.smooth_fluid(ratio_p, usable, L.BASE_SMOOTH_M)
    ratio = np.full(v.shape, np.nan)
    ratio[usable] = v[usable] / v_wrf[usable]
    kept, zones = L.label_zones(
        smoothed, L.BASE_THRESHOLD, L.BASE_MIN_AREA_M2, above=True, companion=ratio,
    )

    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    dist = L.XY_MAX - np.maximum(np.abs(xx), np.abs(yy))
    ring = np.floor(dist / RING_M).astype(int)
    cell_area_km2 = L.CELL_AREA / 1e6

    rows = []
    solid = masks[height]
    for r in range(int(np.floor(L.XY_MAX / RING_M))):
        sel = usable & (ring == r)
        band = ring == r
        n_fluid = int(np.count_nonzero(sel))
        n_fast = int(np.count_nonzero(sel & kept))
        rows.append({
            "case": name, "height_m": height, "ring": r,
            "dist_lo_m": r * RING_M, "dist_hi_m": (r + 1) * RING_M,
            "fluid_km2": n_fluid * cell_area_km2,
            "fast_km2": n_fast * cell_area_km2,
            "fast_frac": n_fast / n_fluid if n_fluid else np.nan,
            "mean_lp": float(np.nanmean(smoothed[sel])) if n_fluid else np.nan,
            "p90_lp": float(np.nanpercentile(smoothed[sel], 90)) if n_fluid else np.nan,
            "solid_frac": (np.count_nonzero(solid & band) / np.count_nonzero(band))
            if np.count_nonzero(band) else np.nan,
        })

    # Lambda' versus local building coverage (51 cells is about 255 m).
    coverage = ndimage.uniform_filter(solid.astype(float), size=51)
    cov = coverage[usable]
    lp = smoothed[usable]
    finite = np.isfinite(lp)
    edges = np.arange(10) * 0.1
    idx = np.clip((cov[finite] / 0.1).astype(int), 0, 9)
    grouped = pd.DataFrame({"idx": idx, "lp": lp[finite]}).groupby("idx")["lp"].agg(["mean", "size"])
    cov_rows = pd.DataFrame({
        "case": name, "height_m": height,
        "coverage_lo": edges, "coverage_hi": edges + 0.1,
        "mean_lp": grouped["mean"].reindex(range(10)).to_numpy(),
        "cells": grouped["size"].reindex(range(10)).fillna(0).to_numpy(),
    })
    corr = float(np.corrcoef(cov[finite], lp[finite])[0, 1]) if finite.any() else np.nan
    cov_rows["corr_lp_coverage"] = corr
    biggest = max(zones, key=lambda z: z["area_m2"]) if zones else None
    info = {
        "case": name, "height_m": height, "domain_mean_v": mean,
        "n_zones": len(zones),
        "fast_total_km2": float(np.count_nonzero(kept) * cell_area_km2),
        "biggest_km2": (biggest["area_m2"] / 1e6) if biggest else 0.0,
        "biggest_mean_lp": (biggest["mean"] if biggest else np.nan),
        "biggest_mean_R": (biggest["mean_companion"] if biggest else np.nan),
        "biggest_peak_R": (biggest["peak_companion"] if biggest else np.nan),
        "biggest_centroid_x_m": (biggest["centroid_x_m"] if biggest else np.nan),
        "biggest_centroid_y_m": (biggest["centroid_y_m"] if biggest else np.nan),
    }
    return pd.DataFrame(rows), info, cov_rows


def main() -> None:
    names = sys.argv[1:] or [
        "20250901_1000_two_boundaries_as_outlet",
        "20250908_0000_two_boundaries_as_outlet",
    ]
    masks = L.load_masks(L.MASK_NPZ)
    axis = L.grid_axis()
    profiles, infos, covs = [], [], []
    heights = tuple(int(h) for h in os.environ.get("LP_HEIGHTS", "").split(",") if h.strip()) or L.HEIGHTS
    for height in heights:
        for name in names:
            try:
                prof, info, cov = ring_profile(name, height, masks, axis)
            except FileNotFoundError as exc:
                print(f"SKIP {name} z={height} m: {exc}")
                continue
            profiles.append(prof)
            infos.append(info)
            covs.append(cov)
    print("=== per-hour summary ===")
    print(pd.DataFrame(infos).round(3).to_string(index=False))
    print()
    prof = pd.concat(profiles, ignore_index=True)
    for (height, name), grp in prof.groupby(["height_m", "case"]):
        print(f"=== fast area by distance from the lateral boundary: {name} z={height} m ===")
        print(grp[["dist_lo_m", "dist_hi_m", "fluid_km2", "fast_km2", "fast_frac", "mean_lp", "solid_frac"]]
              .round(3).to_string(index=False))
        print()
    print("=== mean Lambda' versus local building coverage (250 m window) ===")
    cov = pd.concat(covs, ignore_index=True)
    print(cov.pivot_table(index=["coverage_lo", "coverage_hi"], columns=["height_m", "case"],
                          values="mean_lp").round(3).to_string())
    print()
    print("cells per coverage bin (identical across cases):")
    print(cov[cov.case == names[0]].pivot_table(index=["coverage_lo", "coverage_hi"], columns="height_m",
                                                values="cells").fillna(0).astype(int).to_string())
    print()
    print("corr(Lambda', building coverage) per case and height:")
    print(cov.pivot_table(index="case", columns="height_m", values="corr_lp_coverage").round(3).to_string())
    print()
    print("=== share of the interior fast area that sits within X m of the boundary ===")
    for name, grp in prof.groupby("case"):
        tot = grp.fast_km2.sum()
        row = {"case": name, "fast_km2": round(tot, 3)}
        for lim in (250, 500, 1000, 1500):
            row[f"within_{lim}m"] = round(grp.loc[grp.dist_lo_m < lim, "fast_km2"].sum() / tot, 3)
        print(pd.Series(row).to_string())
        print()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fast_area_rings.csv")
    prof.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
