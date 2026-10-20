#!/usr/bin/env python3
"""Distance-to-building statistics for every analysed hour and height.

For each (case, height) the interior fluid cells are binned by distance to the
nearest building solid at that height. Lambda' (domain-referenced) is always
available; Lambda (WRF-referenced) is only computed when the matching WRF
NetCDF exists locally, and rows carry a flag saying so.

Outputs
  results/wrf_openfoam/local_wind_ratio/distance_bands.csv   one row per case, height, band
  results/wrf_openfoam/local_wind_ratio/distance_hours.csv   one row per case and height

    python analysis/260923/diag_distance_all.py shard --shard 0 --shards 16
    python analysis/260923/diag_distance_all.py merge --shards 16
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import ndimage
from multiprocessing import Pool

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "analysis", "260923"))
import local_wind_ratio as L  # noqa: E402

ROOT = os.path.join(_REPO, "steady_experiments_finer_ABL")
OUT_DIR = os.path.join(_REPO, "results", "wrf_openfoam", "local_wind_ratio")
BANDS = ((0, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 1e9))
_MASKS: dict | None = None
_AXIS: np.ndarray | None = None


def _init() -> None:
    global _MASKS, _AXIS
    _MASKS = L.load_masks(L.MASK_NPZ)
    _AXIS = L.grid_axis()


def _case_time(case: str) -> str:
    parts = case.split("_")
    return f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:8]} {parts[1][:2]}:{parts[1][2:4]}:00"


def run_case(task: tuple[str, int]) -> tuple[list[dict], dict]:
    case, height = task
    try:
        case_dir = os.path.join(ROOT, case)
        grid = L.load_cfd_grid(os.path.join(case_dir, "postProcessing", f"{height}m.csv"))
        solid = _MASKS[height] & grid["sampled"]
        fluid = grid["sampled"] & ~_MASKS[height] & L.interior_mask()
        v = grid["wind_speed"]
        usable = fluid & np.isfinite(v)
        mean = float(np.mean(v[usable]))
        lp = np.full(v.shape, np.nan)
        lp[usable] = v[usable] / mean

        has_wrf = True
        try:
            v_wrf = L.wrf_speed_on_grid(case_dir, height)
        except FileNotFoundError:
            has_wrf = False
            v_wrf = np.full(v.shape, np.nan)
        if has_wrf:
            usable = usable & np.isfinite(v_wrf) & (v_wrf >= L.WRF_SPEED_FLOOR)
        ratio = np.full(v.shape, np.nan)
        if has_wrf:
            ratio[usable] = v[usable] / v_wrf[usable]

        smoothed = L.smooth_fluid(lp, usable, L.BASE_SMOOTH_M)
        kept, zones = L.label_zones(
            smoothed, L.BASE_THRESHOLD, L.BASE_MIN_AREA_M2, above=True, companion=ratio,
        )
        dist_m = ndimage.distance_transform_edt(~solid) * L.DX
        cell_km2 = L.CELL_AREA / 1e6
        fast = kept & usable
        fast_total = float(np.count_nonzero(fast) * cell_km2)
        n_usable = int(np.count_nonzero(usable))

        band_rows = []
        for lo, hi in BANDS:
            sel = usable & (dist_m >= lo) & (dist_m < hi)
            n = int(np.count_nonzero(sel))
            band_rows.append({
                "case": case, "time_utc": _case_time(case), "height_m": height,
                "has_wrf": has_wrf, "band_m": f"{lo:g}-{hi:g}",
                "cells": n, "fluid_frac": n / n_usable if n_usable else np.nan,
                "mean_lp": float(np.nanmean(lp[sel])) if n else np.nan,
                "mean_R": float(np.nanmean(ratio[sel])) if (n and has_wrf) else np.nan,
                "fast_km2": float(np.count_nonzero(fast & sel) * cell_km2),
                "fast_frac": float(np.count_nonzero(fast & sel) / n) if n else np.nan,
            })
        hour = {
            "case": case, "time_utc": _case_time(case), "height_m": height,
            "has_wrf": has_wrf,
            "fast_km2": fast_total,
            "fast_within_50m_frac": (
                float(np.count_nonzero(fast & (dist_m < 50.0)) * cell_km2 / fast_total)
                if fast_total else np.nan
            ),
            "fluid_within_50m_frac": float(np.count_nonzero(usable & (dist_m < 50.0)) / n_usable)
            if n_usable else np.nan,
            "n_zones": len(zones),
            "n_usable": n_usable,
        }
        return band_rows, hour
    except Exception:  # keep the pool alive; report the case
        print(f"FAILED {case} z={height}m\n{traceback.format_exc()}", flush=True)
        return [], {"case": case, "time_utc": _case_time(case), "height_m": height,
                    "has_wrf": False, "fast_km2": np.nan, "fast_within_50m_frac": np.nan,
                    "fluid_within_50m_frac": np.nan, "n_zones": np.nan, "n_usable": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", nargs="?", default="shard", choices=("shard", "merge"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=16)
    args = parser.parse_args()

    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    cases = sorted(
        d for d in os.listdir(ROOT)
        if os.path.isfile(os.path.join(ROOT, d, "myExpxx.foam"))
        and os.path.isfile(os.path.join(ROOT, d, "postProcessing", "30m.csv"))
    )
    tasks = [(c, h) for c in cases for h in L.HEIGHTS]

    if args.cmd == "merge":
        bands, hours = [], []
        for i in range(args.shards):
            for name, sink in ((f"distance_bands_shard{i}.csv", bands), (f"distance_hours_shard{i}.csv", hours)):
                path = os.path.join(out_dir, name)
                if os.path.isfile(path):
                    sink.append(pd.read_csv(path))
        pd.concat(bands, ignore_index=True).to_csv(
            os.path.join(out_dir, "distance_bands.csv"), index=False)
        pd.concat(hours, ignore_index=True).to_csv(
            os.path.join(out_dir, "distance_hours.csv"), index=False)
        print("wrote distance_bands.csv and distance_hours.csv", flush=True)
        return

    _init()
    my_tasks = tasks[args.shard::args.shards]
    bands_path = os.path.join(out_dir, f"distance_bands_shard{args.shard}.csv")
    hours_path = os.path.join(out_dir, f"distance_hours_shard{args.shard}.csv")
    done: set[tuple[str, int]] = set()
    if os.path.isfile(hours_path):
        prev = pd.read_csv(hours_path)
        done = {t for t in zip(prev.case, prev.height_m)}
    print(f"shard {args.shard}/{args.shards}: {len(my_tasks)} jobs, {len(done)} already done", flush=True)
    for n, (case, height) in enumerate(my_tasks):
        if (case, height) in done:
            continue
        bands, hour = run_case((case, height))
        if bands:
            pd.DataFrame(bands).to_csv(
                bands_path, mode="a", header=not os.path.isfile(bands_path), index=False)
        pd.DataFrame([hour]).to_csv(
            hours_path, mode="a", header=not os.path.isfile(hours_path), index=False)
        if n % 10 == 0:
            print(f"  shard {args.shard}: {n}/{len(my_tasks)} {case} z={height}m", flush=True)
    print(f"shard {args.shard} finished", flush=True)


if __name__ == "__main__":
    main()
