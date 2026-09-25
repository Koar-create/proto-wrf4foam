#!/usr/bin/env python3
"""Hourly along-route exposure on the section 3.2 5 m grid.

Reuses local_wind_ratio grids, building masks, WRF interpolation and
fast/deceleration patches. One row per case x route x height, plus an
along-track npz of wind speed and Lambda.

Shear is filled later by merge_route_shear.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ANA = _REPO / "analysis" / "260923"
_UTIL = _REPO / "util"
for p in (str(_ANA), str(_UTIL), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import local_wind_ratio as L  # noqa: E402
import uav_route_wind_shear_analysis as base  # noqa: E402
import visualize_WRF_CFD_xy_two_panel as vis  # noqa: E402

HEIGHTS = (30, 60, 120)
CASE_ROOT = _REPO / "steady_experiments_finer_ABL"
OUT_DIR = _REPO / "results" / "uav_route_exposure_hourly"
BJT = L.BJT


def snapped_route(name: str, x0: float, y0: float, x1: float, y1: float) -> pd.DataFrame:
    """Walk grid nodes from the snapped start cell to the snapped end cell."""
    axis = L.grid_axis()
    iy0, ix0 = L.xy_to_ij(np.array([x0]), np.array([y0]))
    iy1, ix1 = L.xy_to_ij(np.array([x1]), np.array([y1]))
    iy0, ix0, iy1, ix1 = int(iy0[0]), int(ix0[0]), int(iy1[0]), int(ix1[0])
    n = max(abs(ix1 - ix0), abs(iy1 - iy0))
    if n < 1:
        raise ValueError(name)
    ix = np.rint(np.linspace(ix0, ix1, n + 1)).astype(np.int32)
    iy = np.rint(np.linspace(iy0, iy1, n + 1)).astype(np.int32)
    x = axis[ix]
    y = axis[iy]
    step = np.hypot(np.diff(x), np.diff(y))
    dist = np.concatenate([[0.0], np.cumsum(step)])
    return pd.DataFrame(
        {
            "route": name,
            "seq": np.arange(len(x), dtype=int),
            "x": x,
            "y": y,
            "ix": ix,
            "iy": iy,
            "distance_m": dist,
        }
    )


def build_routes() -> dict[str, pd.DataFrame]:
    routes = {}
    for name, spec in base.ROUTE_SPECS.items():
        x0, y0 = spec["start"]
        x1, y1 = spec["end"]
        routes[name] = snapped_route(name, x0, y0, x1, y1)
    roof = base.load_building_roof_index(base.DEFAULT_STL)
    for name, df in routes.items():
        h = roof.query_many(df["x"].to_numpy(), df["y"].to_numpy())
        df["building_height_m"] = h
    return routes


def _pct(vals: np.ndarray, q: float) -> float:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def sample_case(case_dir: str, routes: dict[str, pd.DataFrame], masks: dict[int, np.ndarray]) -> tuple[pd.DataFrame, dict]:
    ts = pd.Timestamp(vis.parse_timestamp_from_cfd_dir(case_dir).replace("_", " "))
    ts = ts.tz_localize("UTC")
    bjt = ts.tz_convert(BJT)
    regime = L.regime_of(ts)
    case = os.path.basename(case_dir)
    rows = []
    along: dict[str, dict] = {}

    for height in HEIGHTS:
        csv_path = os.path.join(case_dir, "postProcessing", f"{height}m.csv")
        grid = L.load_cfd_grid(csv_path)
        inside = masks[height]
        fluid = grid["sampled"] & ~inside & L.interior_mask()
        v_cfd = grid["wind_speed"]
        v_wrf = L.wrf_speed_on_grid(case_dir, height)
        fluid_speed = v_cfd[fluid & np.isfinite(v_cfd)]
        domain_mean = float(np.mean(fluid_speed)) if fluid_speed.size else float("nan")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.full(v_cfd.shape, np.nan)
            ratio_p = np.full(v_cfd.shape, np.nan)
            usable = (
                fluid
                & np.isfinite(v_cfd)
                & np.isfinite(v_wrf)
                & (v_wrf >= L.WRF_SPEED_FLOOR)
            )
            ratio[usable] = v_cfd[usable] / v_wrf[usable]
            if np.isfinite(domain_mean) and domain_mean > 0:
                ratio_p[usable] = v_cfd[usable] / domain_mean
        sm_p = L.smooth_fluid(ratio_p, usable, L.BASE_SMOOTH_M)
        fast_mask, _ = L.label_zones(
            sm_p, L.BASE_THRESHOLD, L.BASE_MIN_AREA_M2, above=True
        )
        decel_mask, _ = L.label_zones(
            sm_p, L.DECEL_THRESHOLD, L.BASE_MIN_AREA_M2, above=False
        )

        for name, df in routes.items():
            iy = df["iy"].to_numpy()
            ix = df["ix"].to_numpy()
            cfd = v_cfd[iy, ix]
            wrf = v_wrf[iy, ix]
            sampled = grid["sampled"][iy, ix]
            solid = inside[iy, ix]
            flyable = sampled & ~solid & np.isfinite(cfd)
            lam_ok = flyable & np.isfinite(wrf) & (wrf >= L.WRF_SPEED_FLOOR)
            lam = np.full(len(df), np.nan)
            lam[lam_ok] = cfd[lam_ok] / wrf[lam_ok]
            in_fast = fast_mask[iy, ix] & flyable
            in_decel = decel_mask[iy, ix] & flyable
            n_fly = int(flyable.sum())
            n_lam = int(lam_ok.sum())
            cfd_f = cfd[flyable]
            wrf_f = wrf[flyable & np.isfinite(wrf)]
            cfd_l = cfd[lam_ok]
            wrf_l = wrf[lam_ok]
            mean_cfd = float(np.mean(cfd_f)) if n_fly else float("nan")
            mean_wrf = float(np.mean(wrf_f)) if wrf_f.size else float("nan")
            mean_cfd_l = float(np.mean(cfd_l)) if n_lam else float("nan")
            mean_wrf_l = float(np.mean(wrf_l)) if n_lam else float("nan")
            rows.append(
                {
                    "case": case,
                    "time_utc": ts.strftime("%Y-%m-%d %H:%M:%S%z"),
                    "time_bjt": bjt.strftime("%Y-%m-%d %H:%M:%S%z"),
                    "regime": regime,
                    "route": name,
                    "height_m": height,
                    "n_points": int(len(df)),
                    "n_flyable": n_fly,
                    "n_lambda": n_lam,
                    "ws_cfd_mean": mean_cfd,
                    "ws_wrf_mean": mean_wrf,
                    "lambda_of_means": (
                        mean_cfd_l / mean_wrf_l if mean_wrf_l and np.isfinite(mean_wrf_l) else float("nan")
                    ),
                    "frac_lambda_gt1": float(np.mean(lam[lam_ok] > 1.0)) if n_lam else float("nan"),
                    "frac_fast": float(in_fast[flyable].mean()) if n_fly else float("nan"),
                    "frac_decel": float(in_decel[flyable].mean()) if n_fly else float("nan"),
                    "p10_cfd": _pct(cfd_f, 10),
                    "p90_cfd": _pct(cfd_f, 90),
                    "p10_wrf": _pct(wrf_f, 10),
                    "p90_wrf": _pct(wrf_f, 90),
                    "p90_p10_cfd": (
                        _pct(cfd_f, 90) / _pct(cfd_f, 10)
                        if _pct(cfd_f, 10) > 0
                        else float("nan")
                    ),
                    "p90_p10_wrf": (
                        _pct(wrf_f, 90) / _pct(wrf_f, 10)
                        if _pct(wrf_f, 10) > 0
                        else float("nan")
                    ),
                }
            )
            slot = along.setdefault(
                name,
                {
                    "ws_cfd": np.full((len(HEIGHTS), len(df)), np.nan),
                    "ws_wrf": np.full((len(HEIGHTS), len(df)), np.nan),
                    "lam": np.full((len(HEIGHTS), len(df)), np.nan),
                },
            )
            k = HEIGHTS.index(height)
            slot["ws_cfd"][k] = np.where(flyable, cfd, np.nan)
            slot["ws_wrf"][k] = np.where(flyable & np.isfinite(wrf), wrf, np.nan)
            slot["lam"][k] = lam

    return pd.DataFrame(rows), along


def _worker(payload: tuple) -> str:
    case_dir, route_pkl, out_dir = payload
    routes = pd.read_pickle(route_pkl)
    masks = L.load_masks(L.MASK_NPZ)
    case = os.path.basename(case_dir)
    stats_path = Path(out_dir) / "hourly" / f"{case}.csv"
    if stats_path.is_file() and stats_path.stat().st_size > 0:
        return f"skip {case}"
    stats, along = sample_case(case_dir, routes, masks)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(stats_path, index=False)
    along_dir = Path(out_dir) / "along"
    along_dir.mkdir(parents=True, exist_ok=True)
    payload_np = {}
    for name, slot in along.items():
        payload_np[f"{name}__ws_cfd"] = slot["ws_cfd"]
        payload_np[f"{name}__ws_wrf"] = slot["ws_wrf"]
        payload_np[f"{name}__lam"] = slot["lam"]
    np.savez_compressed(along_dir / f"{case}.npz", **payload_np)
    return f"ok {case}"


def save_route_table(routes: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat(routes.values(), ignore_index=True)
    path = out_dir / "route_points.csv"
    df.to_csv(path, index=False)
    pkl = out_dir / "route_points.pkl"
    pd.to_pickle(routes, pkl)
    return pkl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--case", default=None, help="Single case directory name or path")
    args = ap.parse_args()

    routes = build_routes()
    pkl = save_route_table(routes, OUT_DIR)
    if args.case:
        case_dir = args.case
        if not os.path.isdir(case_dir):
            case_dir = str(CASE_ROOT / args.case)
        cases = [case_dir]
    else:
        cases = L.iter_target_cases(str(CASE_ROOT))
    if args.limit:
        cases = cases[: args.limit]
    print(f"[routes] " + ", ".join(f"{k}={len(v)}" for k, v in routes.items()), flush=True)
    print(f"[cases] {len(cases)} workers={args.workers}", flush=True)

    if args.workers <= 1 or len(cases) == 1:
        for c in cases:
            print(_worker((c, str(pkl), str(OUT_DIR))), flush=True)
        return 0

    jobs = [(c, str(pkl), str(OUT_DIR)) for c in cases]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            msg = fut.result()
            if done % 10 == 0 or msg.startswith("ok") and done <= 3:
                print(f"[{done}/{len(cases)}] {msg}", flush=True)
    print(f"[done] {done}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
