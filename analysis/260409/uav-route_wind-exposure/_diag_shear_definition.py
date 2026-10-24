#!/usr/bin/env python3
"""Diagnostic: how much of the Section 3.1.2 vs Section 3.3 shear difference is definitional?

Reproduces the Section 3.3 route shear for one regular-weather hour
(2025-09-01 18:00 BJT) and re-evaluates the same profiles under three
operators, to separate (a) vector vs scalar differencing from
(b) the difference between a layer-mean gradient and a 40 m central difference.

Not part of the manuscript pipeline. Read-only with respect to results/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_XY = REPO / "analysis" / "260409" / "ws-xy-plane"
for p in (str(HERE), str(_XY)):
    if p not in sys.path:
        sys.path.insert(0, p)

import uav_route_wind_shear_analysis as base  # noqa: E402
import visualize_WRF_CFD_xy_two_panel as vis  # noqa: E402

OUT = REPO / "results" / "uav_route_exposure_hourly"
CASE = "20250901_1000_two_boundaries_as_outlet"
PROBE_LEVELS = (10, 40, 50, 80, 100, 140)
PAIRS = {30: (10, 50), 60: (40, 80), 120: (100, 140)}
DZ = 40.0


def wrf_vectors(nc_path: Path, xy: pd.DataFrame) -> dict[int, np.ndarray]:
    with xr.open_dataset(nc_path, mask_and_scale=False) as ds:
        xc = np.asarray(ds["x_rel"].values).squeeze().astype(float)
        yc = np.asarray(ds["y_rel"].values).squeeze().astype(float)
        zc = np.asarray(ds["z"].values).squeeze().astype(float)
        uu = np.asarray(ds["U"].values).squeeze().astype(float)
        vv = np.asarray(ds["V"].values).squeeze().astype(float)
        ww = np.asarray(ds["W"].values).squeeze().astype(float) if "W" in ds else None
    pts = np.column_stack([xy["y"].to_numpy(), xy["x"].to_numpy()])
    out = {}
    for z in PROBE_LEVELS:
        zi = int(np.argmin(np.abs(zc - z)))
        f = lambda a: RegularGridInterpolator(  # noqa: E731
            (yc, xc), a[zi], bounds_error=False, fill_value=np.nan
        )(pts)
        u, v = f(uu), f(vv)
        w = f(ww) if ww is not None else np.zeros_like(u)
        out[int(z)] = np.column_stack([u, v, w])
    return out


def op_vector_3d(lo, hi):
    return np.sqrt(np.sum(((hi - lo) / DZ) ** 2, axis=1))


def op_vector_horiz(lo, hi):
    return np.sqrt(np.sum(((hi[:, :2] - lo[:, :2]) / DZ) ** 2, axis=1))


def op_speed(lo, hi):
    sp_lo = np.sqrt((lo[:, :2] ** 2).sum(axis=1))
    sp_hi = np.sqrt((hi[:, :2] ** 2).sum(axis=1))
    return np.abs(sp_hi - sp_lo) / DZ


OPS = {
    "vector |dV|/dz (current 3.3)": op_vector_3d,
    "horizontal |d(U,V)|/dz": op_vector_horiz,
    "speed |d|Vh||/dz": op_speed,
}


def layer_mean_speed_shear(by_z, ix) -> float:
    """Section 3.1.2-style operator: mean |d(speed)/dz| over consecutive levels."""
    vals = []
    for z_lo, z_hi in zip(PROBE_LEVELS[:-1], PROBE_LEVELS[1:]):
        lo = by_z[z_lo][ix]
        hi = by_z[z_hi][ix]
        ok = np.isfinite(lo).all(axis=1) & np.isfinite(hi).all(axis=1)
        if not ok.any():
            continue
        sl = np.sqrt((lo[ok][:, :2] ** 2).sum(axis=1))
        sh = np.sqrt((hi[ok][:, :2] ** 2).sum(axis=1))
        vals.append(np.mean(np.abs(sh - sl) / (z_hi - z_lo)))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    points = pd.read_csv(OUT / "probe_points.csv")
    probed = pd.read_csv(OUT / "probes" / f"{CASE}.csv")
    probed = probed.copy()
    probed["route"] = points["route"].to_numpy()
    probed["z"] = points["z"].to_numpy()
    probed["building_height_m"] = points["building_height_m"].to_numpy()
    solid = probed["building_height_m"].to_numpy() >= probed["z"].to_numpy()
    valid = (probed["vtkValidPointMask"].to_numpy() == 1) & ~solid
    u = np.column_stack([probed["U:0"], probed["U:1"], probed["U:2"]]).astype(float)
    u[~valid] = np.nan
    by_z = {z: u[probed["z"].to_numpy() == z] for z in PROBE_LEVELS}

    ts = pd.Timestamp(vis.parse_timestamp_from_cfd_dir(CASE).replace("_", " "))
    xy = points[points["z"] == 10].reset_index(drop=True)
    wrf = wrf_vectors(base.resolve_wrf_nc(ts), xy)
    wrf_by_z = {z: wrf[z][xy.index.to_numpy()] for z in PROBE_LEVELS}

    print(f"case={CASE}  ts={ts}  n_points_per_level={len(xy)}")
    for route in xy["route"].unique():
        ix = xy.index[xy["route"] == route].to_numpy()
        roof = xy.loc[ix, "building_height_m"].to_numpy()
        print(f"\n=== {route} ===")
        for h, (z_lo, z_hi) in PAIRS.items():
            air = roof < h
            row = [f"z={h:>3} m  n_air={air.sum():>4}"]
            for label, fn in OPS.items():
                c = fn(by_z[z_lo][ix], by_z[z_hi][ix])
                w = fn(wrf_by_z[z_lo][ix], wrf_by_z[z_hi][ix])
                c, w = c[air], w[air]
                row.append(f"{label}: WRF={np.nanmean(w):.4f} CFD={np.nanmean(c):.4f} ratio={np.nanmean(c)/np.nanmean(w):.3f}")
            print("    " + "\n    ".join(row))
        lc = layer_mean_speed_shear(by_z, ix)
        lw = layer_mean_speed_shear(wrf_by_z, ix)
        print(f"    layer-mean speed shear over 10-140 m (3.1.2-style operator): "
              f"WRF={lw:.4f} CFD={lc:.4f} ratio={lc/lw:.3f}")
    return 0


MERGED = REPO / "data" / "260409" / "processed" / "merged_lidar_simulation_final.csv"
STATION_WINDOW = ("2025-09-01 11:00:00", "2025-09-01 16:00:00")
Z_MIN, Z_MAX = 52.0, 300.0


def _layer_mean(df: pd.DataFrame, ucol: str, vcol: str, wscol: str, vector: bool) -> float:
    h = df["Height"].to_numpy(float)
    o = np.argsort(h)
    h = h[o]
    u = df[ucol].to_numpy(float)[o]
    v = df[vcol].to_numpy(float)[o]
    ws = df[wscol].to_numpy(float)[o]
    ok = np.isfinite(h) & np.isfinite(ws)
    h, u, v, ws = h[ok], u[ok], v[ok], ws[ok]
    if len(h) < 2:
        return float("nan")
    dz = np.diff(h)
    if vector:
        du, dv = np.diff(u), np.diff(v)
        m = np.isfinite(du) & np.isfinite(dv) & (dz > 0)
        return float(np.mean(np.sqrt(du[m] ** 2 + dv[m] ** 2) / dz[m]))
    du = np.diff(ws)
    m = np.isfinite(du) & (dz > 0)
    return float(np.mean(np.abs(du[m]) / dz[m]))


def _central(df: pd.DataFrame, ucol, vcol, wscol, z_lo, z_hi, vector) -> float:
    def at(z, col):
        d = df.iloc[(df["Height"] - z).abs().argsort()[:1]]
        return float(d[col].iloc[0])

    dz = at(z_hi, "Height") - at(z_lo, "Height")
    if dz <= 0:
        return float("nan")
    if vector:
        du = at(z_hi, ucol) - at(z_lo, ucol)
        dv = at(z_hi, vcol) - at(z_lo, vcol)
        return float(np.sqrt(du**2 + dv**2) / dz)
    return float(abs(at(z_hi, wscol) - at(z_lo, wscol)) / dz)


def station_check() -> None:
    df = pd.read_csv(MERGED, parse_dates=["datetime"])
    t0, t1 = (pd.Timestamp(t) for t in STATION_WINDOW)
    sub = df[(df["datetime"] >= t0) & (df["datetime"] <= t1) & df["ws_obs"].notna()]
    print("\n\n=== Station check: Section 3.1.2 window, 2025-09-01 19:00-00:00 BJT ===")
    print("(levels start at 59.2 m; the paper's 52-300 m floor therefore begins there)\n")
    hdr = (f"{'station':<8} {'source':<5} {'layer|dws|':>10} {'layer|d(u,v)|':>13} "
           f"{'cd120|dws|':>10} {'cd120|d(u,v)|':>14} {'cd60|dws|':>9}")
    print(hdr)
    for site in sorted(sub["obtid"].unique()):
        for src in ("obs", "wrf", "cfd"):
            uc, vc, wc = f"u_{src}", f"v_{src}", f"ws_{src}"
            L_s, L_v, C_s, C_v, C60 = [], [], [], [], []
            for dt in sorted(sub["datetime"].unique()):
                g = sub[(sub["obtid"] == site) & (sub["datetime"] == dt)]
                low = g[(g["Height"] >= Z_MIN) & (g["Height"] <= Z_MAX)]
                if len(low) < 3:
                    continue
                L_s.append(_layer_mean(low, uc, vc, wc, vector=False))
                L_v.append(_layer_mean(low, uc, vc, wc, vector=True))
                C_s.append(_central(low, uc, vc, wc, 98.2, 137.2, vector=False))
                C_v.append(_central(low, uc, vc, wc, 98.2, 137.2, vector=True))
                C60.append(_central(low, uc, vc, wc, 59.2, 85.2, vector=False))
            f = lambda a: np.nanmean(a)  # noqa: E731
            print(f"{site:<8} {src:<5} {f(L_s):>10.4f} {f(L_v):>13.4f} "
                  f"{f(C_s):>10.4f} {f(C_v):>14.4f} {f(C60):>9.4f}")


if __name__ == "__main__":
    station_check()
    raise SystemExit(main())
