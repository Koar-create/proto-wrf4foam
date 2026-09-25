#!/usr/bin/env python3
"""Vector shear along the two routes from line probes and WRF nearest levels.

|dV/dz| uses a 40 m central difference at 30/60/120 m, matching the existing
along-track figures. Building solids and invalid probe points are dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (str(_HERE), str(_REPO / "util")):
    if p not in sys.path:
        sys.path.insert(0, p)

import uav_route_wind_shear_analysis as base  # noqa: E402
import visualize_WRF_CFD_xy_two_panel as vis  # noqa: E402

OUT = _REPO / "results" / "uav_route_exposure_hourly"
HEIGHTS = (30, 60, 120)
SHEAR_DZ = 40.0
PAIRS = {30: (10, 50), 60: (40, 80), 120: (100, 140)}


def _wrf_vectors(nc_path: Path, xs: np.ndarray, ys: np.ndarray, zs: tuple[float, ...]) -> dict[int, np.ndarray]:
    with xr.open_dataset(nc_path, mask_and_scale=False) as ds:
        x_coords = np.asarray(ds["x_rel"].values).squeeze().astype(float)
        y_coords = np.asarray(ds["y_rel"].values).squeeze().astype(float)
        z_coords = np.asarray(ds["z"].values).squeeze().astype(float)
        u_all = np.asarray(ds["U"].values).squeeze().astype(float)
        v_all = np.asarray(ds["V"].values).squeeze().astype(float)
        w_all = np.asarray(ds["W"].values).squeeze().astype(float) if "W" in ds else None
    pts = np.column_stack([ys, xs])
    out = {}
    for z in zs:
        zi = int(np.argmin(np.abs(z_coords - z)))
        fu = RegularGridInterpolator((y_coords, x_coords), u_all[zi], bounds_error=False, fill_value=np.nan)
        fv = RegularGridInterpolator((y_coords, x_coords), v_all[zi], bounds_error=False, fill_value=np.nan)
        u = fu(pts)
        v = fv(pts)
        if w_all is not None:
            fw = RegularGridInterpolator((y_coords, x_coords), w_all[zi], bounds_error=False, fill_value=np.nan)
            w = fw(pts)
        else:
            w = np.zeros_like(u)
        out[int(z)] = np.column_stack([u, v, w])
    return out


def _shear(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    d = (hi - lo) / SHEAR_DZ
    return np.sqrt(np.sum(d * d, axis=1))


def main() -> int:
    points = pd.read_csv(OUT / "probe_points.csv")
    hourly_files = sorted((OUT / "hourly").glob("*.csv"))
    rows = []
    along = {}
    z_order = [10, 40, 50, 80, 100, 140]
    n_per_z = points[points["z"] == 10].shape[0]

    for i, stats_path in enumerate(hourly_files, start=1):
        case = stats_path.stem
        probe_path = OUT / "probes" / f"{case}.csv"
        if not probe_path.is_file():
            print(f"[MISS] {case}", flush=True)
            continue
        probed = pd.read_csv(probe_path)
        if len(probed) != len(points):
            raise RuntimeError(f"{case}: probe rows {len(probed)} != points {len(points)}")
        probed = probed.copy()
        probed["route"] = points["route"].to_numpy()
        probed["seq"] = points["seq"].to_numpy()
        probed["z"] = points["z"].to_numpy()
        probed["building_height_m"] = points["building_height_m"].to_numpy()
        solid = probed["building_height_m"].to_numpy() >= probed["z"].to_numpy()
        valid = (probed["vtkValidPointMask"].to_numpy() == 1) & ~solid
        u = np.column_stack([probed["U:0"], probed["U:1"], probed["U:2"]]).astype(float)
        u[~valid] = np.nan

        ts = pd.Timestamp(vis.parse_timestamp_from_cfd_dir(case).replace("_", " "))
        nc = base.resolve_wrf_nc(ts)
        # WRF on the z=10 point order (route1 then route2); same xy at every z.
        xy = points[points["z"] == 10]
        wrf = _wrf_vectors(nc, xy["x"].to_numpy(), xy["y"].to_numpy(), tuple(z_order))

        by_z = {}
        for z in z_order:
            sl = probed["z"].to_numpy() == z
            by_z[z] = u[sl]

        for route, g in xy.groupby("route", sort=False):
            idx = g.index.to_numpy() - g.index.to_numpy()[0]
            # groupby index is the original index within z=10 block, which starts at 0
            # for the first route only. Use seq match instead.
        route_index = {name: xy.index[xy["route"] == name].to_numpy() for name in xy["route"].unique()}
        # xy index equals row position because z=10 is the first block.
        for name, ix in route_index.items():
            roof = xy.loc[ix, "building_height_m"].to_numpy()
            dist = xy.loc[ix, "distance_m"].to_numpy()
            for h, (z_lo, z_hi) in PAIRS.items():
                c_lo = by_z[z_lo][ix]
                c_hi = by_z[z_hi][ix]
                w_lo = wrf[z_lo][ix]
                w_hi = wrf[z_hi][ix]
                air = (roof < h) & np.isfinite(c_lo).all(axis=1) & np.isfinite(c_hi).all(axis=1)
                c_sh = _shear(c_lo, c_hi)
                w_sh = _shear(w_lo, w_hi)
                c_sh = np.where(air, c_sh, np.nan)
                w_sh = np.where(np.isfinite(w_sh).all() if False else np.isfinite(w_lo).all(axis=1) & np.isfinite(w_hi).all(axis=1), w_sh, np.nan)
                # WRF has no buildings; keep it on the same flyable mask so the comparison is the corridor, not the solid.
                w_sh = np.where(roof < h, w_sh, np.nan)
                finite_c = c_sh[np.isfinite(c_sh)]
                finite_w = w_sh[np.isfinite(w_sh)]
                rows.append(
                    {
                        "case": case,
                        "route": name,
                        "height_m": h,
                        "n_shear": int(np.isfinite(c_sh).sum()),
                        "shear_cfd_mean": float(np.mean(finite_c)) if finite_c.size else float("nan"),
                        "shear_cfd_p95": float(np.percentile(finite_c, 95)) if finite_c.size else float("nan"),
                        "shear_wrf_mean": float(np.mean(finite_w)) if finite_w.size else float("nan"),
                        "shear_wrf_p95": float(np.percentile(finite_w, 95)) if finite_w.size else float("nan"),
                    }
                )
                slot = along.setdefault(name, {})
                slot.setdefault("shear_cfd", {})
                slot.setdefault("shear_wrf", {})
                # stored after the loop via lists keyed by case — keep per-case npz below
                _ = dist
            along_case = OUT / "shear_along" / case
            along_case.mkdir(parents=True, exist_ok=True)
        # second pass writes arrays cleanly
        payload = {}
        for name, ix in route_index.items():
            roof = xy.loc[ix, "building_height_m"].to_numpy()
            c_stack = []
            w_stack = []
            for h, (z_lo, z_hi) in PAIRS.items():
                c_sh = _shear(by_z[z_lo][ix], by_z[z_hi][ix])
                w_sh = _shear(wrf[z_lo][ix], wrf[z_hi][ix])
                c_sh = np.where(roof < h, c_sh, np.nan)
                w_sh = np.where(roof < h, w_sh, np.nan)
                c_stack.append(c_sh)
                w_stack.append(w_sh)
            payload[f"{name}__shear_cfd"] = np.vstack(c_stack)
            payload[f"{name}__shear_wrf"] = np.vstack(w_stack)
        np.savez_compressed(OUT / "shear_along" / f"{case}.npz", **payload)
        if i % 20 == 0:
            print(f"[{i}/{len(hourly_files)}] {case}", flush=True)

    out = pd.DataFrame(rows)
    out_path = OUT / "shear_hourly.csv"
    out.to_csv(out_path, index=False)
    print(f"[CSV] {out_path} rows={len(out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
