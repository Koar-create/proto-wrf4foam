"""M0.2: interpolate VTP cutting planes to a regular grid and apply the STL mask."""
from __future__ import annotations

import time

import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from lesio import inner_axes, read_vtp_points_U, vtp_path
from paths import CACHE, INNER_DX, INNER_LIM, TIMES


def roof_grid(roof, xs, ys, dest=None):
    dest = dest or (CACHE / "roof_inner10.npy")
    if dest.is_file():
        return np.load(dest)
    print(f"[slices] querying roof on {ys.size}×{xs.size} grid …", flush=True)
    t0 = time.time()
    X, Y = np.meshgrid(xs, ys)
    h = roof.query_many(X.ravel(), Y.ravel()).reshape(ys.size, xs.size)
    np.save(dest, h.astype(np.float32))
    print(f"[slices] roof grid in {time.time()-t0:.1f}s", flush=True)
    return h


def _delaunay_xy(pts):
    return Delaunay(pts[:, :2].astype(np.float64))


def interpolate_horizontal(z: int, roof, times=TIMES, dx=INNER_DX, lim=INNER_LIM, force=False):
    dest = CACHE / f"slices_z{int(z)}.nc"
    if dest.is_file() and not force:
        return xr.open_dataset(dest)
    xs, ys = inner_axes(dx, lim)
    X, Y = np.meshgrid(xs, ys)
    tgt = np.column_stack([X.ravel(), Y.ravel()])
    h = roof_grid(roof, xs, ys)
    solid = h >= float(z)

    t0 = times[0]
    pts, U0 = read_vtp_points_U(vtp_path(z, t0))
    print(f"[slices] z={z} Delaunay n={len(pts)} …", flush=True)
    tri = _delaunay_xy(pts)
    nt = len(times)
    ny, nx = ys.size, xs.size
    U = np.full((nt, ny, nx, 3), np.nan, np.float32)

    for it, t in enumerate(times):
        tt = time.time()
        if it == 0:
            vec = U0
        else:
            _, vec = read_vtp_points_U(vtp_path(z, t))
        for c in range(3):
            interp = LinearNDInterpolator(tri, vec[:, c].astype(np.float64), fill_value=np.nan)
            g = interp(tgt).reshape(ny, nx)
            g[solid] = np.nan
            U[it, ..., c] = g
        print(f"  t={t:4d}  {time.time()-tt:.1f}s", flush=True)

    ds = xr.Dataset(
        data_vars=dict(
            u=(("t", "y", "x"), U[..., 0]),
            v=(("t", "y", "x"), U[..., 1]),
            w=(("t", "y", "x"), U[..., 2]),
            solid=(("y", "x"), solid),
            roof=(("y", "x"), h.astype(np.float32)),
        ),
        coords=dict(t=("t", np.asarray(times, float)), x=("x", xs), y=("y", ys)),
        attrs=dict(z=int(z), dx=float(dx), note="linear Delaunay; STL roof>=z masked"),
    )
    ds.to_netcdf(dest)
    print(f"[slices] wrote {dest}", flush=True)
    return ds


def interpolate_y800(roof, times=TIMES, dx=10.0, dz=10.0, force=False):
    dest = CACHE / "slices_y800.nc"
    if dest.is_file() and not force:
        return xr.open_dataset(dest)
    xs = np.arange(-INNER_LIM, INNER_LIM + 0.5 * dx, dx)
    zs = np.arange(0.0, 2000.0 + 0.5 * dz, dz)
    X, Z = np.meshgrid(xs, zs)
    tgt = np.column_stack([X.ravel(), Z.ravel()])
    roof_x = roof.query_many(xs, np.full_like(xs, 800.0))
    solid = Z <= roof_x[None, :]

    pts, U0 = read_vtp_points_U(vtp_path("y800", times[0]))
    tri = Delaunay(np.column_stack([pts[:, 0], pts[:, 2]]).astype(np.float64))
    nt, nz, nx = len(times), zs.size, xs.size
    U = np.full((nt, nz, nx, 3), np.nan, np.float32)
    for it, t in enumerate(times):
        vec = U0 if it == 0 else read_vtp_points_U(vtp_path("y800", t))[1]
        for c in range(3):
            interp = LinearNDInterpolator(tri, vec[:, c].astype(np.float64), fill_value=np.nan)
            g = interp(tgt).reshape(nz, nx)
            g[solid] = np.nan
            U[it, ..., c] = g
        print(f"  y800 t={t}", flush=True)
    ds = xr.Dataset(
        data_vars=dict(
            u=(("t", "z", "x"), U[..., 0]),
            v=(("t", "z", "x"), U[..., 1]),
            w=(("t", "z", "x"), U[..., 2]),
            solid=(("z", "x"), solid),
        ),
        coords=dict(t=("t", np.asarray(times, float)), x=("x", xs), z=("z", zs)),
        attrs=dict(y=800.0),
    )
    ds.to_netcdf(dest)
    return ds
