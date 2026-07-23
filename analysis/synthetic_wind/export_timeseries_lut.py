#!/usr/bin/env python3
"""Export synthetic HDF5 wind fields to a time-varying LUT for WindFieldPlugin.

Pipeline:
  1. Read synthetic_<case>.h5 (frame, 3, Nx, Ny, Nz) with non-uniform z levels.
  2. Vertically resample each frame to a uniform z grid (default z=0..200, dz=5).
  3. Spatially crop around a hotspot (default centre 1420, -880, half-width 600 m).
  4. Downsample frames (--out-dt) to control output size.
  5. Compute inside_building / valid_mask from buildings.stl; zero U indoors.
  6. Write per-frame VTI + wind_lut_timeseries.json manifest.

Reuses x-fastest VTK layout helpers from util/export_wind_lut_3d.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.synthetic_wind.config import (  # noqa: E402
    DX_M,
    OUTPUT_ROOT,
    SYNTH_FIELD_SUBDIR,
    dump_json,
)

DEFAULT_BUILDINGS_STL = (
    _REPO / "gazebo_wind_plugin" / "models" / "guangzhou_buildings" / "meshes" / "buildings.stl"
)
DEFAULT_OUT_ROOT = _REPO / "data" / "wind_lut_timeseries"


def _xyz_to_vtk(arr: np.ndarray) -> np.ndarray:
    """(Nx,Ny,Nz[,c]) -> VTK flat (x fastest)."""
    if arr.ndim == 3:
        return arr.transpose(2, 1, 0).reshape(-1)
    if arr.ndim == 4:
        return arr.transpose(2, 1, 0, 3).reshape(-1, arr.shape[3])
    raise ValueError(f"Unsupported array ndim={arr.ndim}")


def _building_mask(grid, buildings_stl: Path, method: str = "auto") -> np.ndarray:
    """Return flat uint8 mask (1 = inside building) for grid points."""
    import pyvista as pv

    bld = pv.read(str(buildings_stl))
    if not isinstance(bld, pv.PolyData):
        bld = bld.extract_surface()
    bld = bld.triangulate()
    npts = grid.n_points

    def _via_enclosed():
        if hasattr(grid, "select_interior_points"):
            try:
                sel = grid.select_interior_points(bld, check_surface=False)
            except TypeError:
                sel = grid.select_interior_points(bld)
        else:
            try:
                sel = grid.select_enclosed_points(bld, check_surface=False)
            except TypeError:
                sel = grid.select_enclosed_points(bld)
        flag_array = (
            sel["SelectedPoints"]
            if "SelectedPoints" in sel.array_names
            else sel["InteriorPoints"]
        )
        flag = np.asarray(flag_array, dtype=np.uint8)
        if flag.size != npts:
            raise RuntimeError(f"enclosed mask size {flag.size} != {npts}")
        return flag

    def _via_distance():
        out = grid.compute_implicit_distance(bld, inplace=False)
        return (np.asarray(out["implicit_distance"]) < 0).astype(np.uint8)

    if method == "enclosed":
        return _via_enclosed()
    if method == "distance":
        return _via_distance()
    try:
        return _via_enclosed()
    except Exception as exc:
        print(f"[mask] enclosed failed ({type(exc).__name__}); using distance fallback.")
        return _via_distance()


def _case_stamp(case_id: str) -> str:
    return case_id.split("_two_boundaries_as_outlet")[0]


def _crop_indices(
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    cx: float,
    cy: float,
    half_width: float,
) -> tuple[slice, slice, float, float, float, float]:
    x_lo, x_hi = cx - half_width, cx + half_width
    y_lo, y_hi = cy - half_width, cy + half_width
    ix = np.where((coords_x >= x_lo) & (coords_x <= x_hi))[0]
    iy = np.where((coords_y >= y_lo) & (coords_y <= y_hi))[0]
    if ix.size == 0 or iy.size == 0:
        raise ValueError(f"crop empty for centre=({cx},{cy}) half={half_width}")
    return (
        slice(int(ix[0]), int(ix[-1]) + 1),
        slice(int(iy[0]), int(iy[-1]) + 1),
        float(coords_x[ix[0]]),
        float(coords_x[ix[-1]]),
        float(coords_y[iy[0]]),
        float(coords_y[iy[-1]]),
    )


def _resample_z(
    field: np.ndarray,
    coords_z: np.ndarray,
    z_uniform: np.ndarray,
) -> np.ndarray:
    """Linear interpolation along z for field (3, Nx, Ny, Nz_src)."""
    z_ext = np.concatenate([[0.0], coords_z.astype(np.float64)])
    u_ext = np.zeros((3, field.shape[1], field.shape[2], z_ext.size), dtype=np.float64)
    u_ext[:, :, :, 1:] = field.astype(np.float64)
    out = np.zeros((3, field.shape[1], field.shape[2], z_uniform.size), dtype=np.float32)
    for k, zq in enumerate(z_uniform):
        if zq <= z_ext[0]:
            continue
        if zq >= z_ext[-1]:
            out[:, :, :, k] = field[:, :, :, -1]
            continue
        j = int(np.searchsorted(z_ext, zq, side="right") - 1)
        j = max(0, min(j, z_ext.size - 2))
        t = (zq - z_ext[j]) / (z_ext[j + 1] - z_ext[j])
        out[:, :, :, k] = (1.0 - t) * u_ext[:, :, :, j] + t * u_ext[:, :, :, j + 1]
    return out


def _frame_indices(time_s: np.ndarray, src_dt: float, out_dt: float, max_frames: int | None) -> np.ndarray:
    step = max(1, int(round(out_dt / src_dt)))
    idx = np.arange(0, time_s.size, step)
    if max_frames is not None and idx.size > max_frames:
        idx = idx[:max_frames]
    return idx


def export_timeseries_lut(
    case_id: str,
    out_dir: Path | None = None,
    buildings_stl: Path = DEFAULT_BUILDINGS_STL,
    hotspot_xy: tuple[float, float] = (1420.0, -880.0),
    crop_half_width: float = 600.0,
    z_min: float = 0.0,
    z_max: float = 200.0,
    dz: float = 5.0,
    out_dt: float = 2.0,
    max_frames: int | None = 300,
    loop: bool = True,
    frame_pattern: str = "frame_%04d.vti",
    building_mask_method: str = "auto",
) -> Path:
    """Convert one synthetic HDF5 case to a time-varying LUT directory."""
    import pyvista as pv

    src = OUTPUT_ROOT / SYNTH_FIELD_SUBDIR / f"synthetic_{case_id}.h5"
    if not src.is_file():
        raise FileNotFoundError(src)

    stamp = _case_stamp(case_id)
    out_dir = out_dir or (DEFAULT_OUT_ROOT / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    z_uniform = np.arange(z_min, z_max + 0.5 * dz, dz, dtype=np.float64)
    n_z = z_uniform.size

    with h5py.File(src, "r") as f:
        coords_x = f["coords_x"][:].astype(np.float64)
        coords_y = f["coords_y"][:].astype(np.float64)
        coords_z = f["coords_z"][:].astype(np.float64)
        time_s = f["time_s"][:].astype(np.float64)
        src_dt = float(time_s[1] - time_s[0]) if time_s.size > 1 else 0.5
        frame_idx = _frame_indices(time_s, src_dt, out_dt, max_frames)
        n_frames = frame_idx.size

        sx, sy = hotspot_xy
        x_sl, y_sl, x_lo, x_hi, y_lo, y_hi = _crop_indices(
            coords_x, coords_y, sx, sy, crop_half_width,
        )
        nx = x_sl.stop - x_sl.start
        ny = y_sl.stop - y_sl.start
        origin = [x_lo, y_lo, float(z_uniform[0])]
        spacing = [float(DX_M), float(DX_M), float(dz)]
        dimensions = [nx, ny, n_z]

        grid = pv.ImageData(
            dimensions=(nx, ny, n_z),
            spacing=(spacing[0], spacing[1], spacing[2]),
            origin=(origin[0], origin[1], origin[2]),
        )
        inside_flat = _building_mask(grid, buildings_stl, method=building_mask_method)
        inside = inside_flat.reshape(n_z, ny, nx).transpose(2, 1, 0).astype(np.uint8)
        valid = np.ones((nx, ny, n_z), dtype=np.uint8)

        print(
            f"[export] case={case_id} crop=({nx}x{ny}x{n_z}) "
            f"frames={n_frames} out_dt={out_dt}s inside={int(inside.sum())}",
            flush=True,
        )

        t0 = time.time()
        for out_i, src_i in enumerate(frame_idx):
            u_src = f["U"][src_i, :, x_sl, y_sl, :]  # (3, nx, ny, nz_src)
            u_eq = _resample_z(u_src, coords_z, z_uniform)  # (3, nx, ny, n_z)
            u_xyz = np.transpose(u_eq, (1, 2, 3, 0))  # (nx, ny, n_z, 3)
            u_xyz[inside != 0] = 0.0

            out_grid = grid.copy()
            out_grid["U"] = _xyz_to_vtk(u_xyz)
            out_grid["inside_building"] = _xyz_to_vtk(inside)
            out_grid["valid_mask"] = _xyz_to_vtk(valid)
            out_grid.set_active_vectors("U")

            vti_name = frame_pattern % out_i
            out_grid.save(str(out_dir / vti_name))

            if out_i == 0 or (out_i + 1) % 50 == 0 or out_i + 1 == n_frames:
                elapsed = time.time() - t0
                print(f"  frame {out_i + 1}/{n_frames} (src t={time_s[src_i]:.1f}s) [{elapsed:.1f}s]", flush=True)

    manifest = {
        "case_id": case_id,
        "source_h5": str(src.relative_to(_REPO)).replace("\\", "/"),
        "origin": origin,
        "spacing": spacing,
        "dimensions": dimensions,
        "dt_s": float(out_dt),
        "n_frames": int(n_frames),
        "frame_pattern": frame_pattern,
        "loop": 1 if loop else 0,
        "crop_bbox": {
            "x_lo": x_lo, "x_hi": x_hi,
            "y_lo": y_lo, "y_hi": y_hi,
            "z_lo": float(z_uniform[0]), "z_hi": float(z_uniform[-1]),
        },
        "hotspot_xy": [sx, sy],
        "masks": {
            "n_inside_building": int(inside.sum()),
            "n_valid": int(valid.sum()),
        },
        "units": {"length": "m", "velocity": "m/s", "time": "s"},
        "produced_by": "analysis/synthetic_wind/export_timeseries_lut.py",
    }
    manifest_path = out_dir / "wind_lut_timeseries.json"
    dump_json(manifest_path, manifest)
    print(f"[export] wrote {n_frames} VTI + {manifest_path}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export synthetic HDF5 to time-varying wind LUT.")
    parser.add_argument("--case", required=True, help="Case ID (e.g. 20250903_1400_two_boundaries_as_outlet)")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--buildings-stl", type=Path, default=DEFAULT_BUILDINGS_STL)
    parser.add_argument("--hotspot-xy", type=str, default="1420,-880")
    parser.add_argument("--crop-half-width", type=float, default=600.0)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=200.0)
    parser.add_argument("--dz", type=float, default=5.0)
    parser.add_argument("--out-dt", type=float, default=2.0, help="Output frame interval (s)")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--building-mask-method", choices=["enclosed", "distance", "auto"], default="auto")
    args = parser.parse_args()

    hx, hy = (float(v) for v in args.hotspot_xy.split(","))
    export_timeseries_lut(
        case_id=args.case,
        out_dir=args.out_dir,
        buildings_stl=args.buildings_stl,
        hotspot_xy=(hx, hy),
        crop_half_width=args.crop_half_width,
        z_min=args.z_min,
        z_max=args.z_max,
        dz=args.dz,
        out_dt=args.out_dt,
        max_frames=args.max_frames,
        loop=not args.no_loop,
        building_mask_method=args.building_mask_method,
    )


if __name__ == "__main__":
    main()
