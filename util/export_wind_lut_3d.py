#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export a 3D wind look-up table (u, v, w on a regular grid) for one OpenFOAM
steady frame, intended as input to the next task: a Gazebo wind-field plugin.

Default frame: steady_experiments_finer_ABL/20250903_1400_two_boundaries_as_outlet
               (UTC 2025-09-03 14:00 = local 22:00 LST, nocturnal stable ABL).

Pipeline (mirrors util/render_3d_microhazard_pyvista.py for the reading step):
  1. POpenFOAMReader -> latest time -> internal block -> cell_data_to_point_data.
  2. Build a regular pv.ImageData (default 501 x 501 x 101 covering 5 km x 5 km
     x 500 m, dx=dy=10 m, dz=5 m) and probe the unstructured field onto it.
  3. Compute an inside-building mask from constant/triSurface/buildings.stl
     (select_enclosed_points; falls back to compute_implicit_distance < 0).
  4. Force U = 0 inside buildings (matches simpleFoam no-slip wall semantics
     and gives smooth trilinear taper to zero at building surfaces).
  5. Save:
       data/wind_lut/<case-stamp>/wind_lut.vti      (VTK XML ImageData)
       data/wind_lut/<case-stamp>/wind_lut.npz      (Python LUT)
       data/wind_lut/<case-stamp>/wind_lut.json     (metadata + interface contract)
  6. QC plots:
       results/wind_lut/<case-stamp>/qc_slice_z100m.png       (LUT vs 100m.csv)
       results/wind_lut/<case-stamp>/qc_profile_hotspot.png   ((u,v,w,|U|) vs z)
       results/wind_lut/<case-stamp>/qc_summary.json

Index convention in the on-disk arrays
--------------------------------------
NPZ U has shape (Nx, Ny, Nz, 3) with axis 0 = x (east), 1 = y (north),
2 = z (up). x_coords/y_coords/z_coords are the 1-D node coordinates so that
U[i,j,k,:] corresponds to position (x_coords[i], y_coords[j], z_coords[k]).
The .vti file is a standard VTK ImageData with origin/spacing/dimensions
matching the .npz; vtkXMLImageDataReader on the C++ side recovers the same
geometry without any reshape gymnastics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export 3D wind LUT (u,v,w) on regular grid for a single OpenFOAM frame."
    )
    p.add_argument(
        "--case-dir",
        type=Path,
        default=_repo_root()
        / "steady_experiments_finer_ABL"
        / "20250903_1400_two_boundaries_as_outlet",
        help="OpenFOAM case directory (must contain myExpxx.foam and time-step folders).",
    )
    p.add_argument(
        "--buildings-stl",
        type=Path,
        default=_repo_root() / "constant" / "triSurface" / "buildings.stl",
        help="Path to buildings STL used to derive the inside-building mask.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for LUT artefacts. Default: data/wind_lut/<case-stamp>/",
    )
    p.add_argument(
        "--qc-dir",
        type=Path,
        default=None,
        help="Output directory for QC plots. Default: results/wind_lut/<case-stamp>/",
    )
    p.add_argument(
        "--xrange",
        type=str,
        default="-2500,2500",
        help="Comma-separated x_min,x_max in metres (LUT extent).",
    )
    p.add_argument(
        "--yrange",
        type=str,
        default="-2500,2500",
        help="Comma-separated y_min,y_max in metres.",
    )
    p.add_argument(
        "--zrange",
        type=str,
        default="0,500",
        help="Comma-separated z_min,z_max in metres.",
    )
    p.add_argument("--dx", type=float, default=10.0, help="Grid spacing in x (m).")
    p.add_argument("--dy", type=float, default=10.0, help="Grid spacing in y (m).")
    p.add_argument("--dz", type=float, default=5.0, help="Grid spacing in z (m).")
    p.add_argument(
        "--time-value",
        type=str,
        default="last",
        help="Either 'last' (default) or a specific OpenFOAM time value (e.g. '5000').",
    )
    p.add_argument(
        "--no-vti",
        action="store_true",
        help="Skip writing the .vti file (keep .npz + .json).",
    )
    p.add_argument(
        "--no-npz",
        action="store_true",
        help="Skip writing the .npz file (keep .vti + .json).",
    )
    p.add_argument(
        "--no-qc",
        action="store_true",
        help="Skip QC plots / qc_summary.json.",
    )
    p.add_argument(
        "--hotspot-xy",
        type=str,
        default="1416,-881",
        help="Comma-separated (x0,y0) in metres for the QC vertical profile. "
        "Default matches AUTO-CHECKPOINT.md hotspot for 20250903_1400.",
    )
    p.add_argument(
        "--qc-slice-z",
        type=float,
        default=100.0,
        help="z (m) for the QC horizontal slice (matches postProcessing/100m.csv when 100).",
    )
    p.add_argument(
        "--building-mask-method",
        type=str,
        choices=["enclosed", "distance", "auto"],
        default="auto",
        help="'enclosed' = select_enclosed_points; 'distance' = compute_implicit_distance<0; "
        "'auto' tries enclosed first and falls back on failure.",
    )
    return p.parse_args()


def _parse_pair(spec: str) -> Tuple[float, float]:
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected 'a,b' got '{spec}'")
    return float(parts[0]), float(parts[1])


def _case_stamp(case_dir: Path) -> str:
    """Use the leading YYYYMMDD_HHMM token if present, else the dir name."""
    name = case_dir.name
    head = name.split("_two_boundaries_as_outlet")[0]
    return head if head else name


def _utc_to_lst(utc_dt: datetime, tz_offset_hours: int = 8) -> datetime:
    return utc_dt.astimezone(timezone(timedelta(hours=tz_offset_hours)))


def _parse_time_from_case(case_dir: Path) -> Tuple[str, str]:
    """Return (time_utc_iso, time_lst_iso) parsed from leading YYYYMMDD_HHMM."""
    stamp = _case_stamp(case_dir)
    try:
        utc_dt = datetime.strptime(stamp[:13], "%Y%m%d_%H%M").replace(tzinfo=timezone.utc)
    except Exception:
        return ("", "")
    lst_dt = _utc_to_lst(utc_dt, tz_offset_hours=8)
    return (utc_dt.isoformat(), lst_dt.isoformat())


def _pick_internal_block(mb):
    """Mirror util/render_3d_microhazard_pyvista.py: resolve 'internalMesh' block."""
    import pyvista as pv

    if isinstance(mb, pv.PolyData):
        return mb
    for i in range(mb.n_blocks):
        b = mb.get_block(i)
        if b is None:
            continue
        name = (mb.get_block_name(i) or "").lower()
        if "internal" in name:
            return b
    for i in range(mb.n_blocks):
        b = mb.get_block(i)
        if b is not None and b.n_cells > 0:
            return b
    raise RuntimeError("Could not find internal mesh block in OpenFOAM reader output.")


def _read_internal_field(case_dir: Path, time_value: str):
    import pyvista as pv

    foam = case_dir / "myExpxx.foam"
    if not foam.is_file():
        # POpenFOAMReader needs *.foam marker; try generic one if missing
        for cand in case_dir.glob("*.foam"):
            foam = cand
            break
    if not foam.is_file():
        raise FileNotFoundError(f"No .foam marker file in {case_dir}")

    reader = pv.POpenFOAMReader(str(foam))
    tvals = list(reader.time_values)
    if not tvals:
        raise RuntimeError("OpenFOAM reader returned no time values.")
    if time_value == "last":
        chosen = float(tvals[-1])
    else:
        chosen = float(time_value)
        if chosen not in tvals:
            # snap to nearest
            chosen = float(min(tvals, key=lambda t: abs(t - chosen)))
    reader.set_active_time_value(chosen)
    mb = reader.read()
    internal = _pick_internal_block(mb).copy(deep=True)

    if "U" in internal.array_names and "U" in internal.cell_data:
        internal = internal.cell_data_to_point_data(pass_cell_data=False)

    if "U" not in internal.array_names:
        raise RuntimeError("Field 'U' not present in internal block after read.")

    return internal, chosen


def _build_image_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    dx: float,
    dy: float,
    dz: float,
):
    import pyvista as pv

    Nx = int(round((xmax - xmin) / dx)) + 1
    Ny = int(round((ymax - ymin) / dy)) + 1
    Nz = int(round((zmax - zmin) / dz)) + 1

    grid = pv.ImageData(
        dimensions=(Nx, Ny, Nz),
        spacing=(float(dx), float(dy), float(dz)),
        origin=(float(xmin), float(ymin), float(zmin)),
    )
    return grid, Nx, Ny, Nz


def _vtk_to_xyz(arr_flat: np.ndarray, Nx: int, Ny: int, Nz: int) -> np.ndarray:
    """Reshape VTK point-ordered flat array (x fastest, then y, then z) into (Nx,Ny,Nz[,...])."""
    if arr_flat.ndim == 1:
        return arr_flat.reshape(Nz, Ny, Nx).transpose(2, 1, 0)
    if arr_flat.ndim == 2:
        c = arr_flat.shape[1]
        return arr_flat.reshape(Nz, Ny, Nx, c).transpose(2, 1, 0, 3)
    raise ValueError(f"Unsupported array ndim={arr_flat.ndim}")


def _xyz_to_vtk(arr: np.ndarray) -> np.ndarray:
    """Inverse of _vtk_to_xyz: (Nx,Ny,Nz[,...]) -> flat (Nz*Ny*Nx[, ...])."""
    if arr.ndim == 3:
        return arr.transpose(2, 1, 0).reshape(-1)
    if arr.ndim == 4:
        c = arr.shape[3]
        return arr.transpose(2, 1, 0, 3).reshape(-1, c)
    raise ValueError(f"Unsupported array ndim={arr.ndim}")


def _building_mask(grid, buildings_stl: Path, method: str) -> np.ndarray:
    """Return a flat uint8 mask (length = grid.n_points) of points inside buildings."""
    import pyvista as pv

    bld = pv.read(str(buildings_stl))
    if not isinstance(bld, pv.PolyData):
        bld = bld.extract_surface()
    bld = bld.triangulate()

    npts = grid.n_points

    def _via_enclosed():
        if hasattr(grid, "select_interior_points"):
            sel = grid.select_interior_points(bld, tolerance=0.0, check_surface=False)
        else:
            sel = grid.select_enclosed_points(bld, tolerance=0.0, check_surface=False)
        flag_array = sel["SelectedPoints"] if "SelectedPoints" in sel.array_names \
            else sel["InteriorPoints"]
        flag = np.asarray(flag_array, dtype=np.uint8)
        if flag.size != npts:
            raise RuntimeError(
                f"interior-point filter returned {flag.size} flags, expected {npts}"
            )
        return flag

    def _via_distance():
        # Negative distance = inside the closed surface.
        out = grid.compute_implicit_distance(bld, inplace=False)
        d = np.asarray(out["implicit_distance"])
        return (d < 0).astype(np.uint8)

    if method == "enclosed":
        return _via_enclosed()
    if method == "distance":
        return _via_distance()
    # auto
    try:
        return _via_enclosed()
    except Exception as e:
        print(f"[building-mask] enclosed failed ({type(e).__name__}: {e}); using distance fallback.")
        return _via_distance()


def _sample_field(grid, internal):
    """Probe unstructured CFD field onto the regular grid; returns sampled grid."""
    sampled = grid.sample(internal)
    return sampled


def _print_step(msg: str) -> None:
    print(f"[wind-lut] {msg}")


def main() -> None:
    args = _parse_args()
    root = _repo_root()
    case_dir = args.case_dir if args.case_dir.is_absolute() else (root / args.case_dir)
    buildings = (
        args.buildings_stl if args.buildings_stl.is_absolute() else (root / args.buildings_stl)
    )

    stamp = _case_stamp(case_dir)
    out_dir = args.out_dir if args.out_dir is not None else (root / "data" / "wind_lut" / stamp)
    qc_dir = args.qc_dir if args.qc_dir is not None else (root / "results" / "wind_lut" / stamp)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    if not qc_dir.is_absolute():
        qc_dir = root / qc_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    if not case_dir.is_dir():
        print(f"ERROR: case dir not found: {case_dir}", file=sys.stderr)
        sys.exit(1)
    if not buildings.is_file():
        print(f"ERROR: buildings STL not found: {buildings}", file=sys.stderr)
        sys.exit(1)

    xmin, xmax = _parse_pair(args.xrange)
    ymin, ymax = _parse_pair(args.yrange)
    zmin, zmax = _parse_pair(args.zrange)

    _print_step("=" * 72)
    _print_step(f"case_dir       : {case_dir}")
    _print_step(f"buildings_stl  : {buildings}")
    _print_step(f"LUT extent (m) : x=[{xmin},{xmax}] y=[{ymin},{ymax}] z=[{zmin},{zmax}]")
    _print_step(f"LUT spacing(m) : dx={args.dx} dy={args.dy} dz={args.dz}")
    _print_step("=" * 72)

    t0 = time.time()
    _print_step("[1/5] reading OpenFOAM case ...")
    import pyvista as pv  # noqa: F401  (imported here so failure is reported clearly)

    internal, chosen_t = _read_internal_field(case_dir, args.time_value)
    _print_step(
        f"  internal: n_points={internal.n_points:,} n_cells={internal.n_cells:,} "
        f"time={chosen_t:g}"
    )

    _print_step("[2/5] building regular ImageData grid ...")
    grid, Nx, Ny, Nz = _build_image_grid(
        xmin, xmax, ymin, ymax, zmin, zmax, args.dx, args.dy, args.dz
    )
    _print_step(f"  dims : Nx={Nx} Ny={Ny} Nz={Nz}  total={Nx*Ny*Nz:,} points")

    _print_step("[3/5] sampling CFD field onto grid (vtkProbeFilter) ...")
    sampled = _sample_field(grid, internal)
    flat_U = np.asarray(sampled["U"], dtype=np.float32)
    flat_valid = np.asarray(
        sampled["vtkValidPointMask"] if "vtkValidPointMask" in sampled.array_names
        else np.ones(sampled.n_points, dtype=np.uint8),
        dtype=np.uint8,
    )
    if flat_U.ndim != 2 or flat_U.shape[1] < 3:
        raise RuntimeError(f"Unexpected sampled U shape: {flat_U.shape}")
    flat_U = flat_U[:, :3]

    U = _vtk_to_xyz(flat_U, Nx, Ny, Nz)  # (Nx,Ny,Nz,3)
    valid = _vtk_to_xyz(flat_valid, Nx, Ny, Nz)  # (Nx,Ny,Nz)
    _print_step(
        f"  sampled U range: u=[{U[..., 0].min():.3f},{U[..., 0].max():.3f}] "
        f"v=[{U[..., 1].min():.3f},{U[..., 1].max():.3f}] "
        f"w=[{U[..., 2].min():.3f},{U[..., 2].max():.3f}]  "
        f"valid={int(valid.sum()):,}/{valid.size:,}"
    )

    _print_step("[4/5] computing inside-building mask ...")
    flat_inside = _building_mask(grid, buildings, args.building_mask_method)
    inside = _vtk_to_xyz(flat_inside, Nx, Ny, Nz).astype(np.uint8)
    n_inside = int(inside.sum())
    _print_step(f"  inside_building points: {n_inside:,} ({100.0 * n_inside / inside.size:.2f}%)")

    # Force U=0 inside buildings (no-slip wall semantics; smooth taper for trilinear interp).
    U[inside.astype(bool), :] = 0.0

    _print_step("[5/5] writing LUT artefacts ...")
    x_coords = (np.arange(Nx, dtype=np.float64) * args.dx + xmin).astype(np.float32)
    y_coords = (np.arange(Ny, dtype=np.float64) * args.dy + ymin).astype(np.float32)
    z_coords = (np.arange(Nz, dtype=np.float64) * args.dz + zmin).astype(np.float32)

    time_utc_iso, time_lst_iso = _parse_time_from_case(case_dir)
    meta = {
        "case_id": case_dir.name,
        "case_dir_relative": str(case_dir.relative_to(root)).replace("\\", "/"),
        "time_value_used": float(chosen_t),
        "time_utc": time_utc_iso,
        "time_lst": time_lst_iso,
        "lst_offset_hours": 8,
        "solver": "simpleFoam",
        "turbulence_model": "k-epsilon (RANS)",
        "origin": [float(xmin), float(ymin), float(zmin)],
        "spacing": [float(args.dx), float(args.dy), float(args.dz)],
        "dimensions": [int(Nx), int(Ny), int(Nz)],
        "domain": {
            "x_min": float(xmin), "x_max": float(xmax),
            "y_min": float(ymin), "y_max": float(ymax),
            "z_min": float(zmin), "z_max": float(zmax),
        },
        "units": {"length": "m", "velocity": "m/s"},
        "axes": {
            "0": "x_east_m",
            "1": "y_north_m",
            "2": "z_up_m",
            "U_components": ["u_east", "v_north", "w_up"],
        },
        "array_layout": {
            "npz_U_shape": [int(Nx), int(Ny), int(Nz), 3],
            "vtk_point_order": "x_fastest_then_y_then_z (standard VTK ImageData)",
        },
        "masks": {
            "inside_building_semantics": "U is forced to 0 (no-slip wall); plugin may "
            "either return 0 wind or treat the cell as solid for collision.",
            "valid_mask_semantics": "1 = grid point fell inside the unstructured CFD "
            "domain during sampling; 0 = outside (rare here because z_max <= 500 m "
            "<< 2000 m domain top). Plugin can use this as a conservative gate.",
            "n_inside_building": int(n_inside),
            "n_invalid": int((valid == 0).sum()),
        },
        "interface_for_gazebo_plugin": {
            "lookup": "Trilinear-interpolate U at drone position (x,y,z) in metres "
            "expressed in the OpenFOAM domain frame (origin = 5km core centre).",
            "out_of_range_policy": "If position is outside [origin, origin + spacing*(dims-1)] "
            "the plugin must decide (zero wind or freestream extrapolation); the LUT itself "
            "does not extrapolate.",
            "force_law": "Apply F = 0.5 * rho * C_D * A * |U_rel| * U_rel with U_rel = U_wind - U_drone.",
        },
        "produced_by": "util/export_wind_lut_3d.py",
    }

    if not args.no_vti:
        # Re-pack arrays back into VTK point order on the same grid.
        out_grid = grid.copy()
        out_grid["U"] = _xyz_to_vtk(U)
        out_grid["inside_building"] = _xyz_to_vtk(inside)
        out_grid["valid_mask"] = _xyz_to_vtk(valid)
        out_grid.set_active_vectors("U")
        vti_path = out_dir / "wind_lut.vti"
        out_grid.save(str(vti_path))
        _print_step(f"  wrote {vti_path}  ({vti_path.stat().st_size/1e6:.1f} MB)")

    if not args.no_npz:
        npz_path = out_dir / "wind_lut.npz"
        np.savez_compressed(
            npz_path,
            U=U,
            inside_building=inside,
            valid_mask=valid,
            x_coords=x_coords,
            y_coords=y_coords,
            z_coords=z_coords,
        )
        _print_step(f"  wrote {npz_path}  ({npz_path.stat().st_size/1e6:.1f} MB)")

    json_path = out_dir / "wind_lut.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    _print_step(f"  wrote {json_path}")

    if not args.no_qc:
        _print_step("[QC] generating sanity plots / summary ...")
        try:
            x0_qc, y0_qc = _parse_pair(args.hotspot_xy)
        except Exception:
            x0_qc, y0_qc = 0.0, 0.0
        _make_qc(
            U=U,
            inside=inside,
            valid=valid,
            x_coords=x_coords,
            y_coords=y_coords,
            z_coords=z_coords,
            qc_dir=qc_dir,
            case_dir=case_dir,
            qc_slice_z=float(args.qc_slice_z),
            x0=float(x0_qc),
            y0=float(y0_qc),
            time_lst=time_lst_iso,
        )

    _print_step(f"DONE in {time.time() - t0:.1f}s -> {out_dir}")


def _make_qc(
    *,
    U: np.ndarray,
    inside: np.ndarray,
    valid: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    qc_dir: Path,
    case_dir: Path,
    qc_slice_z: float,
    x0: float,
    y0: float,
    time_lst: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    Nx, Ny, Nz = U.shape[0], U.shape[1], U.shape[2]
    Uh_lut = np.sqrt(U[..., 0] ** 2 + U[..., 1] ** 2)  # (Nx, Ny, Nz)
    Umag_lut = np.linalg.norm(U[..., :3], axis=-1)

    # ---- (a) horizontal slice at z = qc_slice_z ---------------------------
    k = int(np.argmin(np.abs(z_coords - qc_slice_z)))
    z_used = float(z_coords[k])
    slice_uh = Uh_lut[:, :, k]  # (Nx, Ny)
    slice_inside = inside[:, :, k]

    # Build CFD reference scatter from postProcessing/100m.csv (only when QC level is 100m).
    csv_100 = case_dir / "postProcessing" / "100m.csv"
    cfd_x = cfd_y = cfd_w = None
    if csv_100.is_file() and abs(z_used - 100.0) < 1e-3:
        xs, ys, ws = [], [], []
        for ch in pd.read_csv(csv_100, chunksize=200_000):
            x = ch["Coords:0"].astype(np.float64).to_numpy()
            y = ch["Coords:1"].astype(np.float64).to_numpy()
            u0 = ch["U:0"].astype(np.float64).to_numpy()
            u1 = ch["U:1"].astype(np.float64).to_numpy()
            mask = (x >= x_coords[0]) & (x <= x_coords[-1]) & (y >= y_coords[0]) & (y <= y_coords[-1])
            xs.append(x[mask])
            ys.append(y[mask])
            ws.append(np.sqrt(u0[mask] ** 2 + u1[mask] ** 2))
        cfd_x = np.concatenate(xs) if xs else np.empty(0)
        cfd_y = np.concatenate(ys) if ys else np.empty(0)
        cfd_w = np.concatenate(ws) if ws else np.empty(0)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), constrained_layout=True)

    vmax = float(
        np.nanpercentile(
            np.concatenate(
                [
                    slice_uh.ravel(),
                    cfd_w if cfd_w is not None and cfd_w.size > 0 else slice_uh.ravel(),
                ]
            ),
            98.0,
        )
    )
    vmax = max(vmax, 0.5)

    pcm = axes[0].pcolormesh(
        x_coords,
        y_coords,
        slice_uh.T,
        cmap="turbo",
        vmin=0.0,
        vmax=vmax,
        shading="nearest",
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title(f"(a) LUT |U_h| at z={z_used:.0f} m")
    cb0 = fig.colorbar(pcm, ax=axes[0], fraction=0.046, pad=0.02)
    cb0.set_label("|U_h| (m/s)")

    # Overlay inside-building footprint at this height.
    if slice_inside.any():
        axes[0].contour(
            x_coords,
            y_coords,
            slice_inside.T,
            levels=[0.5],
            colors="black",
            linewidths=0.4,
            alpha=0.65,
        )

    if cfd_w is not None and cfd_w.size > 0:
        sc = axes[1].scatter(
            cfd_x,
            cfd_y,
            c=cfd_w,
            cmap="turbo",
            vmin=0.0,
            vmax=vmax,
            s=2.0,
            linewidths=0.0,
        )
        axes[1].set_aspect("equal")
        axes[1].set_xlabel("x (m)")
        axes[1].set_ylabel("y (m)")
        axes[1].set_title("(b) CFD reference |U_h| from postProcessing/100m.csv")
        cb1 = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.02)
        cb1.set_label("|U_h| (m/s)")
    else:
        axes[1].text(
            0.5,
            0.5,
            "postProcessing/100m.csv\nnot used at this height",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
        )
        axes[1].set_axis_off()

    fig.suptitle(
        f"Wind LUT QC slice  ({case_dir.name}, LST {time_lst or 'n/a'})",
        fontsize=12,
        fontweight="bold",
    )
    out_png = qc_dir / "qc_slice_z100m.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    _print_step(f"  wrote {out_png}")

    # ---- (b) vertical profile at hotspot ----------------------------------
    i = int(np.clip(np.argmin(np.abs(x_coords - x0)), 0, Nx - 1))
    j = int(np.clip(np.argmin(np.abs(y_coords - y0)), 0, Ny - 1))
    u_prof = U[i, j, :, 0]
    v_prof = U[i, j, :, 1]
    w_prof = U[i, j, :, 2]
    mag_prof = Umag_lut[i, j, :]

    fig2, ax2 = plt.subplots(1, 1, figsize=(6.0, 7.0), constrained_layout=True)
    ax2.plot(u_prof, z_coords, label="u (east)", color="#1f77b4")
    ax2.plot(v_prof, z_coords, label="v (north)", color="#2ca02c")
    ax2.plot(w_prof, z_coords, label="w (up)", color="#9467bd")
    ax2.plot(mag_prof, z_coords, label="|U|", color="#d62728", linewidth=2.0)
    ax2.axvline(0.0, color="0.6", linewidth=0.6)
    ax2.set_xlabel("velocity component (m/s)")
    ax2.set_ylabel("z (m)")
    ax2.set_title(
        f"LUT profile at (x,y) = ({x_coords[i]:.0f}, {y_coords[j]:.0f}) m\n"
        f"{case_dir.name} (LST {time_lst or 'n/a'})"
    )
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)
    out_png2 = qc_dir / "qc_profile_hotspot.png"
    fig2.savefig(out_png2, dpi=200)
    plt.close(fig2)
    _print_step(f"  wrote {out_png2}")

    # ---- (c) summary json -------------------------------------------------
    summary = {
        "case_id": case_dir.name,
        "lut_dimensions": [int(Nx), int(Ny), int(Nz)],
        "x_range": [float(x_coords[0]), float(x_coords[-1])],
        "y_range": [float(y_coords[0]), float(y_coords[-1])],
        "z_range": [float(z_coords[0]), float(z_coords[-1])],
        "n_inside_building": int(inside.sum()),
        "n_invalid": int((valid == 0).sum()),
        "U_mag_stats": {
            "mean": float(np.nanmean(Umag_lut)),
            "p50": float(np.nanpercentile(Umag_lut, 50)),
            "p95": float(np.nanpercentile(Umag_lut, 95)),
            "p99": float(np.nanpercentile(Umag_lut, 99)),
            "max": float(np.nanmax(Umag_lut)),
        },
        "qc_slice_z_used": float(z_used),
        "qc_slice_z_requested": float(qc_slice_z),
        "hotspot_xy_used": [float(x_coords[i]), float(y_coords[j])],
        "hotspot_xy_requested": [float(x0), float(y0)],
        "hotspot_profile_max_speed": float(np.nanmax(mag_prof)),
        "hotspot_profile_z_at_max": float(z_coords[int(np.argmax(mag_prof))]),
    }
    with open(qc_dir / "qc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _print_step(f"  wrote {qc_dir / 'qc_summary.json'}")


if __name__ == "__main__":
    main()
