"""
WRF -> mesoscale-driven inlet field -> OpenFOAM boundary comparison figure.

This script makes a Figure-5-like diagnostic for the steady RANS workflow:

    (a) zonal-mean WS(z) profiles at the south boundary
    (b) raw WRF WS cross-section near the south boundary
    (c) interpolated mesoscale-driven inlet WS cross-section at y_rel = -5000 m
    (d) OpenFOAM y-slice WS cross-section at y = -4995 m, t = 5000

Data provenance:

    W_myExp03/auxhist2/tmp/
    auxhist2_d03_2025-09-01_00%3A00%3A00_tmp.nc
        From util/process_auxhist2_hourly_avg.py. For each center hour, the
        script reads seven WRF auxhist2_d03 snapshots at 10-min spacing
        (center +/- 30 min), concatenates them along time, and writes the
        temporal mean as one hourly-averaged WRF state on the native
        terrain-following grid. This file is the raw mesoscale reference used
        in panels (a) and (b).

    W_myExp03/auxhist2/
    auxhist2_d03_2025-09-01_00%3A00%3A00_1h-rolling_cartesian.nc
        From util/convert_wrf_coord_and_do_3D_interp.py, taking the tmp.nc
        above as input. The script destaggers WRF winds and height, projects
        XLONG/XLAT to local (x_rel, y_rel) meters, horizontally regrids onto a
        uniform 100 m Cartesian mesh, vertically interpolates onto stretched
        z levels with wrf.interplevel, and fills near-surface gaps with a log-
        law extension for U/V/WS. The result is the mesoscale-driven inlet
        field fed to OpenFOAM, shown in panels (a) and (c).

    steady_experiments_finer_ABL/20250901_0000_two_boundaries_as_outlet/
    postProcessing/y-320m_t5000.csv
        From util/export_y_slice_at_time_as_csv.py. A ParaView batch script
        loads the steady OpenFOAM case, converts cell data to point data,
        slices a y-normal plane at y = -4995 m, resamples it onto a regular
        x-z grid over [-2500, 2500] m x [0, 2000] m, and exports point
        coordinates plus velocity fields to CSV. This is the converged RANS
        response shown in panels (a) and (d).

Unlike Lin et al. (2021), this is not an LES turbulence-development figure.
It is intended to show boundary-field inheritance and the converged RANS
response for one steady OpenFOAM case.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize
from pyproj import Proj
from matplotlib.collections import PolyCollection
import sys

DEFAULT_RAW_WRF = (
    "W_myExp03/auxhist2/tmp/"
    "auxhist2_d03_2025-09-01_00%3A00%3A00_tmp.nc"
)
DEFAULT_DRIVER = (
    "W_myExp03/auxhist2/"
    "auxhist2_d03_2025-09-01_00%3A00%3A00_1h-rolling_cartesian.nc"
)
DEFAULT_CFD_CSV = (
    "steady_experiments_finer_ABL/20250901_0000_two_boundaries_as_outlet/"
    "postProcessing/y-320m_t5000.csv"
)
DEFAULT_OUTPUT = (
    "results/wrf_openfoam/wrf_driver_cfd_boundary_figure/"
    "fig5_like_20250901_0000_y-320m_u.png"
)
DEFAULT_BUILDINGS_STL = "constant/triSurface/buildings.stl"

LON0 = 113.32
LAT0 = 23.115
TARGET_Y_DRIVER = -320.0
TARGET_Y_CFD = -320.0
MAX_HEIGHT = 2000.0
X_MIN = -2500.0
X_MAX = 2500.0
G = 9.81
WS_LABEL = r"Wind speed (m s$^{-1}$)"
DRIVER_LABEL = "Mesoscale-driven inlet field"


@dataclass(frozen=True)
class CrossSection:
    x: np.ndarray
    z: np.ndarray
    ws: np.ndarray
    label: str
    y_actual: float | None = None
    z_is_edges: bool = False


@dataclass(frozen=True)
class Profile:
    z: np.ndarray
    ws: np.ndarray
    label: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | os.PathLike[str]) -> Path:
    """Resolve repo-relative paths and accept both ':' and '%3A' timestamp names."""
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    # Avoid Path.resolve() here: on Windows it may collapse filenames that look
    # like 8.3 aliases, e.g. y-320m_t5000.csv -> y-4995m.csv.
    p = p.absolute()
    if p.exists():
        return p

    name = p.name
    candidates = []
    if ":" in name:
        candidates.append(p.with_name(name.replace(":", "%3A")))
        candidates.append(p.with_name(name.replace(":", "%3a")))
    if "%3A" in name or "%3a" in name:
        candidates.append(p.with_name(name.replace("%3A", ":").replace("%3a", ":")))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return p


def destagger_np(arr: np.ndarray, axis: int) -> np.ndarray:
    lo = [slice(None)] * arr.ndim
    hi = [slice(None)] * arr.ndim
    lo[axis] = slice(None, -1)
    hi[axis] = slice(1, None)
    return 0.5 * (arr[tuple(lo)] + arr[tuple(hi)])


def get_array(ds: xr.Dataset, name: str) -> np.ndarray:
    arr = ds[name].values
    return arr[0] if arr.ndim >= 1 and ds[name].dims[0] == "Time" else arr


def projected_xy(lon: np.ndarray, lat: np.ndarray,
                 lon0: float = LON0, lat0: float = LAT0) -> tuple[np.ndarray, np.ndarray]:
    proj = Proj(proj="aeqd", lat_0=lat0, lon_0=lon0, datum="WGS84", units="m")
    return proj(lon, lat)


def wrf_height_mass_and_stag(ph: np.ndarray, phb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match util/convert_wrf_coord_and_do_3D_interp.py: H_stag then destagger to mass levels."""
    h_stag = (ph + phb) / G
    h_mass = destagger_np(h_stag, axis=0)
    return h_mass, h_stag


def extract_raw_wrf_section(
    raw_wrf: Path,
    target_y: float,
    x_min: float,
    x_max: float,
    max_height: float,
    lon0: float,
    lat0: float,
) -> CrossSection:
    with xr.open_dataset(raw_wrf) as ds:
        x_mass, y_mass = projected_xy(get_array(ds, "XLONG"), get_array(ds, "XLAT"), lon0, lat0)
        row = int(np.nanargmin(np.abs(np.nanmean(y_mass, axis=1) - target_y)))
        y_actual = float(np.nanmean(y_mass[row, :]))

        ph = get_array(ds, "PH")[:, row, :]
        phb = get_array(ds, "PHB")[:, row, :]
        u_stag = get_array(ds, "U")[:, row, :]
        v_sn = get_array(ds, "V")[:, row, :]
        v_sn1 = get_array(ds, "V")[:, row + 1, :]

    h_mass, h_stag = wrf_height_mass_and_stag(ph, phb)
    u = destagger_np(u_stag, axis=1)
    v = 0.5 * (v_sn + v_sn1)
    ws = np.sqrt(u**2 + v**2)

    x_row = np.asarray(x_mass[row, :], dtype=float)
    in_range = np.where((x_row >= x_min) & (x_row <= x_max))[0]
    if len(in_range) == 0:
        raise ValueError("No raw WRF grid columns fall inside requested x range.")

    first = max(0, int(in_range[0]) - 1)
    last = min(len(x_row) - 1, int(in_range[-1]) + 1)
    col_idx = np.arange(first, last + 1)
    order = np.argsort(x_row[col_idx])
    col_idx = col_idx[order]
    x_sel = x_row[col_idx]
    z_stag_sel = h_stag[:, col_idx]
    ws_sel = ws[:, col_idx]

    # Keep cells crossing max_height so the axis clip, rather than data masking,
    # controls the visible top edge of the terrain-following WRF layer.
    cell_bottom = z_stag_sel[:-1, :]
    valid = cell_bottom < max_height
    ws_sel = np.where(valid, ws_sel, np.nan)

    return CrossSection(
        x=np.broadcast_to(x_sel[None, :], ws_sel.shape),
        z=z_stag_sel,
        ws=ws_sel,
        label="Raw WRF",
        y_actual=y_actual,
        z_is_edges=True,
    )


def extract_driver_section(
    driver_nc: Path,
    target_y: float,
    x_min: float,
    x_max: float,
    max_height: float,
) -> CrossSection:
    with xr.open_dataset(driver_nc) as ds:
        x = ds["x_rel"].values.astype(float)
        y = ds["y_rel"].values.astype(float)
        z = ds["z"].values.astype(float)
        yi = int(np.nanargmin(np.abs(y - target_y)))
        keep_x = (x >= x_min) & (x <= x_max)
        keep_z = z <= max_height

        if "WS" in ds:
            ws = ds["WS"].isel(y_rel=yi).values[np.ix_(keep_z, keep_x)]
        else:
            u = ds["U"].isel(y_rel=yi).values[np.ix_(keep_z, keep_x)]
            v = ds["V"].isel(y_rel=yi).values[np.ix_(keep_z, keep_x)]
            ws = np.sqrt(u**2 + v**2)

        x_sel = x[keep_x]
        z_sel = z[keep_z]

    xx, zz = np.meshgrid(x_sel, z_sel)
    return CrossSection(
        x=xx,
        z=zz,
        ws=ws,
        label=DRIVER_LABEL,
        y_actual=float(y[yi]),
    )


def load_cfd_section(
    cfd_csv: Path,
    x_min: float,
    x_max: float,
    max_height: float,
) -> CrossSection:
    req = ["Coords:0", "Coords:2", "U:0", "U:1"]
    chunks = []
    for chunk in pd.read_csv(cfd_csv, chunksize=100_000):
        if any(col not in chunk.columns for col in req):
            raise KeyError(f"Missing columns in {cfd_csv}. Expected: {req}")
        sub = chunk[req].astype(float)
        sub = sub[
            (sub["Coords:0"] >= x_min)
            & (sub["Coords:0"] <= x_max)
            & (sub["Coords:2"] <= max_height)
        ]
        chunks.append(sub)

    df = pd.concat(chunks, ignore_index=True)
    if df.empty:
        raise ValueError("No CFD points remain after x/z filtering.")

    u = df["U:0"].to_numpy()
    v = df["U:1"].to_numpy()
    return CrossSection(
        x=df["Coords:0"].to_numpy(),
        z=df["Coords:2"].to_numpy(),
        ws=np.sqrt(u**2 + v**2),
        label="OpenFOAM",
        y_actual=TARGET_Y_CFD,
    )


def profile_from_grid(section: CrossSection, label: str) -> Profile:
    if section.z_is_edges:
        z_centers = 0.5 * (section.z[:-1, :] + section.z[1:, :])
    else:
        z_centers = section.z

    row_valid = np.any(np.isfinite(section.ws), axis=1)
    with np.errstate(invalid="ignore"):
        z = np.nanmean(z_centers[row_valid], axis=1)
        ws = np.nanmean(section.ws[row_valid], axis=1)
    keep = np.isfinite(z) & np.isfinite(ws)
    order = np.argsort(z[keep])
    return Profile(z=z[keep][order], ws=ws[keep][order], label=label)


def profile_from_cfd(section: CrossSection, label: str) -> Profile:
    df = pd.DataFrame({"z": section.z, "ws": section.ws})
    prof = df.groupby("z", sort=True)["ws"].mean().reset_index()
    return Profile(
        z=prof["z"].to_numpy(),
        ws=prof["ws"].to_numpy(),
        label=label,
    )


def robust_norm(*arrays: np.ndarray) -> Normalize:
    values = np.concatenate([
        np.asarray(arr, dtype=float).ravel()
        for arr in arrays
        if arr is not None
    ])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin, vmax = np.nanpercentile(values, [2, 98])
    if np.isclose(vmin, vmax):
        pad = max(0.5, abs(float(vmin)) * 0.1)
        vmin -= pad
        vmax += pad
    return Normalize(vmin=float(vmin), vmax=float(vmax))


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 22,
        "axes.labelsize": 26,
        "axes.titlesize": 28,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def add_panel_label(ax: plt.Axes, label: str, show: bool, info: str | None = None) -> None:
    if not show:
        return
    text = label if not info else f"{label} {info}"
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
    )


def cell_edges_1d(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("Need at least two 1-D centers to compute cell edges.")
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def expand_stag_z_to_pcolormesh(z_stag: np.ndarray) -> np.ndarray:
    """Expand column-centered staggered heights to (nz+1, nx+1) pcolormesh corners."""
    z_stag = np.asarray(z_stag, dtype=float)
    nz_p1, nx = z_stag.shape
    z_corners = np.empty((nz_p1, nx + 1), dtype=float)
    z_corners[:, 0] = z_stag[:, 0]
    z_corners[:, -1] = z_stag[:, -1]
    if nx > 1:
        z_corners[:, 1:-1] = 0.5 * (z_stag[:, :-1] + z_stag[:, 1:])
    return z_corners


def draw_grid_section(
    ax: plt.Axes,
    section: CrossSection,
    norm: Normalize,
    cmap: str,
    panel_label: str,
    show_titles: bool,
) -> None:
    if section.z_is_edges:
        x_centers = np.asarray(section.x[0, :], dtype=float)
        x_edges = cell_edges_1d(x_centers)
        z_corners = expand_stag_z_to_pcolormesh(section.z)
        x_edges_2d = np.broadcast_to(x_edges[None, :], z_corners.shape)
        qm = ax.pcolormesh(
            x_edges_2d,
            z_corners,
            np.ma.masked_invalid(section.ws),
            shading="flat",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
    else:
        qm = ax.pcolormesh(
            section.x,
            section.z,
            section.ws,
            shading="auto",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )

    add_panel_label(ax, panel_label, show_titles, section.label)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Height (m)")
    return qm


def draw_cfd_section(
    ax: plt.Axes,
    section: CrossSection,
    norm: Normalize,
    cmap: str,
    panel_label: str,
    show_titles: bool,
):
    hb = ax.hexbin(
        section.x,
        section.z,
        C=section.ws,
        gridsize=150,
        reduce_C_function=np.mean,
        cmap=cmap,
        norm=norm,
        mincnt=1,
        rasterized=True,
        linewidths=0.0,
    )
    add_panel_label(ax, panel_label, show_titles, section.label)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Height (m)")
    return hb


def make_figure(
    raw: CrossSection,
    driver: CrossSection,
    cfd: CrossSection,
    output: Path,
    max_height: float,
    x_min: float,
    x_max: float,
    cmap: str,
    show_titles: bool,
    buildings_stl: Path | None = None,
    target_y: float = TARGET_Y_CFD,
) -> None:
    configure_style()
    norm = robust_norm(raw.ws, driver.ws, cfd.ws)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12.0, 16.0),
        constrained_layout=True,
    )
    ax_raw, ax_driver, ax_cfd = axes

    qm_raw = draw_grid_section(ax_raw, raw, norm, cmap, "(a)", show_titles)
    draw_grid_section(ax_driver, driver, norm, cmap, "(b)", show_titles)
    hb_cfd = draw_cfd_section(ax_cfd, cfd, norm, cmap, "(c)", show_titles)

    stl_polys = None
    if buildings_stl and buildings_stl.exists():
        sys.path.append(str(repo_root() / "scripts"))
        from visualize_openfoam_domain import load_stl_triangles
        stl_tris = load_stl_triangles(buildings_stl)
        dy = 50.0
        ymin = stl_tris[:, :, 1].min(axis=1)
        ymax = stl_tris[:, :, 1].max(axis=1)
        mask = (ymin <= target_y + dy) & (ymax >= target_y - dy)
        stl_polys = stl_tris[mask][:, :, [0, 2]]

    for ax in [ax_raw, ax_driver, ax_cfd]:
        if stl_polys is not None:
            ax.add_collection(
                PolyCollection(stl_polys, facecolor="#3DAA55", edgecolor="none", alpha=0.55, zorder=5)
            )
        ax.set_ylim(0, max_height)
        ax.set_xlim(x_min, x_max)
        ax.set_aspect("equal")

    cbar = fig.colorbar(
        hb_cfd if hb_cfd is not None else qm_raw,
        ax=[ax_raw, ax_driver, ax_cfd],
        fraction=0.08,
        pad=0.04,
        shrink=0.9,
        aspect=40,
        extend="both",
    )
    cbar.ax.tick_params(labelsize=22)
    cbar.set_label(WS_LABEL, fontsize=26, labelpad=15)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Make a WRF-inlet-field-CFD 2x2 boundary comparison figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-wrf", default=DEFAULT_RAW_WRF, help="Raw WRF auxhist2 tmp NetCDF")
    p.add_argument("--driver", default=DEFAULT_DRIVER,
                   help="Interpolated mesoscale-driven inlet-field NetCDF")
    p.add_argument("--cfd-csv", default=DEFAULT_CFD_CSV,
                   help="ParaView-exported OpenFOAM y-slice CSV")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output PNG path")
    p.add_argument("--target-y", type=float, default=TARGET_Y_DRIVER,
                   help="South-boundary y coordinate for WRF/driver extraction")
    p.add_argument("--max-height", type=float, default=MAX_HEIGHT)
    p.add_argument("--x-min", type=float, default=X_MIN)
    p.add_argument("--x-max", type=float, default=X_MAX)
    p.add_argument("--lon0", type=float, default=LON0,
                   help="Local Cartesian projection origin longitude")
    p.add_argument("--lat0", type=float, default=LAT0,
                   help="Local Cartesian projection origin latitude")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--buildings-stl", default=None, nargs="?", const=DEFAULT_BUILDINGS_STL, 
                   help="Path to buildings STL. If flag is passed without path, uses default STL.")
    p.add_argument("--title", default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--no-titles",
        action="store_true",
        help="Hide in-axis panel labels (a)-(d)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    raw_wrf = resolve_path(args.raw_wrf)
    driver_nc = resolve_path(args.driver)
    cfd_csv = resolve_path(args.cfd_csv)
    buildings_stl = resolve_path(args.buildings_stl) if args.buildings_stl else None
    
    out_path = Path(args.output)
    if buildings_stl is not None:
        out_path = out_path.with_name(f"{out_path.stem}_with_buildings{out_path.suffix}")
    output = resolve_path(out_path)

    for path, label in [(raw_wrf, "raw WRF"), (driver_nc, "driver"), (cfd_csv, "CFD CSV")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    print("=" * 72)
    print("WRF -> mesoscale-driven inlet field -> OpenFOAM boundary figure")
    print("=" * 72)
    print(f"Raw WRF : {raw_wrf}")
    print(f"Driver  : {driver_nc}")
    print(f"CFD CSV : {cfd_csv}")
    print(f"Output  : {output}")
    print("=" * 72)

    raw = extract_raw_wrf_section(
        raw_wrf,
        target_y=args.target_y,
        x_min=args.x_min,
        x_max=args.x_max,
        max_height=args.max_height,
        lon0=args.lon0,
        lat0=args.lat0,
    )
    driver = extract_driver_section(
        driver_nc,
        target_y=args.target_y,
        x_min=args.x_min,
        x_max=args.x_max,
        max_height=args.max_height,
    )
    cfd = load_cfd_section(
        cfd_csv,
        x_min=args.x_min,
        x_max=args.x_max,
        max_height=args.max_height,
    )

    make_figure(
        raw,
        driver,
        cfd,
        output,
        max_height=args.max_height,
        x_min=args.x_min,
        x_max=args.x_max,
        cmap=args.cmap,
        show_titles=not args.no_titles,
        buildings_stl=buildings_stl,
        target_y=args.target_y,
    )
    print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()
