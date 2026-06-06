"""
WRF -> Cartesian driver -> OpenFOAM boundary comparison figure.

This script makes a Figure-5-like diagnostic for the steady RANS workflow:

    (a) zonal-mean WS(z) profiles at the south boundary
    (b) raw WRF WS cross-section near the south boundary
    (c) interpolated Cartesian WRF driver WS cross-section at y_rel = -5000 m
    (d) OpenFOAM y-slice WS cross-section at y = -4995 m, t = 5000

Unlike Lin et al. (2021), this is not an LES turbulence-development figure.
It is intended to show boundary-field inheritance and the converged RANS
response for one steady OpenFOAM case.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize
from pyproj import Proj


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
    "postProcessing/y-4995m_t5000.csv"
)
DEFAULT_OUTPUT = (
    "results/wrf_openfoam/wrf_driver_cfd_boundary_figure/"
    "fig5_like_20250901_0000_south_u.png"
)

LON0 = 113.32
LAT0 = 23.115
TARGET_Y_DRIVER = -5000.0
TARGET_Y_CFD = -4995.0
MAX_HEIGHT = 2000.0
X_MIN = -2500.0
X_MAX = 2500.0
G = 9.81
WS_LABEL = r"Wind speed (m s$^{-1}$)"


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
    # like 8.3 aliases, e.g. y-4995m_t5000.csv -> y-4995m.csv.
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

    # Mask cells whose upper interface exceeds max_height; do not extend below
    # the terrain-following WRF staggered bottom interface.
    cell_top = z_stag_sel[1:, :]
    valid = cell_top <= max_height
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
        label="Cartesian WRF driver",
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
        label="OpenFOAM t=5000",
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
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def add_panel_label(ax: plt.Axes, label: str, show: bool) -> None:
    if not show:
        return
    ax.text(
        0.02, 0.96, label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
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

    add_panel_label(ax, panel_label, show_titles)
    if show_titles:
        ax.set_title(section.label)
    if section.y_actual is not None and show_titles:
        ax.text(
            0.98, 0.04, f"y ~= {section.y_actual:.0f} m",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )
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
    add_panel_label(ax, panel_label, show_titles)
    if show_titles:
        ax.set_title(section.label)
    if section.y_actual is not None and show_titles:
        ax.text(
            0.98, 0.04, f"y = {section.y_actual:.0f} m",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Height (m)")
    return hb


def parse_case_time(cfd_csv: Path) -> str:
    m = re.search(r"(\d{8})_(\d{4})", str(cfd_csv))
    if not m:
        return ""
    day, hm = m.groups()
    return f"{day[:4]}-{day[4:6]}-{day[6:]} {hm[:2]}:{hm[2:]} UTC"


def make_figure(
    raw: CrossSection,
    driver: CrossSection,
    cfd: CrossSection,
    output: Path,
    max_height: float,
    x_min: float,
    x_max: float,
    cmap: str,
    title: str,
    show_titles: bool,
) -> None:
    configure_style()
    raw_prof = profile_from_grid(raw, "Raw WRF")
    driver_prof = profile_from_grid(driver, "Cartesian driver")
    cfd_prof = profile_from_cfd(cfd, "OpenFOAM t=5000")

    norm = robust_norm(raw.ws, driver.ws, cfd.ws)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(11.0, 8.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    ax_prof, ax_raw = axes[0]
    ax_driver, ax_cfd = axes[1]

    ax_prof.plot(raw_prof.ws, raw_prof.z, "ko-", ms=3.0, lw=1.4, label=raw_prof.label)
    ax_prof.plot(driver_prof.ws, driver_prof.z, color="#1f4ed8", ls="--", lw=1.7,
                 label=driver_prof.label)
    ax_prof.plot(cfd_prof.ws, cfd_prof.z, color="#d62728", lw=1.7, label=cfd_prof.label)
    add_panel_label(ax_prof, "(a)", show_titles)
    ax_prof.set_xlabel(WS_LABEL)
    ax_prof.set_ylabel("Height (m)")
    ax_prof.set_ylim(0, max_height)
    if show_titles:
        ax_prof.set_title("Boundary-mean profile")
    ax_prof.legend(loc="upper right", fontsize=8, frameon=True)

    qm_raw = draw_grid_section(ax_raw, raw, norm, cmap, "(b)", show_titles)
    draw_grid_section(ax_driver, driver, norm, cmap, "(c)", show_titles)
    hb_cfd = draw_cfd_section(ax_cfd, cfd, norm, cmap, "(d)", show_titles)

    for ax in [ax_raw, ax_driver, ax_cfd]:
        ax.set_ylim(0, max_height)
        ax.set_xlim(x_min, x_max)

    cbar = fig.colorbar(
        hb_cfd if hb_cfd is not None else qm_raw,
        ax=[ax_raw, ax_driver, ax_cfd],
        fraction=0.035,
        pad=0.018,
        extend="both",
    )
    cbar.set_label(WS_LABEL)

    if show_titles:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Make a WRF-driver-CFD 2x2 boundary comparison figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-wrf", default=DEFAULT_RAW_WRF, help="Raw WRF auxhist2 tmp NetCDF")
    p.add_argument("--driver", default=DEFAULT_DRIVER,
                   help="Interpolated Cartesian WRF driver NetCDF")
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
    p.add_argument("--title", default=None)
    p.add_argument(
        "--no-titles",
        action="store_true",
        help="Hide figure title, subplot titles, and panel labels (a)-(d)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    raw_wrf = resolve_path(args.raw_wrf)
    driver_nc = resolve_path(args.driver)
    cfd_csv = resolve_path(args.cfd_csv)
    output = resolve_path(args.output)

    for path, label in [(raw_wrf, "raw WRF"), (driver_nc, "driver"), (cfd_csv, "CFD CSV")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    print("=" * 72)
    print("WRF -> Cartesian driver -> OpenFOAM boundary figure")
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

    case_time = parse_case_time(cfd_csv)
    title = args.title or (
        "South-Boundary Wind Speed Inheritance: Raw WRF, Cartesian Driver, and OpenFOAM"
        + (f"\n{case_time}" if case_time else "")
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
        title=title,
        show_titles=not args.no_titles,
    )
    print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()
