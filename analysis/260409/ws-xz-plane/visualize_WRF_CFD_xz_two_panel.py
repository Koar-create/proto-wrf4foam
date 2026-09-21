"""
WRF–CFD X-Z Vertical Wind Field Comparison (2-panel)
====================================================
Produces a publication-quality 2-panel figure.

Encoding
--------
* Colour fill — horizontal wind speed ``sqrt(U^2 + V^2)`` (m/s)
* Both panels share the same local-X footprint (m) and building silhouettes

Layout ``horizon`` (default, 1×2):

    ┌──────────────────┬──────────────────┐
    │ (a) WRF          │ (b) CFD     ▐ CB │
    └──────────────────┴──────────────────┘

Layout ``vertical`` (2×1):

    ┌──────────────────┐
    │ (a) WRF      ▐   │
    ├──────────────────┤ CB
    │ (b) CFD      ▐   │
    └──────────────────┘

Usage
-----
    python analysis/260409/ws-xz-plane/visualize_WRF_CFD_xz_two_panel.py  /path/to/CFD_run_directory
    python analysis/260409/ws-xz-plane/visualize_WRF_CFD_xz_two_panel.py  /path/to/CFD_run_directory --layout vertical

Example
-------
    python analysis/260409/ws-xz-plane/visualize_WRF_CFD_xz_two_panel.py \\
        steady_experiments_finer_ABL/20250901_1000_two_boundaries_as_outlet

Path inference
--------------
Given CFD path ``<root>/<YYYYMMDD_HHMM>_<tag>`` the script resolves:

* WRF nc  →  ``W_myExp03|05/auxhist2/tmp/auxhist2_d03_<YYYY-MM-DD_HH:MM:00>_tmp.nc``
  (``W_myExp03`` for dates ≤ 09-06; ``W_myExp05`` for ≥ 09-07).
  On Windows, ``:`` in the filename is often stored as ``%3A``; path resolution
  accepts both forms.
* CFD CSV →  ``<cfd_dir>/postProcessing/y<y_slice>m.csv``  (default y_slice=800)
* PNG out →  ``results/wrf_openfoam/xz_wrf_cfd/<horizon|vertical>_layout/xz_wrf_cfd_<YYYYMMDD_HHMM>.png``
* SHP     →  ``data/Guangzhou_shp_file/project_UTM49/Export_Output.shp``
  (white building silhouettes on both panels at the Y-slice)

Override with ``--wrf-nc``, ``--cfd-csv``, ``--shp``, or ``--output``.
"""

import os
import re
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, FixedLocator

# Repo util: fixed Guangzhou origin lon/lat ↔ local XY (UTM49N)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
_UTIL_DIR = os.path.join(_REPO_ROOT, "util")
if _UTIL_DIR not in sys.path:
    sys.path.insert(0, _UTIL_DIR)
from convert_lonlat_xy_origin import (  # noqa: E402
    ORIGIN_LAT, ORIGIN_LON, lonlat_to_xy, xy_to_lonlat,
)

# ---------------------------------------------------------------------------
# CONSTANTS / DEFAULTS
# ---------------------------------------------------------------------------
WRF_ROOT_EXP03  = os.path.join("W_myExp03", "auxhist2", "tmp")
WRF_ROOT_EXP05  = os.path.join("W_myExp05", "auxhist2", "tmp")
# Dates on/after this calendar day use W_myExp05; earlier dates use W_myExp03
WRF_EXP05_START = (2025, 9, 7)
WRF_NC_TEMPLATE = "auxhist2_d03_{wrf_time}_tmp.nc"
DEFAULT_Y_SLICE = 800
DEFAULT_LAYOUT  = "horizon"
LAYOUT_DIRS     = {
    "horizon": "horizon_layout",
    "vertical": "vertical_layout",
}

# Default crop centre matches building/OpenFOAM origin
TARGET_LAT   = ORIGIN_LAT
TARGET_LON   = ORIGIN_LON
LAT_TOL      = 0.004
LON_TOL      = 0.0225000225
MAX_HEIGHT   = 2000
CFD_TOP      = 2000
HEXBIN_GRID  = 120
# Cap colorbar at this value (m/s); colours saturate above, but low-layer
# gradients remain visible even when ambient wind speed is very high.
COLORBAR_VMAX_CAP = 12.0

WIND_SPEED_COLORBAR_TICK_FORMAT = '%g'
# Back-compat fallbacks for callers that still reference fixed colorbar limits
WIND_SPEED_COLORBAR_VMAX = 16.0
WIND_SPEED_COLORBAR_TICKS = np.arange(0.0, 17.0, 2.0)
RESULTS_XZ_DIR = os.path.join("results", "wrf_openfoam", "xz_wrf_cfd")
DEFAULT_SHP = os.path.join(
    "data", "Guangzhou_shp_file", "project_UTM49", "Export_Output.shp",
)
# Legacy alias (older scripts expected a fixed y800m path)
CSV_RELPATH = os.path.join("postProcessing", f"y{DEFAULT_Y_SLICE}m.csv")


def _nice_vmax(v: float) -> float:
    """Round *v* up to a neat colorbar ceiling."""
    if not np.isfinite(v) or v <= 0:
        return 1.0
    step = 1.0 if v <= 5 else 2.0
    return float(np.ceil(v / step) * step)


def wind_speed_colorbar_ticks(vmax: float) -> np.ndarray:
    vmax = float(vmax)
    step = 1.0 if vmax <= 5 else 2.0
    ticks = np.arange(0.0, vmax + 0.5 * step, step)
    return ticks[ticks <= vmax + 1e-9]


# ---------------------------------------------------------------------------
# PATH HELPERS
# ---------------------------------------------------------------------------

def _repo_root() -> str:
    return _REPO_ROOT


def default_output_path(cfd_dir: str, layout: str = DEFAULT_LAYOUT) -> str:
    """
    ``results/wrf_openfoam/xz_wrf_cfd/<horizon|vertical>_layout/xz_wrf_cfd_<YYYYMMDD_HHMM>.png``
    """
    cfd_dir = cfd_dir.rstrip(os.sep)
    case = os.path.basename(cfd_dir)
    m = re.match(r"(\d{8}_\d{4})", case)
    stamp = m.group(1) if m else case
    layout_dir = LAYOUT_DIRS.get(layout, LAYOUT_DIRS[DEFAULT_LAYOUT])
    return os.path.join(
        _repo_root(), RESULTS_XZ_DIR, layout_dir,
        f"xz_wrf_cfd_{stamp}.png",
    )


def parse_timestamp_from_cfd_dir(cfd_dir: str):
    basename = os.path.basename(cfd_dir.rstrip("/"))
    m = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})", basename)
    if not m:
        raise ValueError(
            f"Cannot parse YYYYMMDD_HHMM from directory name: '{basename}'\n"
            "Expected format: <root>/<YYYYMMDD_HHMM>_<tag>"
        )
    yr, mo, dy, hh, mm = m.groups()
    return f"{yr}-{mo}-{dy}_{hh}:{mm}:00"


def wrf_root_for_time(wrf_time: str) -> str:
    """
    Pick auxhist tmp root from WRF timestamp ``YYYY-MM-DD_HH:MM:00``.

    ``W_myExp03`` for dates before 2025-09-07; ``W_myExp05`` from 09-07 onward.
    """
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})_", wrf_time)
    if not m:
        return WRF_ROOT_EXP03
    ymd = tuple(int(x) for x in m.groups())
    return WRF_ROOT_EXP05 if ymd >= WRF_EXP05_START else WRF_ROOT_EXP03


def resolve_existing_wrf_nc_path(wrf_nc_path: str) -> str:
    """
    Return an existing WRF NetCDF path.

    Accepts both colon timestamps (``10:00:00``) and Windows-safe URL-encoded
    names (``10%3A00%3A00``).
    """
    if os.path.exists(wrf_nc_path):
        return wrf_nc_path

    directory, filename = os.path.split(wrf_nc_path)
    candidates = []
    if ":" in filename:
        candidates.append(os.path.join(directory, filename.replace(":", "%3A")))
        candidates.append(os.path.join(directory, filename.replace(":", "%3a")))
    if "%3A" in filename or "%3a" in filename:
        candidates.append(
            os.path.join(directory, filename.replace("%3A", ":").replace("%3a", ":"))
        )

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return wrf_nc_path


def infer_paths(cfd_dir: str, y_slice: float = DEFAULT_Y_SLICE):
    wrf_time = parse_timestamp_from_cfd_dir(cfd_dir)
    nc_filename = WRF_NC_TEMPLATE.format(wrf_time=wrf_time)
    wrf_nc_path = resolve_existing_wrf_nc_path(
        os.path.join(wrf_root_for_time(wrf_time), nc_filename)
    )
    # Integer slices → y800m.csv; keep decimals if needed → y800.5m.csv
    if float(y_slice).is_integer():
        y_tag = f"{int(y_slice)}"
    else:
        y_tag = f"{y_slice:g}"
    cfd_csv = os.path.join(cfd_dir, "postProcessing", f"y{y_tag}m.csv")
    return wrf_nc_path, cfd_csv, wrf_time


def list_available_y_slices(cfd_dir: str) -> list:
    """Return sorted Y-slice values for which ``postProcessing/y*m.csv`` exists."""
    post = os.path.join(cfd_dir, "postProcessing")
    if not os.path.isdir(post):
        return []
    vals = []
    for name in os.listdir(post):
        m = re.fullmatch(r"y(-?\d+(?:\.\d+)?)m\.csv", name)
        if m:
            vals.append(float(m.group(1)))
    return sorted(vals)


def resolve_cfd_csv(cfd_dir: str, y_slice: float, cfd_csv_override: str = None) -> str:
    """Resolve CFD Y-slice CSV; raise with available slices if missing."""
    if cfd_csv_override:
        cfd_csv = cfd_csv_override
    else:
        _, cfd_csv, _ = infer_paths(cfd_dir, y_slice)
    if os.path.isfile(cfd_csv):
        return cfd_csv
    available = list_available_y_slices(cfd_dir)
    avail_txt = (
        ", ".join(f"y{v:g}m" for v in available) if available else "(none found)"
    )
    raise FileNotFoundError(
        f"CFD Y-slice CSV not found: {cfd_csv}\n"
        f"  Requested --y-slice {y_slice:g} → postProcessing/y{y_slice:g}m.csv\n"
        f"  Available Y-slice CSVs in {cfd_dir}/postProcessing/: {avail_txt}\n"
        f"  Tip: use e.g. --y-slice 800, or pass --cfd-csv explicitly."
    )


def parse_y_slice_from_csv(csv_path: str, fallback: float = DEFAULT_Y_SLICE) -> float:
    m = re.search(r"y(-?\d+(?:\.\d+)?)m\.csv$", os.path.basename(csv_path))
    return float(m.group(1)) if m else float(fallback)


def cfd_xz_limits(cfd_data: dict):
    """Axis-aligned CFD panel view ``[x_min,x_max] × [z_min,z_max]``."""
    x = cfd_data['x']
    z = cfd_data['z']
    return (
        float(np.nanmin(x)), float(np.nanmax(x)),
        float(np.nanmin(z)), float(np.nanmax(z)),
    )


def lon_limits_from_x_range(x0: float, x1: float, y_slice: float):
    """Map local-X ends at fixed Y to longitude via ``xy_to_lonlat``."""
    lon_a, _ = xy_to_lonlat(x0, y_slice)
    lon_b, _ = xy_to_lonlat(x1, y_slice)
    return (min(lon_a, lon_b), max(lon_a, lon_b))


def lat_of_y_slice(y_slice: float, x_ref: float = 0.0) -> float:
    """Latitude of the OpenFOAM Y-slice plane at *x_ref*."""
    _, lat = xy_to_lonlat(x_ref, y_slice)
    return float(lat)


def lon_to_x_at_y(lon: np.ndarray, y_slice: float) -> np.ndarray:
    """Map longitude array at fixed Y-slice to local X (m)."""
    lat = lat_of_y_slice(y_slice)
    lon = np.asarray(lon, dtype=float)
    x_out = np.empty_like(lon, dtype=float)
    for i, lo in np.ndenumerate(lon):
        x_out[i], _ = lonlat_to_xy(float(lo), lat)
    return x_out


# ---------------------------------------------------------------------------
# BUILDING SILHOUETTES (SHP ∩ Y-slice → white X-Z basemap)
# ---------------------------------------------------------------------------

def _rings_from_shape(shp) -> list:
    pts = np.asarray(shp.points, dtype=float)
    if pts.size == 0:
        return []
    parts = list(shp.parts) + [len(pts)]
    rings = []
    for i in range(len(parts) - 1):
        if i > 0:
            break
        ring = pts[parts[i]:parts[i + 1]]
        if ring.shape[0] >= 2 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if ring.shape[0] < 3:
            continue
        x, y = ring[:, 0], ring[:, 1]
        area = abs(0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))
        if area < 5.0:
            continue
        rings.append(ring)
    return rings


def _x_intervals_at_y(ring_utm: np.ndarray, y0: float) -> list:
    """Intersection of a UTM ring with the horizontal line y = y0 → x intervals."""
    xs = []
    n = len(ring_utm)
    for i in range(n):
        x1, y1 = ring_utm[i]
        x2, y2 = ring_utm[(i + 1) % n]
        # Proper crossing (exclude horizontal edges)
        if (y1 > y0) == (y2 > y0):
            continue
        if y1 == y2:
            continue
        t = (y0 - y1) / (y2 - y1)
        if 0.0 <= t <= 1.0:
            xs.append(x1 + t * (x2 - x1))
    if len(xs) < 2:
        return []
    xs = sorted(xs)
    # Pair consecutive intersections into intervals
    return [(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)]


def load_building_xz_rects_x(
    shp_path: str,
    y_slice: float,
    x_lim=None,
    height_field: str = "jzgd",
    encoding: str = "gbk",
    pad: float = 50.0,
):
    """
    Buildings intersecting local Y = *y_slice*, as rectangles in (X, z).

    Returns list of (N,2) polygon vertex arrays for PolyCollection.
    """
    import shapefile
    from pyproj import Transformer

    if not os.path.exists(shp_path):
        warnings.warn(f"Building shapefile not found: {shp_path}")
        return []

    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    ox, oy = to_utm.transform(ORIGIN_LON, ORIGIN_LAT)
    y_utm = oy + y_slice

    x0 = x1 = None
    if x_lim is not None:
        x0, x1 = x_lim[0] - pad, x_lim[1] + pad

    reader = shapefile.Reader(shp_path, encoding=encoding)
    field_names = [f[0] for f in reader.fields[1:]]
    if height_field not in field_names:
        warnings.warn(f"Height field '{height_field}' not in {field_names}; skip buildings")
        return []
    hi = field_names.index(height_field)

    polys = []
    for shp, rec in zip(reader.shapes(), reader.records()):
        if shp.shapeType not in (5, 15, 25):
            continue
        try:
            h = float(rec[hi])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(h) or h <= 0:
            continue
        for ring_utm in _rings_from_shape(shp):
            y_min, y_max = float(ring_utm[:, 1].min()), float(ring_utm[:, 1].max())
            if y_utm < y_min - 1e-6 or y_utm > y_max + 1e-6:
                continue
            for xa, xb in _x_intervals_at_y(ring_utm, y_utm):
                xl, xr = sorted([xa - ox, xb - ox])
                if x0 is not None and (xr < x0 or xl > x1):
                    continue
                if xr - xl < 1e-8:
                    continue
                polys.append(np.array([
                    [xl, 0.0],
                    [xr, 0.0],
                    [xr, h],
                    [xl, h],
                ], dtype=float))

    print(f"  Building basemap: {len(polys)} silhouettes at Y={y_slice:g} m "
          f"from {os.path.basename(shp_path)}")
    return polys


def draw_building_basemap(ax, polys, facecolor='white', edgecolor='#8a8a8a',
                          lw=0.25, zorder=1, alpha=1.0):
    """White building silhouettes as a silent basemap (no legend)."""
    if not polys:
        return
    pc = PolyCollection(
        polys, facecolors=facecolor, edgecolors=edgecolor,
        linewidths=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_collection(pc)


# ---------------------------------------------------------------------------
# WRF DATA EXTRACTION
# ---------------------------------------------------------------------------

def _destagger_np(arr: np.ndarray, axis: int) -> np.ndarray:
    slc_lo = [slice(None)] * arr.ndim
    slc_hi = [slice(None)] * arr.ndim
    slc_lo[axis] = slice(None, -1)
    slc_hi[axis] = slice(1, None)
    return 0.5 * (arr[tuple(slc_lo)] + arr[tuple(slc_hi)])


def _v_at_mass_row(v_arr: np.ndarray, sn_idx: int) -> np.ndarray:
    """Average V from south_north-staggered grid onto mass-point row *sn_idx*."""
    n_sn = v_arr.shape[1]
    if sn_idx + 1 < n_sn:
        return 0.5 * (v_arr[:, sn_idx, :] + v_arr[:, sn_idx + 1, :])
    if sn_idx > 0:
        return 0.5 * (v_arr[:, sn_idx - 1, :] + v_arr[:, sn_idx, :])
    return v_arr[:, sn_idx, :].copy()


def extract_wrf_xz(nc_path: str,
                   target_lat=TARGET_LAT, target_lon=TARGET_LON,
                   lat_tol=LAT_TOL, lon_tol=LON_TOL,
                   max_height=MAX_HEIGHT):
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"WRF file not found: {nc_path}")

    print(f"  Opening: {nc_path}")
    ds = xr.open_dataset(nc_path)

    def get_val(name):
        v = ds[name]
        return v.values[0] if 'Time' in v.dims else v.values

    lats = get_val('XLAT')
    if lats.ndim == 3:
        lats = lats[0]
    lat_1d = np.mean(lats, axis=1)
    sn_idx = int(np.argmin(np.abs(lat_1d - target_lat)))
    actual_lat = float(lat_1d[sn_idx])
    print(f"  Nearest south_north row: index={sn_idx}, lat={actual_lat:.6f} deg N")

    lons = get_val('XLONG')
    if lons.ndim == 3:
        lons = lons[0]
    lon_1d = lons[sn_idx, :]
    we_mask = (target_lon - lon_tol <= lon_1d) & (lon_1d <= target_lon + lon_tol)
    we_idx = np.where(we_mask)[0]

    if len(we_idx) < 2:
        print(f"  [!] Only {len(we_idx)} column(s) within lon_tol={lon_tol:.4f} deg. "
              "Falling back to full west_east extent.")
        we_slice = slice(None)
    else:
        we_slice = slice(int(we_idx[0]), int(we_idx[-1]) + 1)

    PH_all = get_val('PH')
    PHB_all = get_val('PHB')
    PH_sn = PH_all[:, sn_idx, :]
    PHB_sn = PHB_all[:, sn_idx, :]

    U_all = get_val('U')
    U_sn = U_all[:, sn_idx, :]

    V_all = get_val('V')
    V_row = _v_at_mass_row(V_all, sn_idx)

    W_all = get_val('W')
    W_sn = W_all[:, sn_idx, :]

    ds.close()

    H_sn = _destagger_np((PH_sn + PHB_sn) / 9.81, axis=0)
    U_dest = _destagger_np(U_sn, axis=1)
    V_dest = V_row
    W_dest = _destagger_np(W_sn, axis=0)
    ws = np.sqrt(U_dest**2 + V_dest**2)

    H_xz = H_sn[:, we_slice]
    ws_xz = ws[:, we_slice]
    U_xz = U_dest[:, we_slice]
    W_xz = W_dest[:, we_slice]
    lon_xz = lon_1d[we_slice]

    nan_mask = H_xz > max_height
    ws_xz[nan_mask] = np.nan
    U_xz[nan_mask] = np.nan
    W_xz[nan_mask] = np.nan

    lon_2d = np.broadcast_to(lon_xz[np.newaxis, :], H_xz.shape).copy()

    print(f"  Extracted slice: shape={H_xz.shape}, "
          f"lon=[{np.nanmin(lon_xz):.5f}, {np.nanmax(lon_xz):.5f}] deg E, "
          f"H=[{np.nanmin(H_xz):.0f}, {np.nanmax(H_xz):.0f}] m, "
          f"WS=[{np.nanmin(ws_xz):.2f}, {np.nanmax(ws_xz):.2f}] m/s")

    return dict(lon=lon_2d, height=H_xz, wind_speed=ws_xz, U=U_xz, W=W_xz)


# ---------------------------------------------------------------------------
# CFD CSV
# ---------------------------------------------------------------------------

def load_cfd_csv(csv_path: str):
    """
    Load an OpenFOAM Y-slice CSV.

    ``wind_speed`` stores horizontal speed ``sqrt(U:0^2 + U:1^2)``.
    ``u2`` is retained for optional vertical-velocity contours.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CFD CSV not found: {csv_path}")

    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=100_000):
        req = ['Coords:0', 'Coords:2', 'U:0', 'U:1', 'U:2']
        if any(c not in chunk.columns for c in req):
            raise KeyError(f"Missing columns in {csv_path}. Expected: {req}")
        chunks.append(chunk[req].astype(float))

    df = pd.concat(chunks, ignore_index=True)
    u0 = df['U:0'].values
    u1 = df['U:1'].values
    u2 = df['U:2'].values
    return dict(
        x=df['Coords:0'].values, z=df['Coords:2'].values,
        u0=u0, u1=u1, u2=u2,
        wind_speed=np.sqrt(u0**2 + u1**2),
    )


# ---------------------------------------------------------------------------
# PANEL DRAWING
# ---------------------------------------------------------------------------

_STYLE_DONE = False


def _apply_global_style():
    global _STYLE_DONE
    if _STYLE_DONE:
        return
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 13,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.linewidth': 0.8,
        'figure.dpi': 150,
    })
    _STYLE_DONE = True


def _add_panel_label(ax, label, fontsize=15):
    ax.text(0.015, 0.965, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75))


def add_wind_speed_colorbar(fig, mappable, ax=None, label='Wind Speed (m/s)',
                            ticks=None, tick_format=None, cax=None):
    """Add a colorbar with neat tick positions derived from the mappable clim."""
    _, vmax = mappable.get_clim()
    ticks = np.asarray(
        ticks if ticks is not None else wind_speed_colorbar_ticks(vmax),
        dtype=float,
    )
    fmt = tick_format or WIND_SPEED_COLORBAR_TICK_FORMAT
    if cax is not None:
        cb = fig.colorbar(mappable, cax=cax, label=label)
    else:
        cb = fig.colorbar(mappable, ax=ax, label=label, pad=0.02, fraction=0.04)
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    cb.ax.tick_params(labelsize=10)
    return cb


def wrf_lon_to_local_x(wrf_data: dict, y_slice: float) -> dict:
    """
    Convert WRF longitude columns to local X (m) at the Y-slice.

    Does **not** regrid or densify the field — only remaps the horizontal
    coordinate so both panels share the same X axis. Native WRF column
    spacing (and blocky resolution) is preserved.
    """
    lon_1d = wrf_data['lon'][0, :]
    x_1d = lon_to_x_at_y(lon_1d, y_slice)
    wrf_data['x'] = np.broadcast_to(x_1d[np.newaxis, :], wrf_data['lon'].shape).copy()
    return wrf_data


def draw_wrf_panel(ax, data: dict, cfd_top=CFD_TOP,
                   vmax=None, max_height=MAX_HEIGHT,
                   label='(a) WRF',
                   building_polys=None,
                   xz_lim=None):
    """Draw WRF panel on native columns in local X (m); no display regridding."""
    x_coord = data['x']
    height = data['height']
    ws = data['wind_speed']

    if vmax is None:
        vmax = np.nanpercentile(ws, 98)

    polys = building_polys or []
    draw_building_basemap(ax, polys, facecolor='white', edgecolor='none', zorder=1)

    # pcolormesh on native mass-point columns — keeps WRF's coarse footprint honest
    qm = ax.pcolormesh(
        x_coord, height, ws,
        vmin=0, vmax=vmax, cmap='viridis',
        shading='auto', alpha=0.85, rasterized=True, zorder=2,
    )

    draw_building_basemap(
        ax, polys, facecolor='white', edgecolor='#8a8a8a', lw=0.25,
        alpha=0.45, zorder=3,
    )

    if xz_lim is not None:
        x0, x1, z0, z1 = xz_lim
        ax.set_xlim(x0, x1)
        ax.set_ylim(z0, z1)
    else:
        ax.set_ylim(0, max_height)
    ax.margins(0)

    h_top = xz_lim[3] if xz_lim is not None else max_height
    if cfd_top <= h_top:
        ax.axhline(cfd_top, color='black', ls=':', lw=1.6, alpha=0.75, zorder=4)
        x_c = 0.5 * (xz_lim[0] + xz_lim[1]) if xz_lim is not None else float(np.nanmean(x_coord))
        ax.text(x_c, cfd_top + 35,
                f'CFD top ({cfd_top:g} m)',
                color='black', fontsize=10, ha='center', va='bottom', alpha=0.85,
                zorder=4)

    ax.set_xlabel('X (m)', fontweight='bold')
    ax.set_ylabel('Height (m)', fontweight='bold')
    ax.grid(True, alpha=0.25, ls='--', lw=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    _add_panel_label(ax, label)
    return qm


def draw_cfd_panel(ax, data: dict, vmax=None,
                   label='(b) WRF-to-OpenFOAM',
                   xz_lim=None, building_polys=None):
    x = data['x']
    z = data['z']
    ws = data['wind_speed']

    if vmax is None:
        vmax = np.nanpercentile(ws, 98)

    polys = building_polys or []
    draw_building_basemap(ax, polys, facecolor='white', edgecolor='none', zorder=1)

    hb = ax.hexbin(x, z, C=ws,
                   gridsize=HEXBIN_GRID, cmap='viridis',
                   reduce_C_function=np.mean,
                   vmin=0, vmax=vmax,
                   alpha=0.88, edgecolors='none', rasterized=True, zorder=2)

    draw_building_basemap(
        ax, polys, facecolor='white', edgecolor='#8a8a8a', lw=0.25,
        alpha=0.45, zorder=3,
    )

    if xz_lim is not None:
        x0, x1, z0, z1 = xz_lim
    else:
        x0, x1, z0, z1 = cfd_xz_limits(data)
    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)
    ax.margins(0)

    ax.set_xlabel('X (m)', fontweight='bold')
    ax.set_ylabel('')  # right panel: no ylabel (shared Height with left)
    ax.set_aspect('auto')
    ax.grid(True, alpha=0.25, ls='--', lw=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    _add_panel_label(ax, label)
    return hb


def short_case_label(case: str) -> str:
    """Keep only YYYYMMDD_HHMM; drop tags like ``two_boundaries_as_outlet``."""
    m = re.match(r"(\d{8}_\d{4})", case)
    return m.group(1) if m else case


def title_timestamp(case: str) -> str:
    """
    Format case stamp for the figure title.

    Case IDs store UTC time (``YYYYMMDD_HHMM``); the title shows local
    Beijing time as ``YYYY-mm-dd HH:MM (UTC+8)``.
    """
    from datetime import datetime, timedelta, timezone

    m = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})", case)
    if not m:
        return short_case_label(case)
    yr, mo, dy, hh, mm = (int(x) for x in m.groups())
    utc = datetime(yr, mo, dy, hh, mm, tzinfo=timezone.utc)
    local = utc.astimezone(timezone(timedelta(hours=8)))
    return local.strftime("%Y-%m-%d %H:%M (UTC+8)")


def compose_figure(wrf_data, cfd_data, case_label: str, output_path: str,
                   max_height=MAX_HEIGHT, cfd_top=CFD_TOP,
                   layout: str = DEFAULT_LAYOUT, shp_path: str = None,
                   y_slice: float = DEFAULT_Y_SLICE):
    """Build a 2-panel figure (horizon=1×2 or vertical=2×1); colorbar on the right."""
    _apply_global_style()

    cfd_p98 = float(np.nanpercentile(cfd_data['wind_speed'], 98))
    x0, x1, z0, z1 = cfd_xz_limits(cfd_data)
    xz_lim = (x0, x1, z0, z1)

    if wrf_data is not None:
        wrf_lon_to_local_x(wrf_data, y_slice)
        in_view = (
            (wrf_data['height'] >= z0) & (wrf_data['height'] <= z1)
        )
        wrf_ws_view = wrf_data['wind_speed'][in_view]
        wrf_p98 = float(np.nanpercentile(
            wrf_ws_view if np.any(np.isfinite(wrf_ws_view)) else wrf_data['wind_speed'],
            98,
        ))
        x_wrf = wrf_data['x'][0, :]
        print(
            f"  WRF native X : [{float(np.nanmin(x_wrf)):.0f}, "
            f"{float(np.nanmax(x_wrf)):.0f}] m  "
            f"({wrf_data['x'].shape[1]} columns, no regrid)"
        )
    else:
        wrf_p98 = cfd_p98
    shared_vmax = min(_nice_vmax(max(cfd_p98, wrf_p98)), COLORBAR_VMAX_CAP)
    print(
        f"  Shared color  : horizontal WS  |  "
        f"CFD p98={cfd_p98:.2f}  WRF p98={wrf_p98:.2f}  →  vmax={shared_vmax:g} m/s "
        f"(cap={COLORBAR_VMAX_CAP:g})"
    )
    print(
        f"  Shared domain : X [{x0:.0f},{x1:.0f}] m @ Y={y_slice:g} m;  "
        f"Z [{z0:.0f},{z1:.0f}] m  (both panels)"
    )

    shp = shp_path or os.path.join(_repo_root(), DEFAULT_SHP)
    building_polys_x = []
    if os.path.exists(shp):
        building_polys_x = load_building_xz_rects_x(
            shp, y_slice=y_slice, x_lim=(x0, x1),
        )
    elif shp_path:
        warnings.warn(f"Building shapefile not found: {shp}")

    if layout == "horizon":
        fig = plt.figure(figsize=(14.0, 5.2))
        panel_h = 0.70
        gap = 0.05
        panel_bottom = 0.16
        panel_w = 0.38
        panel_left = 0.07
        cfd_left = panel_left + panel_w + gap
        cbar_left = cfd_left + panel_w + 0.02
        cbar_w, cbar_h, cbar_bottom = 0.018, panel_h, panel_bottom

        ax_wrf = fig.add_axes([panel_left, panel_bottom, panel_w, panel_h])
        ax_cfd = fig.add_axes([cfd_left, panel_bottom, panel_w, panel_h])
        cax = fig.add_axes([cbar_left, cbar_bottom, cbar_w, cbar_h])
    else:
        fig = plt.figure(figsize=(12.0, 11.0))
        panel_h = 0.34
        gap = 0.10
        panel_bottom = 0.10
        panel_left = 0.12
        panel_w = 0.68
        wrf_bottom = panel_bottom + panel_h + gap
        cbar_left = 0.84
        cbar_w = 0.025
        cbar_h = 2 * panel_h + gap
        cbar_bottom = panel_bottom

        ax_cfd = fig.add_axes([panel_left, panel_bottom, panel_w, panel_h])
        ax_wrf = fig.add_axes([panel_left, wrf_bottom, panel_w, panel_h])
        cax = fig.add_axes([cbar_left, cbar_bottom, cbar_w, cbar_h])

    wrf_label = '(a) WRF'
    cfd_label = '(b) WRF-to-OpenFOAM'

    mappable = None
    if wrf_data is not None:
        mappable = draw_wrf_panel(
            ax_wrf, wrf_data, vmax=shared_vmax,
            max_height=max_height, cfd_top=cfd_top,
            label=wrf_label, building_polys=building_polys_x,
            xz_lim=xz_lim,
        )
    else:
        ax_wrf.text(0.5, 0.5, 'WRF data unavailable\n(file not found)',
                    ha='center', va='center', transform=ax_wrf.transAxes,
                    fontsize=12, color='grey')
        _add_panel_label(ax_wrf, wrf_label)
        ax_wrf.set_xlim(x0, x1)
        ax_wrf.set_ylim(z0, z1)

    hb_cfd = draw_cfd_panel(
        ax_cfd, cfd_data, vmax=shared_vmax, label=cfd_label, xz_lim=xz_lim,
        building_polys=building_polys_x,
    )
    if mappable is None:
        mappable = hb_cfd
    mappable.set_clim(0, shared_vmax)
    hb_cfd.set_clim(0, shared_vmax)
    add_wind_speed_colorbar(
        fig, mappable, cax=cax, label='Wind Speed (m/s)',
    )

    if layout == "horizon":
        ax_wrf.set_position([panel_left, panel_bottom, panel_w, panel_h])
        ax_cfd.set_position([cfd_left, panel_bottom, panel_w, panel_h])
    else:
        ax_wrf.set_position([panel_left, wrf_bottom, panel_w, panel_h])
        ax_cfd.set_position([panel_left, panel_bottom, panel_w, panel_h])
    cax.set_position([cbar_left, cbar_bottom, cbar_w, cbar_h])

    stamp = title_timestamp(case_label)
    fig.suptitle(
        f'X-Z Vertical Wind Field at Y = {y_slice:g} m — {stamp}',
        fontsize=13, fontweight='bold', y=0.975,
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"\nDONE: Figure saved -> {output_path}  (300 DPI)\n")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description='WRF–CFD X-Z wind field 2-panel comparison figure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('cfd_dir',
                   help='Path to the CFD run directory')
    p.add_argument('--y-slice', type=float, default=DEFAULT_Y_SLICE, metavar='Y',
                   help='OpenFOAM Y-slice in metres; reads '
                        f'postProcessing/y<Y>m.csv (default: {DEFAULT_Y_SLICE})')
    p.add_argument('--wrf-nc', default=None,
                   help='Override auto-detected WRF NetCDF file path')
    p.add_argument('--cfd-csv', default=None,
                   help='Override auto-detected CFD CSV path '
                        '(default: <cfd_dir>/postProcessing/y<y-slice>m.csv)')
    p.add_argument('--layout', choices=('horizon', 'vertical'),
                   default=DEFAULT_LAYOUT,
                   help='Panel arrangement: horizon=1×2 (default), vertical=2×1')
    p.add_argument('--output', default=None,
                   help='Output PNG path (default: results/wrf_openfoam/xz_wrf_cfd/'
                        '<horizon|vertical>_layout/xz_wrf_cfd_<YYYYMMDD_HHMM>.png)')
    p.add_argument('--shp', default=None,
                   help='Building shapefile for white silhouettes on both panels '
                        f'(default: {DEFAULT_SHP})')
    p.add_argument('--lat', type=float, default=None,
                   help='WRF south-north row target latitude '
                        '(default: from Y-slice via origin transform)')
    p.add_argument('--lon', type=float, default=None,
                   help='WRF west-east crop centre longitude '
                        '(default: from CFD X mid via origin transform)')
    p.add_argument('--lat-tol', type=float, default=None,
                   help=f'Unused for row pick (kept for CLI compat; default {LAT_TOL})')
    p.add_argument('--lon-tol', type=float, default=None,
                   help='WRF lon crop half-width in deg (default: from CFD X range)')
    p.add_argument('--max-height', type=float, default=MAX_HEIGHT)
    p.add_argument('--cfd-top', type=float, default=CFD_TOP)
    p.add_argument('--no-wrf', action='store_true',
                   help='Skip WRF panel even if the file is available')
    return p


def main():
    args = build_parser().parse_args()
    cfd_dir = args.cfd_dir.rstrip('/')

    wrf_nc_path, _, wrf_time = infer_paths(cfd_dir, args.y_slice)
    if args.wrf_nc:
        wrf_nc_path = args.wrf_nc
    wrf_nc_path = resolve_existing_wrf_nc_path(wrf_nc_path)
    cfd_csv = resolve_cfd_csv(cfd_dir, args.y_slice, args.cfd_csv)
    y_slice = parse_y_slice_from_csv(cfd_csv, fallback=args.y_slice)

    basename = os.path.basename(cfd_dir)
    output_path = args.output or default_output_path(cfd_dir, layout=args.layout)
    shp_path = args.shp or os.path.join(_repo_root(), DEFAULT_SHP)
    wrf_exp = "W_myExp05" if "W_myExp05" in wrf_nc_path else "W_myExp03"

    print("=" * 64)
    print("  WRF–CFD X-Z Comparison (2-panel)")
    print("=" * 64)
    print(f"  Y-slice      : {y_slice:g} m")
    print(f"  Layout       : {args.layout} "
          f"({'1×2' if args.layout == 'horizon' else '2×1'})")
    print(f"  CFD CSV      : {cfd_csv}")
    print(f"  WRF source   : {wrf_exp}  (≥09-07 → Exp05, else Exp03)")
    print(f"  WRF nc file  : {wrf_nc_path}")
    print(f"  Building SHP : {shp_path}")
    print(f"  Output       : {output_path}")
    print("=" * 64)

    print("\n[1/2] Loading CFD CSV …")
    cfd_data = load_cfd_csv(cfd_csv)
    print(f"      {len(cfd_data['x']):,} points  |  "
          f"WS range [{cfd_data['wind_speed'].min():.2f}, "
          f"{cfd_data['wind_speed'].max():.2f}] m/s")

    x0, x1, _, _ = cfd_xz_limits(cfd_data)
    lon0, lon1 = lon_limits_from_x_range(x0, x1, y_slice)
    target_lat = args.lat if args.lat is not None else lat_of_y_slice(y_slice)
    target_lon = args.lon if args.lon is not None else 0.5 * (lon0 + lon1)
    lon_tol = args.lon_tol if args.lon_tol is not None else 0.55 * (lon1 - lon0)
    lat_tol = args.lat_tol if args.lat_tol is not None else LAT_TOL

    wrf_data = None
    if not args.no_wrf:
        if not os.path.exists(wrf_nc_path):
            warnings.warn(
                f"WRF file not found: {wrf_nc_path}\n"
                "WRF panel will show a placeholder. Use --wrf-nc to override.")
        else:
            print(f"\n[2/2] Loading WRF data ...  ({wrf_time})")
            print(f"      Target lat={target_lat:.6f} N, lon centre={target_lon:.6f} E, "
                  f"lon_tol=±{lon_tol:.5f}°")
            wrf_data = extract_wrf_xz(
                wrf_nc_path,
                target_lat=target_lat, target_lon=target_lon,
                lat_tol=lat_tol, lon_tol=lon_tol,
                max_height=args.max_height,
            )
            print(f"      WS range: "
                  f"[{np.nanmin(wrf_data['wind_speed']):.2f}, "
                  f"{np.nanmax(wrf_data['wind_speed']):.2f}] m/s")

    print("\nRendering figure …")
    compose_figure(
        wrf_data, cfd_data,
        case_label=basename,
        output_path=output_path,
        max_height=args.max_height,
        cfd_top=args.cfd_top,
        layout=args.layout,
        shp_path=shp_path,
        y_slice=y_slice,
    )


if __name__ == "__main__":
    main()
