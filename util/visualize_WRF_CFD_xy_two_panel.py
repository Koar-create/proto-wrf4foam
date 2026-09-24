"""
WRF–CFD X-Y Horizontal Wind Field Comparison (2-panel)
======================================================
Produces a publication-quality 2-panel figure at a fixed height.

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
    python visualize_WRF_CFD_xy_two_panel.py  /path/to/CFD_run_directory
    python visualize_WRF_CFD_xy_two_panel.py  /path/to/CFD_run_directory --height 120
    python visualize_WRF_CFD_xy_two_panel.py  /path/to/CFD_run_directory --layout vertical

Example
-------
    # default height 100 m → postProcessing/100m.csv
    python visualize_WRF_CFD_xy_two_panel.py \\
        steady_experiments_finer_ABL/20250901_1000_two_boundaries_as_outlet

    # other slice heights → 30m.csv / 60m.csv / 120m.csv
    python visualize_WRF_CFD_xy_two_panel.py \\
        steady_experiments_finer_ABL/20250901_1000_two_boundaries_as_outlet --height 30
    python visualize_WRF_CFD_xy_two_panel.py \\
        steady_experiments_finer_ABL/20250901_1000_two_boundaries_as_outlet --height 60
    python visualize_WRF_CFD_xy_two_panel.py \\
        steady_experiments_finer_ABL/20250901_1000_two_boundaries_as_outlet --height 120

Path inference
--------------
Given CFD path ``<root>/<YYYYMMDD_HHMM>_<tag>`` the script resolves:

* WRF nc  →  ``W_myExp03|05/auxhist2/tmp/auxhist2_d03_<YYYY-MM-DD_HH:MM:00>_tmp.nc``
  (``W_myExp03`` for dates ≤ 09-06; ``W_myExp05`` for ≥ 09-07).
  On Windows, ``:`` in the filename is often stored as ``%3A``; path resolution
  accepts both forms.
* CFD CSV →  ``<cfd_dir>/postProcessing/<height>m.csv``  (default height=100)
* PNG out →  ``results/wrf_openfoam/xy_wrf_cfd/<horizon|vertical>_layout/xy_wrf_cfd_<YYYYMMDD_HHMM>_<height>m.png``
* SHP     →  ``data/Guangzhou_shp_file/project_UTM49/Export_Output.shp`` (white basemap on panel a)

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
from scipy.interpolate import griddata

# Sibling util: fixed Guangzhou origin lon/lat ↔ local XY (UTM49N)
_UTIL_DIR = os.path.dirname(os.path.abspath(__file__))
if _UTIL_DIR not in sys.path:
    sys.path.insert(0, _UTIL_DIR)
from convert_lonlat_xy_origin import ORIGIN_LAT, ORIGIN_LON, xy_to_lonlat  # noqa: E402

# ---------------------------------------------------------------------------
# CONSTANTS / DEFAULTS
# ---------------------------------------------------------------------------
WRF_ROOT_EXP03   = os.path.join("W_myExp03", "auxhist2", "tmp")
WRF_ROOT_EXP05   = os.path.join("W_myExp05", "auxhist2", "tmp")
# Dates on/after this calendar day use W_myExp05; earlier dates use W_myExp03
WRF_EXP05_START  = (2025, 9, 7)
WRF_NC_TEMPLATE  = "auxhist2_d03_{wrf_time}_tmp.nc"
DEFAULT_HEIGHT   = 100
DEFAULT_LAYOUT   = "horizon"
LAYOUT_DIRS      = {
    "horizon": "horizon_layout",
    "vertical": "vertical_layout",
}

# Default crop centre matches building/OpenFOAM origin (see convert_lonlat_xy_origin.py)
TARGET_LAT       = ORIGIN_LAT
TARGET_LON       = ORIGIN_LON
LAT_TOL          = 0.05
LON_TOL          = 0.05

# Quiver: scale_units='width' → scale = m/s per axes-width (larger = shorter).
# Keep typical (~10 m/s) arrow ≈ 2/3 of vector spacing to avoid tip-to-tail streaks.
QUIVER_GRID_CFD  = 12          # CFD vector grid
QUIVER_GRID_WRF  = 9           # WRF vectors regridded (mesoscale field is smooth)
# scale_units='width': arrow length ≈ U/scale of axes width.
# Smaller scale → longer arrows (was 160 → tiny ~2.5% width at 4 m/s).
QUIVER_SCALE     = 35          # ~4 m/s → ~11% of panel width
QUIVER_KEY_SPEED = 1.0
QUIVER_WIDTH_WRF = 0.005
QUIVER_WIDTH_CFD = 0.004
QUIVER_WIDTH     = QUIVER_WIDTH_CFD  # alias for multi-height script
HEXBIN_GRID      = 120

WIND_SPEED_COLORBAR_TICK_FORMAT = '%g'
RESULTS_XY_DIR   = os.path.join("results", "wrf_openfoam", "xy_wrf_cfd")
DEFAULT_SHP      = os.path.join(
    "data", "Guangzhou_shp_file", "project_UTM49", "Export_Output.shp",
)


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


def cfd_xy_square_limits(cfd_data: dict):
    """
    Axis-aligned square matching the CFD panel view
    ``[xc ± side/2] × [yc ± side/2]`` in local metres.
    """
    x = cfd_data['x']
    y = cfd_data['y']
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    side = max(x_max - x_min, y_max - y_min)
    xc = 0.5 * (x_min + x_max)
    yc = 0.5 * (y_min + y_max)
    x0, x1 = xc - 0.5 * side, xc + 0.5 * side
    y0, y1 = yc - 0.5 * side, yc + 0.5 * side
    return x0, x1, y0, y1


def lonlat_limits_from_xy_square(x0, x1, y0, y1):
    """
    Map a local-XY square to a lon/lat box via ``xy_to_lonlat`` (same origin
    as OpenFOAM / buildings). Uses the four corners so panel (a) covers the
    same geographic footprint as panel (b).
    """
    corners = [
        xy_to_lonlat(x0, y0),
        xy_to_lonlat(x1, y0),
        xy_to_lonlat(x1, y1),
        xy_to_lonlat(x0, y1),
    ]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    lon0, lon1 = min(lons), max(lons)
    lat0, lat1 = min(lats), max(lats)
    return lon0, lon1, lat0, lat1


def _square_pad_limits(u0, u1, v0, v1):
    """Pad the shorter axis so (u,v) limits form a square in data units."""
    du, dv = u1 - u0, v1 - v0
    side = max(du, dv)
    uc, vc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    return uc - 0.5 * side, uc + 0.5 * side, vc - 0.5 * side, vc + 0.5 * side


# ---------------------------------------------------------------------------
# PATH HELPERS
# ---------------------------------------------------------------------------

def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_output_path(cfd_dir: str, height: int, layout: str = DEFAULT_LAYOUT) -> str:
    """
    ``results/wrf_openfoam/xy_wrf_cfd/<horizon|vertical>_layout/xy_wrf_cfd_<YYYYMMDD_HHMM>_<height>m.png``
    """
    cfd_dir = cfd_dir.rstrip(os.sep)
    case = os.path.basename(cfd_dir)
    m = re.match(r"(\d{8}_\d{4})", case)
    stamp = m.group(1) if m else case
    layout_dir = LAYOUT_DIRS.get(layout, LAYOUT_DIRS[DEFAULT_LAYOUT])
    return os.path.join(
        _repo_root(), RESULTS_XY_DIR, layout_dir,
        f"xy_wrf_cfd_{stamp}_{height}m.png",
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


def infer_paths(cfd_dir: str, height: int):
    wrf_time = parse_timestamp_from_cfd_dir(cfd_dir)
    nc_filename = WRF_NC_TEMPLATE.format(wrf_time=wrf_time)
    wrf_nc_path = resolve_existing_wrf_nc_path(
        os.path.join(wrf_root_for_time(wrf_time), nc_filename)
    )
    csv_relpath = os.path.join("postProcessing", f"{height}m.csv")
    cfd_csv = os.path.join(cfd_dir, csv_relpath)
    return wrf_nc_path, cfd_csv, wrf_time


def list_available_slice_heights(cfd_dir: str) -> list:
    """Return sorted heights (m) for which ``postProcessing/<H>m.csv`` exists."""
    post = os.path.join(cfd_dir, "postProcessing")
    if not os.path.isdir(post):
        return []
    heights = []
    for name in os.listdir(post):
        m = re.fullmatch(r"(\d+)m\.csv", name)
        if m:
            heights.append(int(m.group(1)))
    return sorted(heights)


def resolve_cfd_csv(cfd_dir: str, height: int, cfd_csv_override: str = None) -> str:
    """Resolve CFD slice CSV; raise with available heights if missing."""
    if cfd_csv_override:
        cfd_csv = cfd_csv_override
    else:
        cfd_csv = os.path.join(cfd_dir, "postProcessing", f"{height}m.csv")
    if os.path.isfile(cfd_csv):
        return cfd_csv
    available = list_available_slice_heights(cfd_dir)
    avail_txt = (
        ", ".join(f"{h}m" for h in available) if available else "(none found)"
    )
    raise FileNotFoundError(
        f"CFD slice CSV not found: {cfd_csv}\n"
        f"  Requested --height {height} → postProcessing/{height}m.csv\n"
        f"  Available slice CSVs in {cfd_dir}/postProcessing/: {avail_txt}\n"
        f"  Tip: use e.g. --height 30 / 60 / 100 / 120, or pass --cfd-csv explicitly."
    )


# ---------------------------------------------------------------------------
# WRF DATA EXTRACTION
# ---------------------------------------------------------------------------

def _destagger_np(arr: np.ndarray, axis: int) -> np.ndarray:
    slc_lo = [slice(None)] * arr.ndim
    slc_hi = [slice(None)] * arr.ndim
    slc_lo[axis] = slice(None, -1)
    slc_hi[axis] = slice(1, None)
    return 0.5 * (arr[tuple(slc_lo)] + arr[tuple(slc_hi)])


def extract_wrf_xy(nc_path: str, target_height: float,
                   target_lat=TARGET_LAT, target_lon=TARGET_LON,
                   lat_tol=LAT_TOL, lon_tol=LON_TOL):
    """Extract an X-Y horizontal plane from a WRF file at *target_height* (m)."""
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"WRF file not found: {nc_path}")

    print(f"  Opening: {nc_path}")
    ds = xr.open_dataset(nc_path)

    def get_val(name):
        v = ds[name]
        return v.values[0] if 'Time' in v.dims else v.values

    ph = get_val('PH')
    phb = get_val('PHB')
    z_stag = (ph + phb) / 9.81
    z = _destagger_np(z_stag, axis=0)

    u_stag = get_val('U')
    u = _destagger_np(u_stag, axis=2)

    v_stag = get_val('V')
    v = _destagger_np(v_stag, axis=1)

    lats = get_val('XLAT')
    lons = get_val('XLONG')
    if lats.ndim == 3:
        lats = lats[0]
    if lons.ndim == 3:
        lons = lons[0]

    lat_mask = (lats >= target_lat - lat_tol) & (lats <= target_lat + lat_tol)
    lon_mask = (lons >= target_lon - lon_tol) & (lons <= target_lon + lon_tol)
    combined_mask = lat_mask & lon_mask

    rows = np.any(combined_mask, axis=1)
    cols = np.any(combined_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        print("  [!] Target area outside WRF domain. Using full domain.")
        i_min, i_max = 0, lats.shape[0]
        j_min, j_max = 0, lats.shape[1]
    else:
        i_min, i_max = np.where(rows)[0][[0, -1]]
        j_min, j_max = np.where(cols)[0][[0, -1]]
        i_max += 1
        j_max += 1

    z_crop = z[:, i_min:i_max, j_min:j_max]
    u_crop = u[:, i_min:i_max, j_min:j_max]
    v_crop = v[:, i_min:i_max, j_min:j_max]
    lat_crop = lats[i_min:i_max, j_min:j_max]
    lon_crop = lons[i_min:i_max, j_min:j_max]

    ny, nx = lat_crop.shape
    u_interp = np.zeros((ny, nx))
    v_interp = np.zeros((ny, nx))

    for i in range(ny):
        for j in range(nx):
            u_interp[i, j] = np.interp(target_height, z_crop[:, i, j], u_crop[:, i, j])
            v_interp[i, j] = np.interp(target_height, z_crop[:, i, j], v_crop[:, i, j])

    ws_interp = np.sqrt(u_interp**2 + v_interp**2)
    ds.close()

    print(f"  Extracted plane at {target_height:g} m: shape={ws_interp.shape}, "
          f"WS=[{np.nanmin(ws_interp):.2f}, {np.nanmax(ws_interp):.2f}] m/s")

    return dict(lon=lon_crop, lat=lat_crop,
                u=u_interp, v=v_interp,
                wind_speed=ws_interp)


# ---------------------------------------------------------------------------
# CFD CSV
# ---------------------------------------------------------------------------

def load_cfd_csv(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CFD CSV not found: {csv_path}")

    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=100_000):
        req = ['Coords:0', 'Coords:1', 'U:0', 'U:1']
        if any(c not in chunk.columns for c in req):
            raise KeyError(f"Missing columns in {csv_path}. Expected: {req}")
        keep = req + (['vtkValidPointMask'] if 'vtkValidPointMask' in chunk.columns else [])
        chunks.append(chunk[keep].astype(float))

    df = pd.concat(chunks, ignore_index=True)
    u0 = df['U:0'].values
    u1 = df['U:1'].values
    out = dict(x=df['Coords:0'].values, y=df['Coords:1'].values,
               u0=u0, u1=u1, wind_speed=np.sqrt(u0**2 + u1**2))
    if 'vtkValidPointMask' in df.columns:
        # Resampled points inside a building solid carry no fluid solution and
        # are written as zeros; flag them so they are never plotted as calm air.
        out['valid'] = df['vtkValidPointMask'].values == 1
    return out


# ---------------------------------------------------------------------------
# BUILDING SHAPEFILE (UTM49N → lon/lat basemap for WRF panel)
# ---------------------------------------------------------------------------

def _rings_from_shape(shp) -> list:
    """Exterior rings from a pyshp polygon (skip holes / tiny slivers)."""
    pts = np.asarray(shp.points, dtype=float)
    if pts.size == 0:
        return []
    parts = list(shp.parts) + [len(pts)]
    rings = []
    for i in range(len(parts) - 1):
        if i > 0:
            break  # first ring = exterior
        ring = pts[parts[i]:parts[i + 1]]
        if ring.shape[0] >= 2 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if ring.shape[0] < 3:
            continue
        x, y = ring[:, 0], ring[:, 1]
        area = abs(0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))
        if area < 5.0:  # m²
            continue
        rings.append(ring)
    return rings


def load_building_rings_lonlat(shp_path: str, lon_lim=None, lat_lim=None,
                               pad: float = 0.002):
    """
    Load building footprints from UTM Zone 49N shapefile, reproject to WGS84
    lon/lat, optionally clipped to a padded lon/lat box.
    """
    import shapefile
    from pyproj import Transformer

    if not os.path.exists(shp_path):
        warnings.warn(f"Building shapefile not found: {shp_path}")
        return []

    transformer = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
    reader = shapefile.Reader(shp_path, encoding="gbk")

    lon0 = lat0 = lon1 = lat1 = None
    if lon_lim is not None and lat_lim is not None:
        lon0, lon1 = lon_lim[0] - pad, lon_lim[1] + pad
        lat0, lat1 = lat_lim[0] - pad, lat_lim[1] + pad

    rings = []
    for shp in reader.shapes():
        if shp.shapeType not in (5, 15, 25):
            continue
        for ring_utm in _rings_from_shape(shp):
            lon, lat = transformer.transform(ring_utm[:, 0], ring_utm[:, 1])
            ring = np.column_stack([np.asarray(lon), np.asarray(lat)])
            if lon0 is not None:
                if (ring[:, 0].max() < lon0 or ring[:, 0].min() > lon1
                        or ring[:, 1].max() < lat0 or ring[:, 1].min() > lat1):
                    continue
            rings.append(ring)

    print(f"  Building basemap: {len(rings)} footprints from {os.path.basename(shp_path)}")
    return rings


def draw_building_basemap(ax, rings, facecolor='white', edgecolor='#c8c8c8',
                          lw=0.25, zorder=1, alpha=1.0):
    """White building footprints as a silent basemap (no legend)."""
    if not rings:
        return
    pc = PolyCollection(
        rings, facecolors=facecolor, edgecolors=edgecolor,
        linewidths=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_collection(pc)


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
        'font.size': 17,
        'axes.titlesize': 18,
        'axes.labelsize': 17,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'axes.linewidth': 0.8,
        'figure.dpi': 150,
    })
    _STYLE_DONE = True


def _add_panel_label(ax, label, fontsize=19):
    ax.text(0.015, 0.965, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left',
            zorder=20, clip_on=False,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))


def add_wind_speed_colorbar(fig, mappable, ax=None, label='Wind Speed (m/s)',
                            ticks=None, tick_format=None, cax=None):
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
    cb.ax.tick_params(labelsize=14)
    return cb


def _quiver_key_speed(vmax: float) -> float:
    """Pick a readable reference arrow length from the colorbar ceiling."""
    if vmax <= 2:
        return 1.0
    if vmax <= 5:
        return 2.0
    return 4.0


def draw_wrf_panel(ax, data: dict, vmax=None,
                   label='(a) WRF (mesoscale boundary)',
                   show_vectors=True, quiver_key_speed=None,
                   building_rings=None, lonlat_lim=None,
                   show_quiver_key=True, quiver_key_xy=(0.82, 1.035),
                   add_panel_label=True):
    lon = data['lon']
    lat = data['lat']
    ws = data['wind_speed']
    u = data['u']
    v = data['v']

    if vmax is None:
        vmax = np.nanpercentile(ws, 98)
    key_u = quiver_key_speed if quiver_key_speed is not None else _quiver_key_speed(vmax)

    # White building basemap: fill under semi-transparent wind, soft white overlay + outline
    rings = building_rings or []
    draw_building_basemap(ax, rings, facecolor='white', edgecolor='none', zorder=1)

    qm = ax.pcolormesh(lon, lat, ws,
                       vmin=0, vmax=vmax, cmap='viridis',
                       shading='auto', alpha=0.70, rasterized=True,
                       zorder=2)

    draw_building_basemap(
        ax, rings, facecolor='white', edgecolor='#8a8a8a', lw=0.25,
        alpha=0.45, zorder=3,
    )

    if show_vectors:
        # Regrid onto a regular lon/lat mesh so sparse WRF crops still show
        # a clean vector field (mesoscale wind is smooth at this scale).
        if lonlat_lim is not None:
            lon0, lon1, lat0, lat1 = lonlat_lim
        else:
            lon0, lon1 = float(np.nanmin(lon)), float(np.nanmax(lon))
            lat0, lat1 = float(np.nanmin(lat)), float(np.nanmax(lat))
        lon_q = np.linspace(lon0, lon1, QUIVER_GRID_WRF)
        lat_q = np.linspace(lat0, lat1, QUIVER_GRID_WRF)
        Lon_q, Lat_q = np.meshgrid(lon_q, lat_q)
        pts = np.column_stack([lon.ravel(), lat.ravel()])
        # nearest fills the full view; linear alone leaves NaNs outside the
        # coarse WRF crop hull and makes panel (a) look almost empty.
        gu = griddata(pts, u.ravel(), (Lon_q, Lat_q), method='nearest')
        gv = griddata(pts, v.ravel(), (Lon_q, Lat_q), method='nearest')
        mask = np.isfinite(gu) & np.isfinite(gv)

        qv = ax.quiver(Lon_q[mask], Lat_q[mask], gu[mask], gv[mask],
                       color='black', alpha=0.90,
                       scale_units='width', scale=QUIVER_SCALE,
                       width=QUIVER_WIDTH_WRF,
                       headwidth=4.0, headlength=4.5, headaxislength=4.0,
                       minshaft=1.0, pivot='tail', zorder=5)

        if show_quiver_key:
            ax.quiverkey(qv, X=quiver_key_xy[0], Y=quiver_key_xy[1], U=key_u,
                         label=f'{key_u:g} m/s',
                         labelpos='E', coordinates='axes',
                         fontproperties={'family': 'serif', 'size': 14, 'weight': 'bold'},
                         labelsep=0.04)

    if lonlat_lim is not None:
        lon_min, lon_max, lat_min, lat_max = lonlat_lim
    else:
        lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
        lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    # Pad shorter axis so the box is square in degree units (keeps
    # adjustable='datalim' from rewriting limits). When lonlat_lim comes from
    # the CFD XY square the pad is only ~O(100 m).
    lon_min, lon_max, lat_min, lat_max = _square_pad_limits(
        lon_min, lon_max, lat_min, lat_max,
    )
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.margins(0)

    ax.set_xlabel('Longitude (°E)', fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontweight='bold')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.grid(True, alpha=0.25, ls='--', lw=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_aspect('equal', adjustable='datalim')
    if add_panel_label and label:
        _add_panel_label(ax, label)
    return qm


def draw_cfd_panel(ax, data: dict, vmax=None,
                   label='(b) CFD (OpenFOAM)',
                   show_vectors=True, quiver_key_speed=None,
                   xy_lim=None,
                   show_quiver_key=True, quiver_key_xy=(0.82, 1.035),
                   add_panel_label=True, zone_mask=None, zone_axis=None,
                   zone_color='#ff2d95'):
    x = data['x']
    y = data['y']
    u0 = data['u0']
    u1 = data['u1']
    ws = data['wind_speed']
    # Points inside building solids (vtkValidPointMask = 0) hold the zero fill
    # of the resampler, not a flow solution. Keeping them out of the field, the
    # colour scale and the vectors leaves them blank, matching the white
    # building footprints drawn on the WRF panels.
    valid = data.get('valid')
    if valid is None:
        valid = np.ones(ws.shape, dtype=bool)

    if vmax is None:
        vmax = np.nanpercentile(ws[valid], 98)
    key_u = quiver_key_speed if quiver_key_speed is not None else _quiver_key_speed(vmax)

    if xy_lim is not None:
        x0, x1, y0, y1 = xy_lim
    else:
        x0, x1, y0, y1 = cfd_xy_square_limits(data)
    extent = (x0, x1, y0, y1)

    hb = ax.hexbin(x[valid], y[valid], C=ws[valid],
                   gridsize=HEXBIN_GRID, extent=extent, cmap='viridis',
                   reduce_C_function=np.mean,
                   vmin=0, vmax=vmax,
                   alpha=0.85, edgecolors='none', rasterized=True, zorder=2)

    if show_vectors and np.count_nonzero(valid) > 3:
        xv, yv = x[valid], y[valid]
        x_g, y_g = np.mgrid[x0:x1:complex(0, QUIVER_GRID_CFD),
                            y0:y1:complex(0, QUIVER_GRID_CFD)]
        sub = max(1, len(xv) // 100_000)
        gu0 = griddata((xv[::sub], yv[::sub]), u0[valid][::sub], (x_g, y_g), method='linear')
        gu1 = griddata((xv[::sub], yv[::sub]), u1[valid][::sub], (x_g, y_g), method='linear')
        mask = np.isfinite(gu0) & np.isfinite(gu1)

        qv = ax.quiver(x_g.ravel()[mask.ravel()], y_g.ravel()[mask.ravel()],
                       gu0.ravel()[mask.ravel()], gu1.ravel()[mask.ravel()],
                       color='black', alpha=0.90,
                       scale_units='width', scale=QUIVER_SCALE,
                       width=QUIVER_WIDTH_CFD,
                       headwidth=4.0, headlength=4.5, headaxislength=4.0,
                       minshaft=1.0, pivot='tail', zorder=5)

        if show_quiver_key:
            ax.quiverkey(qv, X=quiver_key_xy[0], Y=quiver_key_xy[1], U=key_u,
                         label=f'{key_u:g} m/s',
                         labelpos='E', coordinates='axes',
                         fontproperties={'family': 'serif', 'size': 14, 'weight': 'bold'},
                         labelsep=0.04)

    if zone_mask is not None and zone_axis is not None:
        ax.contour(zone_axis, zone_axis, zone_mask.astype(float),
                   levels=[0.5], colors=[zone_color],
                   linewidths=1.3, zorder=6)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.margins(0)

    ax.set_xlabel('X (m)', fontweight='bold')
    ax.set_ylabel('Y (m)', fontweight='bold')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.25, ls='--', lw=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if add_panel_label and label:
        _add_panel_label(ax, label)
    return hb


def short_case_label(case: str) -> str:
    """Keep only YYYYMMDD_HHMM; drop tags like ``two_boundaries_as_outlet``."""
    m = re.match(r"(\d{8}_\d{4})", case)
    return m.group(1) if m else case


def compose_figure(wrf_data, cfd_data, case_label: str, output_path: str,
                   height: float, layout: str = DEFAULT_LAYOUT,
                   shp_path: str = None):
    """Build a 2-panel figure (horizon=1×2 or vertical=2×1); colorbar on the right."""
    _apply_global_style()

    cfd_p98 = float(np.nanpercentile(cfd_data['wind_speed'], 98))
    wrf_p98 = (
        float(np.nanpercentile(wrf_data['wind_speed'], 98))
        if wrf_data is not None else cfd_p98
    )
    shared_vmax = _nice_vmax(max(cfd_p98, wrf_p98))
    key_u = _quiver_key_speed(shared_vmax)

    # Shared geographic footprint: CFD XY square → lon/lat for WRF panel
    xy_lim = cfd_xy_square_limits(cfd_data)
    lon0, lon1, lat0, lat1 = lonlat_limits_from_xy_square(*xy_lim)
    lonlat_lim = (lon0, lon1, lat0, lat1)
    print(
        f"  Shared domain : XY [{xy_lim[0]:.0f},{xy_lim[1]:.0f}]×"
        f"[{xy_lim[2]:.0f},{xy_lim[3]:.0f}] m  →  "
        f"lon/lat [{lon0:.5f},{lon1:.5f}]×[{lat0:.5f},{lat1:.5f}]"
    )

    building_rings = []
    if wrf_data is not None:
        shp = shp_path or os.path.join(_repo_root(), DEFAULT_SHP)
        building_rings = load_building_rings_lonlat(
            shp,
            lon_lim=(lon0, lon1),
            lat_lim=(lat0, lat1),
        )

    if layout == "horizon":
        # 1×2: identical square panels side-by-side; vertical colorbar on the right
        fig_w, fig_h = 11.8, 5.6
        fig = plt.figure(figsize=(fig_w, fig_h))
        panel_h = 0.72
        gap = 0.055
        panel_bottom = 0.14
        panel_w = panel_h * fig_h / fig_w
        panel_left = 0.075
        cfd_left = panel_left + panel_w + gap
        cbar_left = cfd_left + panel_w + 0.025
        cbar_w, cbar_h, cbar_bottom = 0.022, panel_h, panel_bottom

        ax_wrf = fig.add_axes([panel_left, panel_bottom, panel_w, panel_h])
        ax_cfd = fig.add_axes([cfd_left, panel_bottom, panel_w, panel_h])
        cax = fig.add_axes([cbar_left, cbar_bottom, cbar_w, cbar_h])
    else:
        # 2×1: identical square panels stacked; vertical colorbar spanning both
        fig_w, fig_h = 6.5, 10.2
        fig = plt.figure(figsize=(fig_w, fig_h))
        panel_h = 0.40
        gap = 0.060
        panel_bottom = 0.055
        panel_w = panel_h * fig_h / fig_w
        panel_left = 0.17
        wrf_bottom = panel_bottom + panel_h + gap
        cfd_left = panel_left
        cbar_left = panel_left + panel_w + 0.04
        cbar_w = 0.04
        cbar_h = 2 * panel_h + gap
        cbar_bottom = panel_bottom

        ax_cfd = fig.add_axes([panel_left, panel_bottom, panel_w, panel_h])
        ax_wrf = fig.add_axes([panel_left, wrf_bottom, panel_w, panel_h])
        cax = fig.add_axes([cbar_left, cbar_bottom, cbar_w, cbar_h])

    wrf_label = f'(a) WRF'
    cfd_label = f'(b) WRF-to-OpenFOAM'

    mappable = None
    if wrf_data is not None:
        mappable = draw_wrf_panel(
            ax_wrf, wrf_data, vmax=shared_vmax, label=wrf_label,
            quiver_key_speed=key_u, building_rings=building_rings,
            lonlat_lim=lonlat_lim,
        )
    else:
        ax_wrf.text(0.5, 0.5, 'WRF data unavailable\n(file not found)',
                    ha='center', va='center', transform=ax_wrf.transAxes,
                    fontsize=16, color='grey')
        _add_panel_label(ax_wrf, wrf_label)
        lon_a, lon_b, lat_a, lat_b = _square_pad_limits(lon0, lon1, lat0, lat1)
        ax_wrf.set_xlim(lon_a, lon_b)
        ax_wrf.set_ylim(lat_a, lat_b)

    hb_cfd = draw_cfd_panel(
        ax_cfd, cfd_data, vmax=shared_vmax, label=cfd_label,
        quiver_key_speed=key_u, xy_lim=xy_lim,
    )
    if mappable is None:
        mappable = hb_cfd
    add_wind_speed_colorbar(fig, mappable, cax=cax)

    # Lock panel geometry (limits already square → datalim won't resize)
    if layout == "horizon":
        ax_wrf.set_position([panel_left, panel_bottom, panel_w, panel_h])
        ax_cfd.set_position([cfd_left, panel_bottom, panel_w, panel_h])
    else:
        ax_wrf.set_position([panel_left, wrf_bottom, panel_w, panel_h])
        ax_cfd.set_position([panel_left, panel_bottom, panel_w, panel_h])
    cax.set_position([cbar_left, cbar_bottom, cbar_w, cbar_h])

    fig.suptitle(
        f'X-Y Horizontal Wind Field at {height:g} m',
        fontsize=16, fontweight='bold', y=0.975,
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Do not use bbox_inches='tight' — it re-crops panels to unequal widths
    plt.savefig(output_path, dpi=300)
    print(f"\nDONE: Figure saved -> {output_path}  (300 DPI)\n")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description='WRF–CFD X-Y wind field 2-panel comparison figure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('cfd_dir',
                   help='Path to the CFD run directory')
    p.add_argument('--height', type=int, default=DEFAULT_HEIGHT, metavar='H',
                   help='Horizontal slice height in metres; reads '
                        f'postProcessing/<H>m.csv (default: {DEFAULT_HEIGHT}; '
                        'common values: 30, 60, 100, 120)')
    p.add_argument('--wrf-nc', default=None,
                   help='Override auto-detected WRF NetCDF file path')
    p.add_argument('--cfd-csv', default=None,
                   help='Override auto-detected CFD CSV path '
                        '(default: <cfd_dir>/postProcessing/<height>m.csv)')
    p.add_argument('--layout', choices=('horizon', 'vertical'),
                   default=DEFAULT_LAYOUT,
                   help='Panel arrangement: horizon=1×2 (default), vertical=2×1')
    p.add_argument('--output', default=None,
                   help='Output PNG path (default: results/wrf_openfoam/xy_wrf_cfd/'
                        '<horizon|vertical>_layout/xy_wrf_cfd_<YYYYMMDD_HHMM>_<height>m.png)')
    p.add_argument('--shp', default=None,
                   help='Building shapefile for WRF panel white basemap '
                        f'(default: {DEFAULT_SHP})')
    p.add_argument('--lat', type=float, default=None,
                   help='WRF crop centre latitude (default: from CFD XY domain)')
    p.add_argument('--lon', type=float, default=None,
                   help='WRF crop centre longitude (default: from CFD XY domain)')
    p.add_argument('--lat-tol', type=float, default=None,
                   help='WRF crop half-height in deg (default: from CFD XY domain)')
    p.add_argument('--lon-tol', type=float, default=None,
                   help='WRF crop half-width in deg (default: from CFD XY domain)')
    p.add_argument('--no-wrf', action='store_true',
                   help='Skip WRF panel even if the file is available')
    return p


def main():
    args = build_parser().parse_args()
    cfd_dir = args.cfd_dir.rstrip('/')

    wrf_nc_path, cfd_csv, wrf_time = infer_paths(cfd_dir, args.height)
    if args.wrf_nc:
        wrf_nc_path = args.wrf_nc
    wrf_nc_path = resolve_existing_wrf_nc_path(wrf_nc_path)
    cfd_csv = resolve_cfd_csv(cfd_dir, args.height, args.cfd_csv)

    basename = os.path.basename(cfd_dir)
    output_path = args.output or default_output_path(
        cfd_dir, args.height, layout=args.layout,
    )
    shp_path = args.shp or os.path.join(_repo_root(), DEFAULT_SHP)
    wrf_exp = "W_myExp05" if "W_myExp05" in wrf_nc_path else "W_myExp03"

    print("=" * 64)
    print("  WRF–CFD X-Y Comparison (2-panel)")
    print("=" * 64)
    print(f"  Height       : {args.height} m")
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

    xy_lim = cfd_xy_square_limits(cfd_data)
    lon0, lon1, lat0, lat1 = lonlat_limits_from_xy_square(*xy_lim)
    target_lat = args.lat if args.lat is not None else 0.5 * (lat0 + lat1)
    target_lon = args.lon if args.lon is not None else 0.5 * (lon0 + lon1)
    # Small pad so pcolormesh cells fully cover the shared view
    lat_tol = args.lat_tol if args.lat_tol is not None else 0.55 * (lat1 - lat0)
    lon_tol = args.lon_tol if args.lon_tol is not None else 0.55 * (lon1 - lon0)

    wrf_data = None
    if not args.no_wrf:
        if not os.path.exists(wrf_nc_path):
            warnings.warn(
                f"WRF file not found: {wrf_nc_path}\n"
                "WRF panel will show a placeholder. Use --wrf-nc to override.")
        else:
            print(f"\n[2/2] Loading WRF data ...  ({wrf_time})")
            print(f"      Crop centre ({target_lon:.6f} E, {target_lat:.6f} N), "
                  f"tol ±{lon_tol:.5f}°/±{lat_tol:.5f}°")
            wrf_data = extract_wrf_xy(
                wrf_nc_path, args.height,
                target_lat=target_lat, target_lon=target_lon,
                lat_tol=lat_tol, lon_tol=lon_tol,
            )
            print(f"      Wind speed range: "
                  f"[{np.nanmin(wrf_data['wind_speed']):.2f}, "
                  f"{np.nanmax(wrf_data['wind_speed']):.2f}] m/s")

    print("\nRendering figure …")
    compose_figure(
        wrf_data, cfd_data,
        case_label=basename,
        output_path=output_path,
        height=args.height,
        layout=args.layout,
        shp_path=shp_path,
    )


if __name__ == "__main__":
    main()
