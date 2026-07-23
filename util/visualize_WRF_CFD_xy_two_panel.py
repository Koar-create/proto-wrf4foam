"""
WRF–CFD X-Y Horizontal Wind Field Comparison (2-panel)
======================================================
Produces a publication-quality 2-panel figure at a fixed height:

    ┌────────────────────────────────────────┐
    │  (a) WRF (mesoscale) @ H m             │
    ├────────────────────────────────────────┤
    │  (b) CFD (OpenFOAM) @ H m              │
    └────────────────────────────────────────┘

Usage
-----
    python visualize_WRF_CFD_xy_two_panel.py  /path/to/CFD_run_directory --height 100

Example
-------
    python visualize_WRF_CFD_xy_two_panel.py \\
        steady_experiments_finer_ABL/20250901_0000_two_boundaries_as_outlet --height 100

Path inference
--------------
Given CFD path ``<root>/<YYYYMMDD_HHMM>_<tag>`` the script resolves:

* WRF nc  →  ``W_myExp03|05/auxhist2/tmp/auxhist2_d03_<YYYY-MM-DD_HH:MM:00>_tmp.nc``
  (``W_myExp03`` for dates ≤ 09-06; ``W_myExp05`` for ≥ 09-07)
* CFD CSV →  ``<cfd_dir>/postProcessing/<height>m.csv``
* PNG out →  ``results/wrf_openfoam/xy_wrf_cfd/<experiment_batch>/xy_wrf_cfd_<YYYYMMDD_HHMM>_<height>m.png``

Override with ``--wrf-nc``, ``--cfd-csv``, or ``--output``.
"""

import os
import re
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, FixedLocator
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------
# CONSTANTS / DEFAULTS
# ---------------------------------------------------------------------------
WRF_ROOT_EXP03   = os.path.join("W_myExp03", "auxhist2", "tmp")
WRF_ROOT_EXP05   = os.path.join("W_myExp05", "auxhist2", "tmp")
# Dates on/after this calendar day use W_myExp05; earlier dates use W_myExp03
WRF_EXP05_START  = (2025, 9, 7)
WRF_NC_TEMPLATE  = "auxhist2_d03_{wrf_time}_tmp.nc"
DEFAULT_HEIGHT   = 100

TARGET_LAT       = 23.1211944444
TARGET_LON       = 113.321102778
LAT_TOL          = 0.05
LON_TOL          = 0.05

QUIVER_GRID_CFD  = 9           # sparse CFD vectors
QUIVER_SCALE     = 15          # shared WRF/CFD scale — same U → same arrow length
QUIVER_KEY_SPEED = 1.0
QUIVER_WIDTH     = 0.006
HEXBIN_GRID      = 120

WIND_SPEED_COLORBAR_TICK_FORMAT = '%g'
RESULTS_XY_DIR   = os.path.join("results", "wrf_openfoam", "xy_wrf_cfd")


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
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_output_path(cfd_dir: str, height: int) -> str:
    """
    ``results/wrf_openfoam/xy_wrf_cfd/<experiment_batch>/xy_wrf_cfd_<YYYYMMDD_HHMM>_<height>m.png``
    """
    cfd_dir = cfd_dir.rstrip(os.sep)
    case = os.path.basename(cfd_dir)
    batch = os.path.basename(os.path.dirname(cfd_dir)) or "misc"
    m = re.match(r"(\d{8}_\d{4})", case)
    stamp = m.group(1) if m else case
    return os.path.join(
        _repo_root(), RESULTS_XY_DIR, batch,
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


def infer_paths(cfd_dir: str, height: int):
    wrf_time = parse_timestamp_from_cfd_dir(cfd_dir)
    nc_filename = WRF_NC_TEMPLATE.format(wrf_time=wrf_time)
    wrf_nc_path = os.path.join(wrf_root_for_time(wrf_time), nc_filename)
    csv_relpath = os.path.join("postProcessing", f"{height}m.csv")
    cfd_csv = os.path.join(cfd_dir, csv_relpath)
    return wrf_nc_path, cfd_csv, wrf_time


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

    print(f"  Extracted plane @ {target_height:g} m: shape={ws_interp.shape}, "
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
        chunks.append(chunk[req].astype(float))

    df = pd.concat(chunks, ignore_index=True)
    u0 = df['U:0'].values
    u1 = df['U:1'].values
    return dict(x=df['Coords:0'].values, y=df['Coords:1'].values,
                u0=u0, u1=u1, wind_speed=np.sqrt(u0**2 + u1**2))


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


def _quiver_key_speed(vmax: float) -> float:
    """Pick a readable reference arrow length from the colorbar ceiling."""
    if vmax <= 2:
        return 1.0
    if vmax <= 5:
        return 2.0
    return 4.0


def draw_wrf_panel(ax, data: dict, vmax=None,
                   label='(a) WRF (mesoscale boundary)',
                   show_vectors=True, quiver_key_speed=None):
    lon = data['lon']
    lat = data['lat']
    ws = data['wind_speed']
    u = data['u']
    v = data['v']

    if vmax is None:
        vmax = np.nanpercentile(ws, 98)
    key_u = quiver_key_speed if quiver_key_speed is not None else _quiver_key_speed(vmax)

    qm = ax.pcolormesh(lon, lat, ws,
                       vmin=0, vmax=vmax, cmap='viridis',
                       shading='auto', alpha=0.85, rasterized=True)

    if show_vectors:
        ny, nx = lon.shape
        skip_y = max(1, ny // 12)
        skip_x = max(1, nx // 12)

        qv = ax.quiver(lon[::skip_y, ::skip_x], lat[::skip_y, ::skip_x],
                       u[::skip_y, ::skip_x], v[::skip_y, ::skip_x],
                       color='black', alpha=0.95,
                       scale=QUIVER_SCALE, width=QUIVER_WIDTH,
                       headwidth=4, headlength=5, headaxislength=4,
                       minshaft=1.5, zorder=5)

        ax.quiverkey(qv, X=0.82, Y=1.035, U=key_u,
                     label=f'{key_u:g} m/s',
                     labelpos='E', coordinates='axes',
                     fontproperties={'family': 'serif', 'size': 10, 'weight': 'bold'})

    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    # Pad the shorter axis so the box is square in data units (avoids
    # adjustable='datalim' silently rewriting limits).
    dx, dy = lon_max - lon_min, lat_max - lat_min
    side = max(dx, dy)
    lon_c, lat_c = 0.5 * (lon_min + lon_max), 0.5 * (lat_min + lat_max)
    ax.set_xlim(lon_c - 0.5 * side, lon_c + 0.5 * side)
    ax.set_ylim(lat_c - 0.5 * side, lat_c + 0.5 * side)
    ax.margins(0)

    ax.set_xlabel('Longitude (°E)', fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontweight='bold')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.grid(True, alpha=0.25, ls='--', lw=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_aspect('equal', adjustable='datalim')
    _add_panel_label(ax, label)
    return qm


def draw_cfd_panel(ax, data: dict, vmax=None,
                   label='(b) CFD (OpenFOAM)',
                   show_vectors=True, quiver_key_speed=None):
    x = data['x']
    y = data['y']
    u0 = data['u0']
    u1 = data['u1']
    ws = data['wind_speed']

    if vmax is None:
        vmax = np.nanpercentile(ws, 98)
    key_u = quiver_key_speed if quiver_key_speed is not None else _quiver_key_speed(vmax)

    hb = ax.hexbin(x, y, C=ws,
                   gridsize=HEXBIN_GRID, cmap='viridis',
                   reduce_C_function=np.mean,
                   vmin=0, vmax=vmax,
                   alpha=0.85, edgecolors='none', rasterized=True)

    if show_vectors:
        x_g, y_g = np.mgrid[x.min():x.max():complex(0, QUIVER_GRID_CFD),
                            y.min():y.max():complex(0, QUIVER_GRID_CFD)]
        sub = max(1, len(x) // 100_000)
        gu0 = griddata((x[::sub], y[::sub]), u0[::sub], (x_g, y_g), method='linear')
        gu1 = griddata((x[::sub], y[::sub]), u1[::sub], (x_g, y_g), method='linear')

        qv = ax.quiver(x_g.ravel(), y_g.ravel(), gu0.ravel(), gu1.ravel(),
                       color='black', alpha=0.95,
                       scale=QUIVER_SCALE, width=QUIVER_WIDTH,
                       headwidth=4, headlength=5, headaxislength=4,
                       minshaft=1.5, zorder=5)

        ax.quiverkey(qv, X=0.82, Y=1.035, U=key_u,
                     label=f'{key_u:g} m/s',
                     labelpos='E', coordinates='axes',
                     fontproperties={'family': 'serif', 'size': 10, 'weight': 'bold'})

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    dx, dy = x_max - x_min, y_max - y_min
    side = max(dx, dy)
    xc, yc = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    ax.set_xlim(xc - 0.5 * side, xc + 0.5 * side)
    ax.set_ylim(yc - 0.5 * side, yc + 0.5 * side)
    ax.margins(0)

    ax.set_xlabel('X Coordinate (m)', fontweight='bold')
    ax.set_ylabel('Y Coordinate (m)', fontweight='bold')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.25, ls='--', lw=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    _add_panel_label(ax, label)
    return hb


def short_case_label(case: str) -> str:
    """Keep only YYYYMMDD_HHMM; drop tags like ``two_boundaries_as_outlet``."""
    m = re.match(r"(\d{8}_\d{4})", case)
    return m.group(1) if m else case


def compose_figure(wrf_data, cfd_data, case_label: str, output_path: str,
                   height: float):
    """Build a 2-panel figure: (a) WRF top, (b) CFD bottom."""
    _apply_global_style()

    cfd_p98 = float(np.nanpercentile(cfd_data['wind_speed'], 98))
    wrf_p98 = (
        float(np.nanpercentile(wrf_data['wind_speed'], 98))
        if wrf_data is not None else cfd_p98
    )
    shared_vmax = _nice_vmax(max(cfd_p98, wrf_p98))
    key_u = _quiver_key_speed(shared_vmax)

    # Narrow canvas; allocate identical *square* axes boxes (no horizontal stretch).
    fig_w, fig_h = 6.5, 10.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    panel_h = 0.40
    gap = 0.060
    panel_bottom = 0.055
    panel_w = panel_h * fig_h / fig_w
    panel_left = 0.17
    wrf_bottom = panel_bottom + panel_h + gap

    ax_cfd = fig.add_axes([panel_left, panel_bottom, panel_w, panel_h])
    ax_wrf = fig.add_axes([panel_left, wrf_bottom, panel_w, panel_h])

    cbar_left = panel_left + panel_w + 0.04
    cax = fig.add_axes([cbar_left, panel_bottom, 0.04, 2 * panel_h + gap])

    wrf_label = f'(a) WRF (mesoscale) @ {height:g} m'
    cfd_label = f'(b) CFD (OpenFOAM) @ {height:g} m'

    mappable = None
    if wrf_data is not None:
        mappable = draw_wrf_panel(
            ax_wrf, wrf_data, vmax=shared_vmax, label=wrf_label,
            quiver_key_speed=key_u,
        )
    else:
        ax_wrf.text(0.5, 0.5, 'WRF data unavailable\n(file not found)',
                    ha='center', va='center', transform=ax_wrf.transAxes,
                    fontsize=12, color='grey')
        _add_panel_label(ax_wrf, wrf_label)

    hb_cfd = draw_cfd_panel(
        ax_cfd, cfd_data, vmax=shared_vmax, label=cfd_label,
        quiver_key_speed=key_u,
    )
    if mappable is None:
        mappable = hb_cfd
    add_wind_speed_colorbar(fig, mappable, cax=cax)

    # Lock identical square boxes (limits already square → datalim won't resize)
    ax_wrf.set_position([panel_left, wrf_bottom, panel_w, panel_h])
    ax_cfd.set_position([panel_left, panel_bottom, panel_w, panel_h])
    cax.set_position([cbar_left, panel_bottom, 0.04, 2 * panel_h + gap])

    stamp = short_case_label(case_label)
    fig.suptitle(
        f'X-Y Horizontal Wind Field — WRF vs CFD\n{stamp} @ {height:g} m',
        fontsize=12, fontweight='bold', y=0.975,
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
    p.add_argument('--height', type=int, default=DEFAULT_HEIGHT,
                   help=f'Horizontal slice height in m (default: {DEFAULT_HEIGHT})')
    p.add_argument('--wrf-nc', default=None,
                   help='Override auto-detected WRF NetCDF file path')
    p.add_argument('--cfd-csv', default=None,
                   help='Override auto-detected CFD CSV path')
    p.add_argument('--output', default=None,
                   help='Output PNG path (default: results/wrf_openfoam/xy_wrf_cfd/'
                        '<experiment_batch>/xy_wrf_cfd_<YYYYMMDD_HHMM>_<height>m.png)')
    p.add_argument('--lat', type=float, default=TARGET_LAT)
    p.add_argument('--lon', type=float, default=TARGET_LON)
    p.add_argument('--lat-tol', type=float, default=LAT_TOL)
    p.add_argument('--lon-tol', type=float, default=LON_TOL)
    p.add_argument('--no-wrf', action='store_true',
                   help='Skip WRF panel even if the file is available')
    return p


def main():
    args = build_parser().parse_args()
    cfd_dir = args.cfd_dir.rstrip('/')

    wrf_nc_path, cfd_csv, wrf_time = infer_paths(cfd_dir, args.height)
    if args.wrf_nc:
        wrf_nc_path = args.wrf_nc
    if args.cfd_csv:
        cfd_csv = args.cfd_csv

    basename = os.path.basename(cfd_dir)
    output_path = args.output or default_output_path(cfd_dir, args.height)
    wrf_exp = "W_myExp05" if "W_myExp05" in wrf_nc_path else "W_myExp03"

    print("=" * 64)
    print("  WRF–CFD X-Y Comparison (2-panel)")
    print("=" * 64)
    print(f"  Height       : {args.height} m")
    print(f"  CFD CSV      : {cfd_csv}")
    print(f"  WRF source   : {wrf_exp}  (≥09-07 → Exp05, else Exp03)")
    print(f"  WRF nc file  : {wrf_nc_path}")
    print(f"  Output       : {output_path}")
    print("=" * 64)

    wrf_data = None
    if not args.no_wrf:
        if not os.path.exists(wrf_nc_path):
            warnings.warn(
                f"WRF file not found: {wrf_nc_path}\n"
                "WRF panel will show a placeholder. Use --wrf-nc to override.")
        else:
            print(f"\n[1/2] Loading WRF data ...  ({wrf_time})")
            wrf_data = extract_wrf_xy(
                wrf_nc_path, args.height,
                target_lat=args.lat, target_lon=args.lon,
                lat_tol=args.lat_tol, lon_tol=args.lon_tol,
            )
            print(f"      Wind speed range: "
                  f"[{np.nanmin(wrf_data['wind_speed']):.2f}, "
                  f"{np.nanmax(wrf_data['wind_speed']):.2f}] m/s")

    print("\n[2/2] Loading CFD CSV …")
    cfd_data = load_cfd_csv(cfd_csv)
    print(f"      {len(cfd_data['x']):,} points  |  "
          f"WS range [{cfd_data['wind_speed'].min():.2f}, "
          f"{cfd_data['wind_speed'].max():.2f}] m/s")

    print("\nRendering figure …")
    compose_figure(
        wrf_data, cfd_data,
        case_label=basename,
        output_path=output_path,
        height=args.height,
    )


if __name__ == "__main__":
    main()
