"""
WRF–CFD X-Z Vertical Wind Field Comparison (2-panel)
====================================================
Produces a publication-quality 2-panel figure:

    ┌────────────────────────────────────────┐
    │  (a) WRF (mesoscale boundary)          │
    ├────────────────────────────────────────┤
    │  (b) CFD (OpenFOAM)                    │
    └────────────────────────────────────────┘

Usage
-----
    python visualize_WRF_CFD_xz_two_panel.py  /path/to/CFD_run_directory

Example
-------
    python visualize_WRF_CFD_xz_two_panel.py \\
        steady_experiments_finer_ABL/20250903_1200_two_boundaries_as_outlet

Path inference
--------------
Given CFD path ``<root>/<YYYYMMDD_HHMM>_<tag>`` the script resolves:

* WRF nc  →  ``W_myExp03/auxhist2/tmp/auxhist2_d03_<YYYY-MM-DD_HH:MM:00>_tmp.nc``
* CFD CSV →  ``<cfd_dir>/postProcessing/y800m.csv``
* PNG out →  ``results/wrf_openfoam/xz_wrf_cfd/<experiment_batch>/comparison_xz_wrf_cfd_<case>.png``

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
WRF_ROOT        = os.path.join("W_myExp03", "auxhist2", "tmp")
WRF_NC_TEMPLATE = "auxhist2_d03_{wrf_time}_tmp.nc"
CSV_RELPATH     = os.path.join("postProcessing", "y800m.csv")

TARGET_LAT   = 23.1211944444
TARGET_LON   = 113.321102778
LAT_TOL      = 0.004
LON_TOL      = 0.0225000225
MAX_HEIGHT   = 2000
CFD_TOP      = 2000
QUIVER_GRID  = 20
QUIVER_SCALE = 40
HEXBIN_GRID  = 120

# Fixed colorbar ticks for stable PNG size (GIF / time-series export)
WIND_SPEED_COLORBAR_TICKS = np.array(
    [0., 2., 4., 6., 8., 10., 12., 14., 16.], dtype=float
)
WIND_SPEED_COLORBAR_VMAX = float(WIND_SPEED_COLORBAR_TICKS.max())
WIND_SPEED_COLORBAR_TICK_FORMAT = '%2.0f'

# Default figure output (repo convention: results/wrf_openfoam/…)
RESULTS_XZ_DIR = os.path.join("results", "wrf_openfoam", "xz_wrf_cfd")

# ---------------------------------------------------------------------------
# PATH HELPERS
# ---------------------------------------------------------------------------

def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_output_path(cfd_dir: str) -> str:
    """
    ``results/wrf_openfoam/xz_wrf_cfd/<experiment_batch>/comparison_xz_wrf_cfd_<case>.png``

  *experiment_batch* is the parent folder of the CFD case directory
  (e.g. ``steady_experiments_finer_ABL``).
    """
    cfd_dir = cfd_dir.rstrip(os.sep)
    case = os.path.basename(cfd_dir)
    batch = os.path.basename(os.path.dirname(cfd_dir)) or "misc"
    return os.path.join(
        _repo_root(), RESULTS_XZ_DIR, batch,
        f"comparison_xz_wrf_cfd_{case}.png",
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


def infer_paths(cfd_dir: str):
    wrf_time = parse_timestamp_from_cfd_dir(cfd_dir)
    nc_filename = WRF_NC_TEMPLATE.format(wrf_time=wrf_time)
    wrf_nc_path = os.path.join(WRF_ROOT, nc_filename)
    cfd_csv = os.path.join(cfd_dir, CSV_RELPATH)
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
    V_sn = V_all[:, sn_idx, :]
    V_sn1 = V_all[:, sn_idx + 1, :]

    W_all = get_val('W')
    W_sn = W_all[:, sn_idx, :]

    ds.close()

    H_sn = _destagger_np((PH_sn + PHB_sn) / 9.81, axis=0)
    U_dest = _destagger_np(U_sn, axis=1)
    V_dest = 0.5 * (V_sn + V_sn1)
    W_dest = _destagger_np(W_sn, axis=0)
    WS = np.sqrt(U_dest**2 + V_dest**2)

    H_xz = H_sn[:, we_slice]
    WS_xz = WS[:, we_slice]
    U_xz = U_dest[:, we_slice]
    W_xz = W_dest[:, we_slice]
    lon_xz = lon_1d[we_slice]

    nan_mask = H_xz > max_height
    WS_xz[nan_mask] = np.nan
    U_xz[nan_mask] = np.nan
    W_xz[nan_mask] = np.nan

    lon_2d = np.broadcast_to(lon_xz[np.newaxis, :], H_xz.shape).copy()

    print(f"  Extracted slice: shape={H_xz.shape}, "
          f"lon=[{np.nanmin(lon_xz):.5f}, {np.nanmax(lon_xz):.5f}] deg E, "
          f"H=[{np.nanmin(H_xz):.0f}, {np.nanmax(H_xz):.0f}] m, "
          f"WS=[{np.nanmin(WS_xz):.2f}, {np.nanmax(WS_xz):.2f}] m/s")

    return dict(lon=lon_2d, height=H_xz, wind_speed=WS_xz, U=U_xz, W=W_xz)


# ---------------------------------------------------------------------------
# CFD CSV
# ---------------------------------------------------------------------------

def load_cfd_csv(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CFD CSV not found: {csv_path}")

    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=100_000):
        req = ['Coords:0', 'Coords:2', 'U:0', 'U:2']
        if any(c not in chunk.columns for c in req):
            raise KeyError(f"Missing columns in {csv_path}. Expected: {req}")
        chunks.append(chunk[req].astype(float))

    df = pd.concat(chunks, ignore_index=True)
    u0 = df['U:0'].values
    u2 = df['U:2'].values
    return dict(x=df['Coords:0'].values, z=df['Coords:2'].values,
                u0=u0, u2=u2, wind_speed=np.sqrt(u0**2 + u2**2))


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


def add_wind_speed_colorbar(fig, mappable, ax, label='Wind Speed (m/s)',
                            ticks=None, tick_format=None):
    """Add a colorbar with fixed tick positions and label width."""
    ticks = np.asarray(
        ticks if ticks is not None else WIND_SPEED_COLORBAR_TICKS,
        dtype=float,
    )
    fmt = tick_format or WIND_SPEED_COLORBAR_TICK_FORMAT
    cb = fig.colorbar(mappable, ax=ax, label=label, pad=0.02, fraction=0.04)
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    cb.ax.tick_params(labelsize=10)
    return cb


def draw_wrf_panel(ax, data: dict, cfd_top=CFD_TOP,
                   vmax=None, max_height=MAX_HEIGHT,
                   label='(a) WRF (mesoscale boundary)'):
    lon = data['lon']
    height = data['height']
    ws = data['wind_speed']
    u = data['U']
    w = data['W']

    if vmax is None:
        vmax = np.nanpercentile(ws, 98)

    qm = ax.pcolormesh(lon, height, ws,
                       vmin=0, vmax=vmax, cmap='viridis',
                       shading='auto', alpha=0.92, rasterized=True)

    ny, nx = lon.shape
    skip_y = max(1, ny // 15)
    skip_x = max(1, nx // 15)

    qv = ax.quiver(lon[::skip_y, ::skip_x], height[::skip_y, ::skip_x],
                   u[::skip_y, ::skip_x], w[::skip_y, ::skip_x],
                   color='black', alpha=0.82,
                   scale=QUIVER_SCALE, width=0.003,
                   headwidth=3.5, headlength=5)

    ax.quiverkey(qv, X=0.87, Y=1.02, U=1, label='1 m/s',
                 labelpos='E', coordinates='axes',
                 fontproperties={'family': 'serif', 'size': 10, 'weight': 'bold'})

    ax.set_ylim(0, max_height)

    h_max = np.nanmax(height)
    if cfd_top <= h_max:
        ax.axhline(cfd_top, color='black', ls=':', lw=1.6, alpha=0.75)
        ax.text(np.nanmean(lon), cfd_top + 35,
                f'CFD top ({cfd_top} m)',
                color='black', fontsize=10, ha='center', va='bottom', alpha=0.85)

    ax.set_xlabel('Longitude (°E)', fontweight='bold', labelpad=10)
    ax.set_ylabel('Height (m)', fontweight='bold')
    # Show full longitude (e.g. 113.321), not offset/scientific notation
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.grid(True, alpha=0.25, ls='--', lw=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    _add_panel_label(ax, label)
    return qm


def draw_cfd_panel(ax, data: dict, vmax=None,
                   label='(b) CFD (OpenFOAM)'):
    x = data['x']
    z = data['z']
    u0 = data['u0']
    u2 = data['u2']
    ws = data['wind_speed']

    if vmax is None:
        vmax = np.nanpercentile(ws, 98)

    hb = ax.hexbin(x, z, C=ws,
                   gridsize=HEXBIN_GRID, cmap='viridis',
                   reduce_C_function=np.mean,
                   vmin=0, vmax=vmax,
                   alpha=0.88, edgecolors='none', rasterized=True)

    x_g, z_g = np.mgrid[x.min():x.max():complex(0, QUIVER_GRID),
                         z.min():z.max():complex(0, QUIVER_GRID)]
    sub = max(1, len(x) // 100_000)
    gu0 = griddata((x[::sub], z[::sub]), u0[::sub], (x_g, z_g), method='linear')
    gu2 = griddata((x[::sub], z[::sub]), u2[::sub], (x_g, z_g), method='linear')

    qv = ax.quiver(x_g.ravel(), z_g.ravel(), gu0.ravel(), gu2.ravel(),
                   color='black', alpha=0.82,
                   scale=QUIVER_SCALE, width=0.002,
                   headwidth=3.5, headlength=5,
                   headaxislength=4, minshaft=2)

    ax.quiverkey(qv, X=0.87, Y=0.975, U=2.5, label='2.5 m/s',
                 labelpos='E', coordinates='axes',
                 fontproperties={'family': 'serif', 'size': 10, 'weight': 'bold'})

    ax.set_xlabel('X Coordinate (m)', fontweight='bold')
    ax.set_ylabel('Height (m)', fontweight='bold')
    ax.set_aspect('auto')
    ax.grid(True, alpha=0.25, ls='--', lw=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    _add_panel_label(ax, label)
    return hb


def compose_figure(wrf_data, cfd_data, case_label: str, output_path: str,
                   max_height=MAX_HEIGHT):
    """Build a 2-panel figure: (a) WRF top, (b) CFD bottom."""
    _apply_global_style()

    wrf_vmax = WIND_SPEED_COLORBAR_VMAX if wrf_data else None
    cfd_vmax = WIND_SPEED_COLORBAR_VMAX

    fig = plt.figure(figsize=(12, 11))
    panel_h = 0.34
    gap = 0.10          # vertical space between panels (WRF xlabel + margin)
    ax_cfd = fig.add_axes([0.12, 0.10, 0.76, panel_h])
    ax_wrf = fig.add_axes([0.12, 0.10 + panel_h + gap, 0.76, panel_h])

    if wrf_data is not None:
        qm_wrf = draw_wrf_panel(ax_wrf, wrf_data, vmax=wrf_vmax,
                                max_height=max_height)
        add_wind_speed_colorbar(fig, qm_wrf, ax_wrf)
    else:
        ax_wrf.text(0.5, 0.5, 'WRF data unavailable\n(file not found)',
                    ha='center', va='center', transform=ax_wrf.transAxes,
                    fontsize=12, color='grey')
        _add_panel_label(ax_wrf, '(a) WRF (mesoscale boundary)')

    hb_cfd = draw_cfd_panel(ax_cfd, cfd_data, vmax=cfd_vmax)
    add_wind_speed_colorbar(fig, hb_cfd, ax_cfd)

    fig.suptitle(
        f'X-Z Vertical Wind Field — WRF vs CFD\n{case_label}',
        fontsize=14, fontweight='bold', y=0.97,
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.10)
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
    p.add_argument('--wrf-nc', default=None,
                   help='Override auto-detected WRF NetCDF file path')
    p.add_argument('--cfd-csv', default=None,
                   help='Override auto-detected CFD CSV path')
    p.add_argument('--output', default=None,
                   help='Output PNG path (default: results/wrf_openfoam/xz_wrf_cfd/'
                        '<experiment_batch>/comparison_xz_wrf_cfd_<case>.png)')
    p.add_argument('--lat', type=float, default=TARGET_LAT)
    p.add_argument('--lon', type=float, default=TARGET_LON)
    p.add_argument('--lat-tol', type=float, default=LAT_TOL)
    p.add_argument('--lon-tol', type=float, default=LON_TOL)
    p.add_argument('--max-height', type=float, default=MAX_HEIGHT)
    p.add_argument('--cfd-top', type=float, default=CFD_TOP)
    p.add_argument('--no-wrf', action='store_true',
                   help='Skip WRF panel even if the file is available')
    return p


def main():
    args = build_parser().parse_args()
    cfd_dir = args.cfd_dir.rstrip('/')

    wrf_nc_path, cfd_csv, wrf_time = infer_paths(cfd_dir)
    if args.wrf_nc:
        wrf_nc_path = args.wrf_nc
    if args.cfd_csv:
        cfd_csv = args.cfd_csv

    basename = os.path.basename(cfd_dir)
    output_path = args.output or default_output_path(cfd_dir)

    print("=" * 64)
    print("  WRF–CFD X-Z Comparison (2-panel)")
    print("=" * 64)
    print(f"  CFD CSV      : {cfd_csv}")
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
            wrf_data = extract_wrf_xz(
                wrf_nc_path,
                target_lat=args.lat, target_lon=args.lon,
                lat_tol=args.lat_tol, lon_tol=args.lon_tol,
                max_height=args.max_height,
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
        max_height=args.max_height,
    )


if __name__ == "__main__":
    main()
