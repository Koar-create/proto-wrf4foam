"""
WRF–CFD X-Z Vertical Wind Field Comparison Panel
==================================================
Produces a publication-quality 3-panel figure for journal submission
(B&E, JWEIA, BLM, GMD, …):

    ┌─────────────────┬──────────────────────┐
    │  (a) WRF        │  (b) CFD control     │
    │  mesoscale      │  (OpenFOAM baseline) │
    ├─────────────────┘                      │
    │  (c) CFD sensitivity run               │
    │  (fvOpt_sensitivity_run)               │
    └────────────────────────────────────────┘

Usage
-----
    python visualize_WRF_CFD_xz_comparison.py  /path/to/CFD_control_run_directory

Example
-------
    python visualize_WRF_CFD_xz_comparison.py \
        steady_experiments_finer_ABL/20250903_1200_two_boundaries_as_outlet

Path inference rules
--------------------
Given CFD control path  ``<root>/<YYYYMMDD_HHMM>_<tag>``  the script resolves:

* WRF nc file  →  ``W_myExp03/auxhist2/tmp/auxhist2_d03_<YYYY-MM-DD_HH:MM:00>_tmp.nc``
* CFD control CSV  →  ``<cfd_control_dir>/postProcessing/y800m.csv``
* CFD sensitivity  →  ``<root>/<YYYYMMDD_HHMM>_<tag>-fvOpt_sensitivity_run/postProcessing/y800m.csv``

Override any auto-detected path with the CLI flags documented below.
"""

import os
import sys
import re
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import griddata

# wrf-python is optional and NOT required – we use pure-numpy destagger
try:
    from wrf import destagger as _wrf_destagger  # noqa: F401 (kept for user convenience)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# CONSTANTS / DEFAULTS
# ---------------------------------------------------------------------------
WRF_ROOT          = "W_myExp03/auxhist2/tmp"
WRF_NC_TEMPLATE   = "auxhist2_d03_{wrf_time}_tmp.nc"   # e.g. 2025-09-03_12:00:00
SENSITIVITY_SUFFIX = "_buoyFoam"
CSV_RELPATH        = os.path.join("postProcessing", "y800m.csv")

TARGET_LAT   = 23.1211944444
TARGET_LON   = 113.321102778
LAT_TOL      = 0.004
LON_TOL      = 0.0225000225
MAX_HEIGHT   = 2000 # 3000   # m  – WRF extraction ceiling
CFD_TOP      = 2000          # m  – shown in WRF panel as reference line
QUIVER_GRID  = 20            # quiver arrow density (NxN)
QUIVER_SCALE = 40#80
HEXBIN_GRID  = 120

# ---------------------------------------------------------------------------
# HELPER: parse timestamp from CFD directory name
# ---------------------------------------------------------------------------

def parse_timestamp_from_cfd_dir(cfd_dir: str):
    """
    Extract YYYYMMDD_HHMM from a directory name such as
    ``steady_experiments_finer_ABL/20250903_1200_two_boundaries_as_outlet``
    and return the WRF-style timestamp string ``YYYY-MM-DD_HH:MM:00``.
    """
    basename = os.path.basename(cfd_dir.rstrip("/"))
    m = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})", basename)
    if not m:
        raise ValueError(
            f"Cannot parse YYYYMMDD_HHMM from directory name: '{basename}'\n"
            "Expected format: <root>/<YYYYMMDD_HHMM>_<tag>"
        )
    yr, mo, dy, hh, mm = m.groups()
    wrf_time = f"{yr}-{mo}-{dy}_{hh}:{mm}:00"
    return wrf_time


def infer_paths(cfd_control_dir: str):
    """Return (wrf_nc_path, cfd_control_csv, cfd_sens_csv) inferred from the control dir."""
    wrf_time        = parse_timestamp_from_cfd_dir(cfd_control_dir)
    nc_filename     = WRF_NC_TEMPLATE.format(wrf_time=wrf_time)  # .replace(':', '%3A'))
    wrf_nc_path     = os.path.join(WRF_ROOT, nc_filename)
    cfd_ctrl_csv    = os.path.join(cfd_control_dir, CSV_RELPATH)
    sens_dir        = cfd_control_dir.rstrip("/") + SENSITIVITY_SUFFIX
    cfd_sens_csv    = os.path.join(sens_dir, CSV_RELPATH)
    return wrf_nc_path, cfd_ctrl_csv, cfd_sens_csv, wrf_time


# ---------------------------------------------------------------------------
# WRF DATA EXTRACTION
# ---------------------------------------------------------------------------

def _destagger_np(arr: np.ndarray, axis: int) -> np.ndarray:
    """
    Arithmetic-mean destagger along *axis* (no wrf-python dependency).
    Reduces the size along *axis* by 1.
    """
    slc_lo = [slice(None)] * arr.ndim
    slc_hi = [slice(None)] * arr.ndim
    slc_lo[axis] = slice(None, -1)
    slc_hi[axis] = slice(1, None)
    return 0.5 * (arr[tuple(slc_lo)] + arr[tuple(slc_hi)])


def extract_wrf_xz(nc_path: str,
                   target_lat=TARGET_LAT, target_lon=TARGET_LON,
                   lat_tol=LAT_TOL, lon_tol=LON_TOL,
                   max_height=MAX_HEIGHT):
    """
    Extract an X-Z cross-section from a WRF auxhist2 NetCDF file.

    Strategy
    --------
    1. Identify the single ``south_north`` row whose zonal-mean latitude is
       closest to *target_lat* (nearest-neighbour; avoids the ragged-array
       problem that arises from applying a 2-D lat/lon mask to a 3-D field
       and then calling ``.where(drop=True)``).
    2. Keep all ``west_east`` columns within *lon_tol* of *target_lon*; if
       fewer than 2 columns qualify (coarse grid), fall back to the full
       west_east extent so the panel is never empty.
    3. Destagger PH/PHB (stagger axis 0), U (axis 2), V (axis 1) using
       simple arithmetic averaging – no wrf-python required.
    4. Mask levels whose height exceeds *max_height* with NaN so
       ``pcolormesh`` clips them cleanly.

    Returns
    -------
    dict with keys ``lon``, ``height``, ``wind_speed``, ``U``  – all
    2-D numpy arrays of shape ``(n_levels, n_west_east)``.
    """
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"WRF file not found: {nc_path}")

    print(f"  Opening: {nc_path}")
    ds = xr.open_dataset(nc_path)
    
    def get_val(name):
        v = ds[name]
        return v.values[0] if 'Time' in v.dims else v.values

    # ------------------------------------------------------------------
    # 1. Find nearest south_north index
    # ------------------------------------------------------------------
    # XLAT has dims (south_north, west_east); average over west_east for a
    # representative zonal-mean lat per row.
    lats = get_val('XLAT')
    if lats.ndim == 3: lats = lats[0] # handle Time dim if present but not detected by get_val
    lat_1d = np.mean(lats, axis=1)
    sn_idx = int(np.argmin(np.abs(lat_1d - target_lat)))
    actual_lat = float(lat_1d[sn_idx])
    print(f"  Nearest south_north row: index={sn_idx}, lat={actual_lat:.6f} deg N")

    # ------------------------------------------------------------------
    # 2. Find west_east columns within lon tolerance
    # ------------------------------------------------------------------
    lons = get_val('XLONG')
    if lons.ndim == 3: lons = lons[0]
    lon_1d = lons[sn_idx, :]
    we_mask = (target_lon - lon_tol <= lon_1d) & (lon_1d <= target_lon + lon_tol)
    we_idx = np.where(we_mask)[0]

    if len(we_idx) < 2:
        we_slice = slice(None)
    else:
        we_slice = slice(int(we_idx[0]), int(we_idx[-1]) + 1)

    # ------------------------------------------------------------------
    # 3. Load staggered arrays
    # ------------------------------------------------------------------
    PH_all  = get_val('PH')
    PHB_all = get_val('PHB')
    PH_sn   = PH_all[:, sn_idx, :]
    PHB_sn  = PHB_all[:, sn_idx, :]

    U_all   = get_val('U')
    U_sn    = U_all[:, sn_idx, :]

    V_all   = get_val('V')
    V_sn    = V_all[:, sn_idx, :]
    V_sn1   = V_all[:, sn_idx + 1, :]

    W_all   = get_val('W')
    W_sn    = W_all[:, sn_idx, :]

    ds.close()

    # ------------------------------------------------------------------
    # 4. Destagger
    # ------------------------------------------------------------------
    H_sn  = _destagger_np((PH_sn + PHB_sn) / 9.81, axis=0) # (bt, we)
    U_dest = _destagger_np(U_sn, axis=1)                   # (bt, we)
    V_dest = 0.5 * (V_sn + V_sn1)                          # (bt, we)
    W_dest = _destagger_np(W_sn, axis=0)                   # (bt, we)

    WS = np.sqrt(U_dest**2 + V_dest**2)

    # ------------------------------------------------------------------
    # 5. Apply lon and height selections
    # ------------------------------------------------------------------
    H_xz  = H_sn[:, we_slice]
    WS_xz = WS[:,  we_slice]
    U_xz  = U_dest[:, we_slice]
    W_xz  = W_dest[:, we_slice]
    lon_xz = lon_1d[we_slice]

    nan_mask = H_xz > max_height
    WS_xz[nan_mask] = np.nan
    U_xz[nan_mask]  = np.nan
    W_xz[nan_mask]  = np.nan

    lon_2d = np.broadcast_to(lon_xz[np.newaxis, :], H_xz.shape).copy()

    return dict(lon=lon_2d, height=H_xz, wind_speed=WS_xz, U=U_xz, W=W_xz)


# ---------------------------------------------------------------------------
# CFD CSV DATA EXTRACTION
# ---------------------------------------------------------------------------

def load_cfd_csv(csv_path: str):
    """
    Load OpenFOAM postProcessing CSV (y800m.csv) and return
    a dict with x, z, u0, u2, wind_speed arrays.
    """
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
# PANEL-DRAWING HELPERS
# ---------------------------------------------------------------------------

_STYLE_DONE = False

def _apply_global_style():
    global _STYLE_DONE
    if _STYLE_DONE:
        return
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family':       'serif',
        'font.size':          13,
        'axes.titlesize':     14,
        'axes.labelsize':     13,
        'xtick.labelsize':    11,
        'ytick.labelsize':    11,
        'axes.linewidth':      0.8,
        'figure.dpi':        150,
    })
    _STYLE_DONE = True


def _add_panel_label(ax, label, fontsize=15):
    """Add bold panel label (a), (b), (c) at top-left inside axes."""
    ax.text(0.015, 0.965, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75))


def draw_wrf_panel(ax, data: dict, cfd_top=CFD_TOP, target_lon=TARGET_LON,
                   vmax=None, max_height=MAX_HEIGHT, label='(a) WRF (mesoscale boundary)'):
    """
    Render the WRF pcolormesh panel onto *ax*.
    Returns the QuadMesh object for shared colorbar construction.
    """
    lon    = data['lon']
    height = data['height']
    ws     = data['wind_speed']
    u      = data['U']
    w      = data['W']

    if vmax is None:
        vmax = np.nanpercentile(ws, 98)

    qm = ax.pcolormesh(lon, height, ws,
                       vmin=0, vmax=vmax, cmap='viridis',
                       shading='auto', alpha=0.92, rasterized=True)

    # --- quiver logic for WRF ---
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

    # CFD domain top reference line
    h_max = np.nanmax(height)
    if cfd_top <= h_max:
        ax.axhline(cfd_top, color='black', ls=':', lw=1.6, alpha=0.75)
        ax.text(np.nanmean(lon), cfd_top + 35,
                f'CFD top ({cfd_top} m)',
                color='black', fontsize=10, ha='center', va='bottom', alpha=0.85)

    ax.set_xlabel('Longitude (°E)', fontweight='bold')
    ax.set_ylabel('Height (m)',     fontweight='bold')
    ax.grid(True, alpha=0.25, ls='--', lw=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    _add_panel_label(ax, label)
    return qm


def draw_cfd_panel(ax, data: dict, title_suffix='',
                   vmax=None, label='(b)'):
    """
    Render a CFD hexbin + quiver panel onto *ax*.
    Returns the PolyCollection for shared colorbar construction.
    """
    x  = data['x'];   z  = data['z']
    u0 = data['u0'];  u2 = data['u2']
    ws = data['wind_speed']

    if vmax is None:
        vmax = np.nanpercentile(ws, 98)

    # --- hexbin background ------------------------------------------------
    hb = ax.hexbin(x, z, C=ws,
                   gridsize=HEXBIN_GRID, cmap='viridis',
                   reduce_C_function=np.mean,
                   vmin=0, vmax=vmax,
                   alpha=0.88, edgecolors='none', rasterized=True)

    # --- uniform quiver grid ----------------------------------------------
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
    ax.set_ylabel('Height (m)',       fontweight='bold')
    ax.set_aspect('auto')  # 必须改为 auto，否则 CFD 面板会根据坐标范围被强制锁死比例
    ax.grid(True, alpha=0.25, ls='--', lw=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    _add_panel_label(ax, label)
    return hb


# ---------------------------------------------------------------------------
# MAIN FIGURE COMPOSER
# ---------------------------------------------------------------------------

def compose_figure(wrf_data, cfd_ctrl_data, cfd_sens_data,
                   case_label: str, output_path: str, max_height=MAX_HEIGHT):
    """
    Build a 3-panel comparison figure:
        (a) top-left   – WRF mesoscale
        (b) top-right  – CFD control run
        (c) bottom     – CFD sensitivity run (full width)
    """
    _apply_global_style()

    # Determine a shared vmax across all three datasets for WRF and CFD panels
    # (WRF and CFD use different colormaps domains so we keep them independent
    #  for clarity; both use viridis but the physical domains differ.)
    wrf_vmax  = float(np.nanpercentile(wrf_data['wind_speed'], 98)) if wrf_data else None
    cfd_vmax  = max(
        float(np.nanpercentile(cfd_ctrl_data['wind_speed'], 98)),
        float(np.nanpercentile(cfd_sens_data['wind_speed'], 98))
    )

    # ---- Layout ----------------------------------------------------------
    fig = plt.figure(figsize=(18, 11))

    # 使用绝对坐标 [left, bottom, width, height] 强制三个子图物理尺寸完全一致且绝对不重叠
    ax_wrf  = fig.add_axes([0.10, 0.55, 0.35, 0.35])  # 顶部左侧
    ax_ctrl = fig.add_axes([0.55, 0.55, 0.35, 0.35])  # 顶部右侧
    ax_sens = fig.add_axes([0.55, 0.10, 0.35, 0.35]) # 底部居中 (0.1+0.55)/2 = 0.325

    # ---- (a) WRF ---------------------------------------------------------
    if wrf_data is not None:
        qm_wrf = draw_wrf_panel(ax_wrf, wrf_data, vmax=wrf_vmax,
                                max_height=max_height,
                                label='(a) WRF (mesoscale boundary)')
        cb_wrf = fig.colorbar(qm_wrf, ax=ax_wrf, label='Wind Speed (m/s)',
                              pad=0.02, fraction=0.04) # 统一 Colorbar 占用空间
        cb_wrf.ax.tick_params(labelsize=10)
    else:
        ax_wrf.text(0.5, 0.5, 'WRF data unavailable\n(file not found)',
                    ha='center', va='center', transform=ax_wrf.transAxes,
                    fontsize=12, color='grey')
        ax_wrf.set_title('(a) WRF (mesoscale)')
        _add_panel_label(ax_wrf, '(a) WRF (mesoscale)')

    # ---- (b) CFD control -------------------------------------------------
    hb_ctrl = draw_cfd_panel(ax_ctrl, cfd_ctrl_data, vmax=cfd_vmax,
                             label='(b) CFD – control run')
    cb_ctrl = fig.colorbar(hb_ctrl, ax=ax_ctrl, label='Wind Speed (m/s)',
                           pad=0.02, fraction=0.04) # 统一 Colorbar 占用空间
    cb_ctrl.ax.tick_params(labelsize=10)

    # ---- (c) CFD sensitivity ---------------------------------------------
    hb_sens = draw_cfd_panel(ax_sens, cfd_sens_data, vmax=cfd_vmax,
                             label='(c) CFD Sensitivity (buoyFoam)')
    cb_sens = fig.colorbar(hb_sens, ax=ax_sens, label='Wind Speed (m/s)',
                           pad=0.02, fraction=0.04) # 统一 Colorbar 占用空间
    cb_sens.ax.tick_params(labelsize=10)

    # ---- Super-title -----------------------------------------------------
    fig.suptitle(
        f'X-Z Vertical Wind Field Comparison — {case_label}',
        fontsize=15, fontweight='bold', y=0.995
    )

    # ---- Save ------------------------------------------------------------
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDONE: Figure saved -> {output_path}  (300 DPI)\n")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description='WRF–CFD X-Z wind field 3-panel comparison figure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('cfd_control_dir',
                   help='Path to the CFD control run directory '
                        '(e.g. steady_experiments_finer_ABL/20250903_1200_two_boundaries_as_outlet)')
    p.add_argument('--wrf-nc',   default=None,
                   help='Override auto-detected WRF NetCDF file path')
    p.add_argument('--sens-dir', default=None,
                   help='Override auto-detected CFD sensitivity run directory')
    p.add_argument('--output',   default=None,
                   help='Output PNG path (default: auto-generated beside the control CSV)')
    p.add_argument('--lat',      type=float, default=TARGET_LAT,
                   help=f'Target latitude for WRF slice (default: {TARGET_LAT})')
    p.add_argument('--lon',      type=float, default=TARGET_LON,
                   help=f'Target longitude for WRF slice (default: {TARGET_LON})')
    p.add_argument('--lat-tol',  type=float, default=LAT_TOL,
                   help=f'Latitude tolerance (default: {LAT_TOL})')
    p.add_argument('--lon-tol',  type=float, default=LON_TOL,
                   help=f'Longitude tolerance (default: {LON_TOL})')
    p.add_argument('--max-height', type=float, default=MAX_HEIGHT,
                   help=f'WRF extraction ceiling in m (default: {MAX_HEIGHT})')
    p.add_argument('--cfd-top',  type=float, default=CFD_TOP,
                   help=f'CFD domain top reference height in m (default: {CFD_TOP})')
    p.add_argument('--no-wrf',   action='store_true',
                   help='Skip WRF panel even if the file is available')
    return p


def main():
    args = build_parser().parse_args()

    cfd_ctrl_dir = args.cfd_control_dir.rstrip('/')

    # --- infer paths ------------------------------------------------------
    wrf_nc_path, cfd_ctrl_csv, cfd_sens_csv, wrf_time = infer_paths(cfd_ctrl_dir)

    if args.wrf_nc:
        wrf_nc_path = args.wrf_nc
    if args.sens_dir:
        cfd_sens_csv = os.path.join(args.sens_dir.rstrip('/'), CSV_RELPATH)

    # --- case label (used in title and default filename) ------------------
    basename    = os.path.basename(cfd_ctrl_dir)
    case_label  = basename                          # e.g. 20250903_1200_two_boundaries_as_outlet

    # --- output path ------------------------------------------------------
    if args.output:
        output_path = args.output
    else:
        ctrl_dir_parent = os.path.dirname(cfd_ctrl_dir)
        output_path = os.path.join(
            ctrl_dir_parent,
            f"comparison_xz_{case_label}.png"
        )

    # --- echo resolved paths ---------------------------------------------
    print("=" * 64)
    print("  WRF–CFD X-Z Comparison Panel")
    print("=" * 64)
    print(f"  CFD control  : {cfd_ctrl_csv}")
    print(f"  CFD sensitiv : {cfd_sens_csv}")
    print(f"  WRF nc file  : {wrf_nc_path}")
    print(f"  Output       : {output_path}")
    print("=" * 64)

    # --- load WRF ---------------------------------------------------------
    wrf_data = None
    if not args.no_wrf:
        if not os.path.exists(wrf_nc_path):
            warnings.warn(f"WRF file not found: {wrf_nc_path}\n"
                          "WRF panel will show a placeholder. "
                          "Use --wrf-nc to provide the correct path.")
        else:
            print(f"\n[1/3] Loading WRF data ...  ({wrf_time})")
            wrf_data = extract_wrf_xz(
                wrf_nc_path,
                target_lat=args.lat, target_lon=args.lon,
                lat_tol=args.lat_tol, lon_tol=args.lon_tol,
                max_height=args.max_height,
            )
            print(f"      Wind speed range: "
                  f"[{np.nanmin(wrf_data['wind_speed']):.2f}, "
                  f"{np.nanmax(wrf_data['wind_speed']):.2f}] m/s")

    # --- load CFD control -------------------------------------------------
    print(f"\n[2/3] Loading CFD control CSV …")
    cfd_ctrl_data = load_cfd_csv(cfd_ctrl_csv)
    print(f"      {len(cfd_ctrl_data['x']):,} points  |  "
          f"WS range [{cfd_ctrl_data['wind_speed'].min():.2f}, "
          f"{cfd_ctrl_data['wind_speed'].max():.2f}] m/s")

    # --- load CFD sensitivity ---------------------------------------------
    print(f"\n[3/3] Loading CFD sensitivity CSV …")
    cfd_sens_data = load_cfd_csv(cfd_sens_csv)
    print(f"      {len(cfd_sens_data['x']):,} points  |  "
          f"WS range [{cfd_sens_data['wind_speed'].min():.2f}, "
          f"{cfd_sens_data['wind_speed'].max():.2f}] m/s")

    # --- compose figure ---------------------------------------------------
    print(f"\nRendering figure …")
    compose_figure(
        wrf_data, cfd_ctrl_data, cfd_sens_data,
        case_label=case_label,
        output_path=output_path,
        max_height=args.max_height,
    )


if __name__ == "__main__":
    main()
