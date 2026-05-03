"""
WRF–CFD X-Y Horizontal Wind Field Comparison Panel
==================================================
Produces a publication-quality 3-panel figure comparing horizontal wind fields
at a specific height (e.g., 100m) for journal submission:

    ┌─────────────────┬──────────────────────┐
    │  (a) WRF        │  (b) CFD control     │
    │  mesoscale      │  (OpenFOAM baseline) │
    │  (@ height H)   │  (@ height H)        │
    ├─────────────────┴──────────────────────┤
    │  (c) CFD sensitivity run               │
    │  (fvOpt_sensitivity_run @ height H)    │
    └────────────────────────────────────────┘

Usage
-----
    python visualize_WRF_CFD_xy_comparison_buoyFoam.py /path/to/CFD_control_run_directory --height 100

Example
-------
    python visualize_WRF_CFD_xy_comparison_buoyFoam.py \
        steady_experiments_finer_ABL/20250901_0000_WN_outlet --height 100

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

# ---------------------------------------------------------------------------
# CONSTANTS / DEFAULTS
# ---------------------------------------------------------------------------
WRF_ROOT           = "W_myExp03/auxhist2/tmp"
WRF_NC_TEMPLATE    = "auxhist2_d03_{wrf_time}_tmp.nc"
SENSITIVITY_SUFFIX = "_buoyFoam"
DEFAULT_HEIGHT     = 100

TARGET_LAT         = 23.1211944444
TARGET_LON         = 113.321102778
LAT_TOL            = 0.05  # Increased tolerance for XY view
LON_TOL            = 0.05

QUIVER_GRID        = 40
QUIVER_SCALE       = 20#80
HEXBIN_GRID        = 120

# ---------------------------------------------------------------------------
# HELPER: parse timestamp from CFD directory name
# ---------------------------------------------------------------------------

def parse_timestamp_from_cfd_dir(cfd_dir: str):
    basename = os.path.basename(cfd_dir.rstrip("/"))
    m = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})", basename)
    if not m:
        raise ValueError(
            f"Cannot parse YYYYMMDD_HHMM from directory name: '{basename}'"
        )
    yr, mo, dy, hh, mm = m.groups()
    wrf_time = f"{yr}-{mo}-{dy}_{hh}:{mm}:00"
    return wrf_time

def infer_paths(cfd_control_dir: str, height: int):
    wrf_time        = parse_timestamp_from_cfd_dir(cfd_control_dir)
    nc_filename     = WRF_NC_TEMPLATE.format(wrf_time=wrf_time)  # .replace(':', '%3A'))
    wrf_nc_path     = os.path.join(WRF_ROOT, nc_filename)
    
    csv_relpath     = os.path.join("postProcessing", f"{height}m.csv")
    cfd_ctrl_csv    = os.path.join(cfd_control_dir, csv_relpath)
    
    sens_dir        = cfd_control_dir.rstrip("/") + SENSITIVITY_SUFFIX
    cfd_sens_csv    = os.path.join(sens_dir, csv_relpath)
    
    return wrf_nc_path, cfd_ctrl_csv, cfd_sens_csv, wrf_time

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
    """
    Extract an X-Y horizontal plane from a WRF file at target_height (m).
    Uses vertical interpolation.
    """
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"WRF file not found: {nc_path}")

    ds = xr.open_dataset(nc_path)
    
    def get_val(name):
        v = ds[name]
        return v.values[0] if 'Time' in v.dims else v.values

    # 1. Load & Destagger
    ph  = get_val('PH')
    phb = get_val('PHB')
    z_stag = (ph + phb) / 9.81
    z = _destagger_np(z_stag, axis=0) # (bottom_top, south_north, west_east)
    
    u_stag = get_val('U')
    u = _destagger_np(u_stag, axis=2) # axis 2 is west_east_stag
    
    v_stag = get_val('V')
    v = _destagger_np(v_stag, axis=1) # axis 1 is south_north_stag
    
    lats = get_val('XLAT')
    lons = get_val('XLONG')
    
    # Ensure lats/lons are 2D
    if lats.ndim == 3: lats = lats[0]
    if lons.ndim == 3: lons = lons[0]
    
    # 2. Horizontal Cropping
    lat_mask = (lats >= target_lat - lat_tol) & (lats <= target_lat + lat_tol)
    lon_mask = (lons >= target_lon - lon_tol) & (lons <= target_lon + lon_tol)
    combined_mask = lat_mask & lon_mask
    
    # Find bounding box indices
    rows = np.any(combined_mask, axis=1)
    cols = np.any(combined_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        print("  [!] Target area outside WRF domain. Using full domain.")
        i_min, i_max = 0, lats.shape[0]
        j_min, j_max = 0, lats.shape[1]
    else:
        i_min, i_max = np.where(rows)[0][[0, -1]]
        j_min, j_max = np.where(cols)[0][[0, -1]]
        i_max += 1 # inclusive
        j_max += 1
        
    z_crop = z[:, i_min:i_max, j_min:j_max]
    u_crop = u[:, i_min:i_max, j_min:j_max]
    v_crop = v[:, i_min:i_max, j_min:j_max]
    lat_crop = lats[i_min:i_max, j_min:j_max]
    lon_crop = lons[i_min:i_max, j_min:j_max]
    
    # 3. Vertical Interpolation to target_height
    # Shape: (y, x)
    ny, nx = lat_crop.shape
    u_interp = np.zeros((ny, nx))
    v_interp = np.zeros((ny, nx))
    
    for i in range(ny):
        for j in range(nx):
            u_interp[i, j] = np.interp(target_height, z_crop[:, i, j], u_crop[:, i, j])
            v_interp[i, j] = np.interp(target_height, z_crop[:, i, j], v_crop[:, i, j])
            
    ws_interp = np.sqrt(u_interp**2 + v_interp**2)
    
    ds.close()
    
    return dict(lon=lon_crop, lat=lat_crop, 
                u=u_interp, v=v_interp, 
                wind_speed=ws_interp)

# ---------------------------------------------------------------------------
# CFD CSV DATA EXTRACTION
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
    ax.text(0.015, 0.965, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75))

def draw_wrf_panel(ax, data: dict, vmax=None, label='(a) WRF'):
    lon = data['lon']
    lat = data['lat']
    ws  = data['wind_speed']
    u   = data['u']
    v   = data['v']

    if vmax is None:
        vmax = np.nanpercentile(ws, 98)

    qm = ax.pcolormesh(lon, lat, ws,
                       vmin=0, vmax=vmax, cmap='viridis',
                       shading='auto', alpha=0.92, rasterized=True)

    # --- quiver logic for WRF ---
    # Pick a density that isn't too crowded (WRF grid is often ~1-3km/100m, but here it's likely cropped)
    ny, nx = lon.shape
    skip_y = max(1, ny // 15)
    skip_x = max(1, nx // 15)
    
    qv = ax.quiver(lon[::skip_y, ::skip_x], lat[::skip_y, ::skip_x], 
                   u[::skip_y, ::skip_x], v[::skip_y, ::skip_x],
                   color='black', alpha=0.82,
                   scale=QUIVER_SCALE, width=0.003,
                   headwidth=3.5, headlength=5)

    ax.quiverkey(qv, X=0.87, Y=1.02, U=1, label='1 m/s',
                 labelpos='E', coordinates='axes',
                 fontproperties={'family': 'serif', 'size': 10, 'weight': 'bold'})

    ax.set_xlabel('Longitude (°E)', fontweight='bold')
    ax.set_ylabel('Latitude (°N)',  fontweight='bold')
    ax.grid(True, alpha=0.25, ls='--', lw=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_aspect('equal', adjustable='box')
    _add_panel_label(ax, label)
    return qm

def draw_cfd_panel(ax, data: dict, vmax=None, label='(b) CFD'):
    x  = data['x'];   y  = data['y']
    u0 = data['u0'];  u1 = data['u1']
    ws = data['wind_speed']

    if vmax is None:
        vmax = np.nanpercentile(ws, 98)

    hb = ax.hexbin(x, y, C=ws,
                   gridsize=HEXBIN_GRID, cmap='viridis',
                   reduce_C_function=np.mean,
                   vmin=0, vmax=vmax,
                   alpha=0.88, edgecolors='none', rasterized=True)

    # --- uniform quiver grid ----------------------------------------------
    grid_res = QUIVER_GRID
    x_g, y_g = np.mgrid[x.min():x.max():complex(0, grid_res),
                         y.min():y.max():complex(0, grid_res)]
    
    sub = max(1, len(x) // 100_000)
    gu0 = griddata((x[::sub], y[::sub]), u0[::sub], (x_g, y_g), method='linear')
    gu1 = griddata((x[::sub], y[::sub]), u1[::sub], (x_g, y_g), method='linear')

    if 'Sensitivity' not in label:
        qv = ax.quiver(x_g.ravel(), y_g.ravel(), gu0.ravel(), gu1.ravel(),
                    color='black', alpha=0.82,
                    scale=QUIVER_SCALE, width=0.002,
                    headwidth=3.5, headlength=5,
                    headaxislength=4, minshaft=2)

        ax.quiverkey(qv, X=0.87, Y=1.02, U=1, label='1 m/s',
                    labelpos='E', coordinates='axes',
                    fontproperties={'family': 'serif', 'size': 10, 'weight': 'bold'})
    else:
        qv = ax.quiver(x_g.ravel(), y_g.ravel(), gu0.ravel(), gu1.ravel(),
                    color='black', alpha=0.82,
                    scale=80, width=0.002,
                    headwidth=3.5, headlength=5,
                    headaxislength=4, minshaft=2)

        ax.quiverkey(qv, X=0.87, Y=0.975, U=4, label='4 m/s',
                    labelpos='E', coordinates='axes',
                    fontproperties={'family': 'serif', 'size': 10, 'weight': 'bold'})

    ax.set_xlabel('X Coordinate (m)', fontweight='bold')
    ax.set_ylabel('Y Coordinate (m)', fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.25, ls='--', lw=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    _add_panel_label(ax, label)
    return hb

# ---------------------------------------------------------------------------
# MAIN FIGURE COMPOSER
# ---------------------------------------------------------------------------

def compose_figure(wrf_data, cfd_ctrl_data, cfd_sens_data,
                   case_label: str, output_path: str, height: float):
    _apply_global_style()

    # Independent vmax for each panel (as per user request)
    wrf_vmax = float(np.nanpercentile(wrf_data['wind_speed'], 98)) if wrf_data else None
    cfd_ctrl_vmax = float(np.nanpercentile(cfd_ctrl_data['wind_speed'], 98))
    cfd_sens_vmax = float(np.nanpercentile(cfd_sens_data['wind_speed'], 98))

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1, 1],
        width_ratios=[1, 1],
        hspace=0.25,
        wspace=0.25,
    )

    ax_wrf  = fig.add_subplot(gs[0, 0])
    ax_ctrl = fig.add_subplot(gs[0, 1])
    ax_sens = fig.add_subplot(gs[1, :])

    # (a) WRF
    if wrf_data:
        qm_wrf = draw_wrf_panel(ax_wrf, wrf_data, vmax=wrf_vmax,
                                label=f'(a) WRF (mesoscale) @ {height}m')
        cb_wrf = fig.colorbar(qm_wrf, ax=ax_wrf, label='Wind Speed (m/s)',
                              pad=0.03, fraction=0.046)
        cb_wrf.ax.tick_params(labelsize=10)
    else:
        ax_wrf.text(0.5, 0.5, 'WRF data unavailable', ha='center', va='center', transform=ax_wrf.transAxes)

    # (b) CFD Control
    hb_ctrl = draw_cfd_panel(ax_ctrl, cfd_ctrl_data, vmax=cfd_ctrl_vmax,
                             label=f'(b) CFD Control @ {height}m')
    cb_ctrl = fig.colorbar(hb_ctrl, ax=ax_ctrl, label='Wind Speed (m/s)',
                           pad=0.03, fraction=0.046)
    cb_ctrl.ax.tick_params(labelsize=10)

    # (c) CFD Sensitivity
    hb_sens = draw_cfd_panel(ax_sens, cfd_sens_data, vmax=cfd_sens_vmax,
                             label=f'(c) CFD Sensitivity\n(buoyFoam) @ {height}m')
    cb_sens = fig.colorbar(hb_sens, ax=ax_sens, label='Wind Speed (m/s)',
                           pad=0.02, fraction=0.023)
    cb_sens.ax.tick_params(labelsize=10)

    fig.suptitle(
        f'X-Y Horizontal Wind Field Comparison ({height}m) — {case_label}',
        fontsize=16, fontweight='bold', y=0.98
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDONE: Figure saved -> {output_path}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='WRF–CFD X-Y wind field comparison')
    p.add_argument('cfd_control_dir')
    p.add_argument('--height', type=int, default=DEFAULT_HEIGHT)
    p.add_argument('--wrf-nc', default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--no-wrf', action='store_true')
    args = p.parse_args()

    cfd_ctrl_dir = args.cfd_control_dir.rstrip('/')
    wrf_nc_path, cfd_ctrl_csv, cfd_sens_csv, wrf_time = infer_paths(cfd_ctrl_dir, args.height)
    
    if args.wrf_nc: wrf_nc_path = args.wrf_nc
    
    basename = os.path.basename(cfd_ctrl_dir)
    case_label = basename
    
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(os.path.dirname(cfd_ctrl_dir), f"comparison_xy_z{args.height}m_{case_label}.png")

    print(f"CFD Control : {cfd_ctrl_csv}")
    print(f"CFD Sens    : {cfd_sens_csv}")
    print(f"WRF NC      : {wrf_nc_path}")

    # Load WRF
    wrf_data = None
    if not args.no_wrf and os.path.exists(wrf_nc_path):
        print(f"Loading WRF at {args.height}m...")
        wrf_data = extract_wrf_xy(wrf_nc_path, args.height)
    elif not args.no_wrf:
        warnings.warn(f"WRF file not found: {wrf_nc_path}")

    # Load CFD
    print("Loading CFD Control...")
    cfd_ctrl_data = load_cfd_csv(cfd_ctrl_csv)
    print("Loading CFD Sensitivity...")
    cfd_sens_data = load_cfd_csv(cfd_sens_csv)

    # Compose
    compose_figure(wrf_data, cfd_ctrl_data, cfd_sens_data, case_label, output_path, args.height)

if __name__ == "__main__":
    main()
