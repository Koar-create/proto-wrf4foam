"""
WRF–CFD X-Z comparison at a specific OpenFOAM time step (2-panel).

Reads a ParaView-exported y-slice CSV whose filename encodes the time
(e.g. ``y-4995m_t10.csv``). WRF NetCDF is inferred from the parent CFD case
directory, same as ``visualize_WRF_CFD_xz_two_panel.py``.

Usage
-----
    python visualize_WRF_CFD_xz_at_time.py /path/to/postProcessing/y-4995m_t10.csv

Example
-------
    python visualize_WRF_CFD_xz_at_time.py \\
        steady_experiments_finer_ABL/20250903_0000_two_boundaries_as_outlet-init_1000_run/postProcessing/y-4995m_t10.csv
"""

import os
import re
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

import visualize_WRF_CFD_xz_two_panel as v2p


def parse_time_from_csv(csv_path: str):
    """Parse time from ``..._t<time>.csv`` (e.g. ``y-4995m_t10.csv`` → 10)."""
    base = os.path.basename(csv_path)
    m = re.search(r"_t([\d.]+)\.csv$", base, re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Cannot parse time from CSV filename: '{base}'\n"
            "Expected pattern: ..._t<time>.csv  (e.g. y-4995m_t10.csv)"
        )
    t = float(m.group(1))
    time_str = str(int(t)) if t == int(t) else m.group(1)
    return t, time_str


def cfd_dir_from_csv(csv_path: str) -> str:
    """``.../<case>/postProcessing/foo.csv`` → ``.../<case>``."""
    csv_path = os.path.abspath(csv_path)
    parent = os.path.dirname(csv_path)
    if os.path.basename(parent) == "postProcessing":
        return os.path.dirname(parent)
    return parent


def default_output_path(cfd_dir: str, time_str: str) -> str:
    cfd_dir = cfd_dir.rstrip(os.sep)
    case = os.path.basename(cfd_dir)
    batch = os.path.basename(os.path.dirname(cfd_dir)) or "misc"
    return os.path.join(
        v2p._repo_root(),
        v2p.RESULTS_XZ_DIR,
        batch,
        f"comparison_xz_wrf_cfd_{case}_t{time_str}.png",
    )


def format_time_label(time_value) -> str:
    if isinstance(time_value, float) and time_value == int(time_value):
        return f"t={int(time_value)}"
    return f"t={time_value}"


def compose_figure(wrf_data, cfd_data, case_label: str, output_path: str,
                   time_value, max_height=v2p.MAX_HEIGHT):
    """2-panel figure with simulation time annotated."""
    v2p._apply_global_style()

    wrf_vmax = v2p.WIND_SPEED_COLORBAR_VMAX if wrf_data else None
    cfd_vmax = v2p.WIND_SPEED_COLORBAR_VMAX

    fig = plt.figure(figsize=(12, 11))
    panel_h = 0.34
    gap = 0.10
    ax_cfd = fig.add_axes([0.12, 0.10, 0.76, panel_h])
    ax_wrf = fig.add_axes([0.12, 0.10 + panel_h + gap, 0.76, panel_h])

    if wrf_data is not None:
        qm_wrf = v2p.draw_wrf_panel(ax_wrf, wrf_data, vmax=wrf_vmax,
                                    max_height=max_height)
        v2p.add_wind_speed_colorbar(fig, qm_wrf, ax_wrf)
    else:
        ax_wrf.text(0.5, 0.5, "WRF data unavailable\n(file not found)",
                    ha="center", va="center", transform=ax_wrf.transAxes,
                    fontsize=12, color="grey")
        v2p._add_panel_label(ax_wrf, "(a) WRF (mesoscale boundary)")

    hb_cfd = v2p.draw_cfd_panel(ax_cfd, cfd_data, vmax=cfd_vmax)
    v2p.add_wind_speed_colorbar(fig, hb_cfd, ax_cfd)

    time_label = format_time_label(time_value)
    fig.text(
        0.015, 0.99, time_label,
        transform=fig.transFigure,
        fontsize=15, fontweight="bold", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
    )

    fig.suptitle(
        f"X-Z Vertical Wind Field — WRF vs CFD\n{case_label}",
        fontsize=14, fontweight="bold", y=0.97,
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    print(f"\nDONE: Figure saved -> {output_path}  (300 DPI)\n")
    plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser(
        description="WRF–CFD X-Z 2-panel figure at one OpenFOAM time (time from CSV filename)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "cfd_csv",
        help="Path to ParaView CSV (e.g. .../postProcessing/y-4995m_t10.csv)",
    )
    p.add_argument("--output", default=None,
                   help="Output PNG (default: results/.../comparison_xz_wrf_cfd_<case>_t<time>.png)")
    p.add_argument("--wrf-nc", default=None, help="Override WRF NetCDF path")
    p.add_argument("--no-wrf", action="store_true",
                   help="Skip WRF panel even if the file is available")
    p.add_argument("--lat", type=float, default=v2p.TARGET_LAT)
    p.add_argument("--lon", type=float, default=v2p.TARGET_LON)
    p.add_argument("--lat-tol", type=float, default=v2p.LAT_TOL)
    p.add_argument("--lon-tol", type=float, default=v2p.LON_TOL)
    p.add_argument("--max-height", type=float, default=v2p.MAX_HEIGHT)
    return p


def main():
    args = build_parser().parse_args()
    cfd_csv = os.path.abspath(args.cfd_csv)
    if not os.path.isfile(cfd_csv):
        raise FileNotFoundError(f"CFD CSV not found: {cfd_csv}")

    time_value, time_str = parse_time_from_csv(cfd_csv)
    cfd_dir = cfd_dir_from_csv(cfd_csv)
    case_label = os.path.basename(cfd_dir)

    wrf_nc_path, _, wrf_time = v2p.infer_paths(cfd_dir)
    if args.wrf_nc:
        wrf_nc_path = args.wrf_nc

    output_path = args.output or default_output_path(cfd_dir, time_str)

    print("=" * 64)
    print("  WRF–CFD X-Z Comparison at time step")
    print("=" * 64)
    print(f"  CFD CSV      : {cfd_csv}")
    print(f"  Time (parsed): {format_time_label(time_value)}")
    print(f"  CFD case     : {cfd_dir}")
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
            wrf_data = v2p.extract_wrf_xz(
                wrf_nc_path,
                target_lat=args.lat, target_lon=args.lon,
                lat_tol=args.lat_tol, lon_tol=args.lon_tol,
                max_height=args.max_height,
            )

    print("\n[2/2] Loading CFD CSV …")
    cfd_data = v2p.load_cfd_csv(cfd_csv)
    print(f"      {len(cfd_data['x']):,} points  |  "
          f"WS range [{cfd_data['wind_speed'].min():.2f}, "
          f"{cfd_data['wind_speed'].max():.2f}] m/s")

    print("\nRendering figure …")
    compose_figure(
        wrf_data, cfd_data,
        case_label=case_label,
        output_path=output_path,
        time_value=time_value,
        max_height=args.max_height,
    )


if __name__ == "__main__":
    main()
