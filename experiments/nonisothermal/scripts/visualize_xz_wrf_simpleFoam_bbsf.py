#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3-panel X-Z comparison for a BBSF case tag:

    ┌──────────┬───────────────────┬────────────────┐
    │ (a) WRF  │ (b) W-OF          │ (c) W-OF  ▐ CB │
    │          │     (simpleFoam)  │     (bbsf)     │
    └──────────┴───────────────────┴────────────────┘

Colour fill is horizontal wind speed ``sqrt(U^2 + V^2)``. Drawing, WRF
extraction, buildings, and colorbar come from
``analysis/260409/ws-xz-plane/visualize_WRF_CFD_xz_two_panel.py``.

Usage
-----
    python experiments/nonisothermal/scripts/visualize_xz_wrf_simpleFoam_bbsf.py 20250901_1000
    python experiments/nonisothermal/scripts/visualize_xz_wrf_simpleFoam_bbsf.py 20250908_0000
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent / "util"))
import case_paths as cp  # noqa: E402

_REPO_ROOT = str(cp.REPO_ROOT)
_XZ_DIR = os.path.join(_REPO_ROOT, "analysis", "260409", "ws-xz-plane")
if _XZ_DIR not in sys.path:
    sys.path.insert(0, _XZ_DIR)

import visualize_WRF_CFD_xz_two_panel as xz  # noqa: E402


def _hide_yticklabels(ax):
    ax.tick_params(axis="y", labelleft=False)
    ax.set_ylabel("")


def compose_three_panel(wrf_data, simple_data, bbsf_data, case_label, output_path,
                        y_slice, shp_path, max_height=xz.MAX_HEIGHT,
                        cfd_top=xz.CFD_TOP):
    xz._apply_global_style()

    x0s, x1s, z0s, z1s = xz.cfd_xz_limits(simple_data)
    x0b, x1b, z0b, z1b = xz.cfd_xz_limits(bbsf_data)
    xz_lim = (
        min(x0s, x0b), max(x1s, x1b),
        min(z0s, z0b), max(z1s, z1b),
    )
    x0, x1, z0, z1 = xz_lim

    simple_p98 = float(np.nanpercentile(simple_data["wind_speed"], 98))
    bbsf_p98 = float(np.nanpercentile(bbsf_data["wind_speed"], 98))
    wrf_p98 = max(simple_p98, bbsf_p98)
    if wrf_data is not None:
        xz.wrf_lon_to_local_x(wrf_data, y_slice)
        in_view = (wrf_data["height"] >= z0) & (wrf_data["height"] <= z1)
        wrf_ws_view = wrf_data["wind_speed"][in_view]
        wrf_p98 = float(np.nanpercentile(
            wrf_ws_view if np.any(np.isfinite(wrf_ws_view))
            else wrf_data["wind_speed"],
            98,
        ))
        x_wrf = wrf_data["x"][0, :]
        print(
            f"  WRF native X : [{float(np.nanmin(x_wrf)):.0f}, "
            f"{float(np.nanmax(x_wrf)):.0f}] m  "
            f"({wrf_data['x'].shape[1]} columns, no regrid)"
        )

    shared_vmax = min(
        xz._nice_vmax(max(simple_p98, bbsf_p98, wrf_p98)),
        xz.COLORBAR_VMAX_CAP,
    )
    print(
        f"  Shared color  : horizontal WS  |  "
        f"simple p98={simple_p98:.2f}  bbsf p98={bbsf_p98:.2f}  "
        f"WRF p98={wrf_p98:.2f}  →  vmax={shared_vmax:g} m/s "
        f"(cap={xz.COLORBAR_VMAX_CAP:g})"
    )
    print(
        f"  Shared domain : X [{x0:.0f},{x1:.0f}] m @ Y={y_slice:g} m;  "
        f"Z [{z0:.0f},{z1:.0f}] m"
    )

    building_polys_x = []
    if os.path.exists(shp_path):
        building_polys_x = xz.load_building_xz_rects_x(
            shp_path, y_slice=y_slice, x_lim=(x0, x1),
        )
    else:
        warnings.warn(f"Building shapefile not found: {shp_path}")

    fig = plt.figure(figsize=(19.0, 5.2))
    panel_h = 0.70
    gap = 0.028
    panel_bottom = 0.16
    panel_w = 0.265
    panel_left = 0.055
    cbar_w = 0.014
    lefts = [panel_left + i * (panel_w + gap) for i in range(3)]
    cbar_left = lefts[2] + panel_w + 0.015

    ax_wrf = fig.add_axes([lefts[0], panel_bottom, panel_w, panel_h])
    ax_simple = fig.add_axes([lefts[1], panel_bottom, panel_w, panel_h])
    ax_bbsf = fig.add_axes([lefts[2], panel_bottom, panel_w, panel_h])
    cax = fig.add_axes([cbar_left, panel_bottom, cbar_w, panel_h])

    labels = (
        "(a) WRF",
        "(b) W-OF (simpleFoam)",
        "(c) W-OF (bbsf)",
    )
    mappable = None
    if wrf_data is not None:
        mappable = xz.draw_wrf_panel(
            ax_wrf, wrf_data, vmax=shared_vmax,
            max_height=max_height, cfd_top=cfd_top,
            label=labels[0], building_polys=building_polys_x,
            xz_lim=xz_lim,
        )
    else:
        ax_wrf.text(
            0.5, 0.5, "WRF data unavailable\n(file not found)",
            ha="center", va="center", transform=ax_wrf.transAxes,
            fontsize=12, color="grey",
        )
        xz._add_panel_label(ax_wrf, labels[0])
        ax_wrf.set_xlim(x0, x1)
        ax_wrf.set_ylim(z0, z1)
        ax_wrf.set_xlabel("X (m)", fontweight="bold")
        ax_wrf.set_ylabel("Height (m)", fontweight="bold")

    hb_simple = xz.draw_cfd_panel(
        ax_simple, simple_data, vmax=shared_vmax, label=labels[1],
        xz_lim=xz_lim, building_polys=building_polys_x,
    )
    hb_bbsf = xz.draw_cfd_panel(
        ax_bbsf, bbsf_data, vmax=shared_vmax, label=labels[2],
        xz_lim=xz_lim, building_polys=building_polys_x,
    )
    _hide_yticklabels(ax_simple)
    _hide_yticklabels(ax_bbsf)

    if mappable is None:
        mappable = hb_simple
    mappable.set_clim(0, shared_vmax)
    hb_simple.set_clim(0, shared_vmax)
    hb_bbsf.set_clim(0, shared_vmax)
    xz.add_wind_speed_colorbar(fig, mappable, cax=cax, label="Wind Speed (m/s)")

    for ax, left in zip((ax_wrf, ax_simple, ax_bbsf), lefts):
        ax.set_position([left, panel_bottom, panel_w, panel_h])
    cax.set_position([cbar_left, panel_bottom, cbar_w, panel_h])

    stamp = xz.title_timestamp(case_label)
    fig.suptitle(
        f"X-Z Vertical Wind Field at Y = {y_slice:g} m — {stamp}",
        fontsize=13, fontweight="bold", y=0.975,
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"\nDONE: Figure saved -> {output_path}  (300 DPI)\n")
    plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser(
        description="WRF / simpleFoam / bbsf X-Z 3-panel comparison",
    )
    p.add_argument(
        "tag",
        help="time tag: 20250901_1000 or 20250908_0000 (optional _bbsf suffix)",
    )
    p.add_argument("--simple-dir", default=None,
                   help="simpleFoam case directory (default from tag)")
    p.add_argument("--bbsf-dir", default=None,
                   help="buoyantBoussinesqSimpleFoam case directory (default from tag)")
    p.add_argument("--y-slice", type=float, default=xz.DEFAULT_Y_SLICE)
    p.add_argument("--wrf-nc", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--shp", default=None)
    p.add_argument("--max-height", type=float, default=xz.MAX_HEIGHT)
    p.add_argument("--cfd-top", type=float, default=xz.CFD_TOP)
    p.add_argument("--no-wrf", action="store_true")
    return p


def main():
    os.chdir(_REPO_ROOT)
    args = build_parser().parse_args()
    tag = cp.normalize_tag(args.tag)
    simple_dir = os.path.abspath(args.simple_dir or str(cp.simplefoam_dir(tag))).rstrip("/")
    bbsf_dir = os.path.abspath(args.bbsf_dir or str(cp.case_dir(tag))).rstrip("/")
    if not args.output:
        args.output = str(cp.figs_dir() / f"xz_wrf_simpleFoam_bbsf_{tag}_y800.png")
    y_slice = args.y_slice

    wrf_nc_path, _, wrf_time = xz.infer_paths(bbsf_dir, y_slice)
    if args.wrf_nc:
        wrf_nc_path = args.wrf_nc
    wrf_nc_path = xz.resolve_existing_wrf_nc_path(wrf_nc_path)

    simple_csv = xz.resolve_cfd_csv(simple_dir, y_slice)
    bbsf_csv = xz.resolve_cfd_csv(bbsf_dir, y_slice)
    y_slice = xz.parse_y_slice_from_csv(bbsf_csv, fallback=y_slice)

    output_path = os.path.abspath(args.output)
    shp_path = args.shp or os.path.join(_REPO_ROOT, xz.DEFAULT_SHP)

    print("=" * 64)
    print("  WRF / simpleFoam / bbsf X-Z Comparison (3-panel)")
    print("=" * 64)
    print(f"  Y-slice      : {y_slice:g} m")
    print(f"  simpleFoam   : {simple_csv}")
    print(f"  bbsf         : {bbsf_csv}")
    print(f"  WRF nc file  : {wrf_nc_path}")
    print(f"  Building SHP : {shp_path}")
    print(f"  Output       : {output_path}")
    print("=" * 64)

    print("\n[1/3] Loading simpleFoam CSV …")
    simple_data = xz.load_cfd_csv(simple_csv)
    print(f"      {len(simple_data['x']):,} points  |  "
          f"WS range [{simple_data['wind_speed'].min():.2f}, "
          f"{simple_data['wind_speed'].max():.2f}] m/s")

    print("\n[2/3] Loading bbsf CSV …")
    bbsf_data = xz.load_cfd_csv(bbsf_csv)
    print(f"      {len(bbsf_data['x']):,} points  |  "
          f"WS range [{bbsf_data['wind_speed'].min():.2f}, "
          f"{bbsf_data['wind_speed'].max():.2f}] m/s")

    x0, x1, _, _ = xz.cfd_xz_limits(bbsf_data)
    lon0, lon1 = xz.lon_limits_from_x_range(x0, x1, y_slice)
    target_lat = xz.lat_of_y_slice(y_slice)
    target_lon = 0.5 * (lon0 + lon1)
    lon_tol = 0.55 * (lon1 - lon0)

    wrf_data = None
    if not args.no_wrf:
        if not os.path.exists(wrf_nc_path):
            warnings.warn(
                f"WRF file not found: {wrf_nc_path}\n"
                "WRF panel will show a placeholder."
            )
        else:
            print(f"\n[3/3] Loading WRF data ...  ({wrf_time})")
            print(f"      Target lat={target_lat:.6f} N, "
                  f"lon centre={target_lon:.6f} E, lon_tol=±{lon_tol:.5f}°")
            wrf_data = xz.extract_wrf_xz(
                wrf_nc_path,
                target_lat=target_lat, target_lon=target_lon,
                lat_tol=xz.LAT_TOL, lon_tol=lon_tol,
                max_height=args.max_height,
            )
            print(f"      WS range: "
                  f"[{np.nanmin(wrf_data['wind_speed']):.2f}, "
                  f"{np.nanmax(wrf_data['wind_speed']):.2f}] m/s")

    print("\nRendering figure …")
    compose_three_panel(
        wrf_data, simple_data, bbsf_data,
        case_label=os.path.basename(bbsf_dir),
        output_path=output_path,
        y_slice=y_slice,
        shp_path=shp_path,
        max_height=args.max_height,
        cfd_top=args.cfd_top,
    )


if __name__ == "__main__":
    main()
