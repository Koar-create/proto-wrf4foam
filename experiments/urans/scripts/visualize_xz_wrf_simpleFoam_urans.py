#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3-panel X-Z: WRF / simpleFoam / URANS (t=1800 s).

    python experiments/urans/scripts/visualize_xz_wrf_simpleFoam_urans.py 20250901_1000
    python experiments/urans/scripts/visualize_xz_wrf_simpleFoam_urans.py 20250908_0000
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_SCRIPTS = Path(__file__).resolve().parent
URANS_ROOT = _SCRIPTS.parent
_REPO_ROOT = str(URANS_ROOT.parent.parent)
sys.path.insert(0, str(URANS_ROOT / "util"))
import case_paths as cp  # noqa: E402

_XZ_DIR = os.path.join(_REPO_ROOT, "analysis", "260409", "ws-xz-plane")
if _XZ_DIR not in sys.path:
    sys.path.insert(0, _XZ_DIR)
import visualize_WRF_CFD_xz_two_panel as xz  # noqa: E402

VMAX_CAP = {
    "20250901_1000": 12.0,
    "20250908_0000": 28.0,
}


def _hide_yticklabels(ax):
    ax.tick_params(axis="y", labelleft=False)
    ax.set_ylabel("")


def compose_three_panel(wrf_data, simple_data, urans_data, case_label, output_path,
                        y_slice, shp_path, vmax_cap, max_height=xz.MAX_HEIGHT,
                        cfd_top=xz.CFD_TOP):
    xz._apply_global_style()

    x0s, x1s, z0s, z1s = xz.cfd_xz_limits(simple_data)
    x0u, x1u, z0u, z1u = xz.cfd_xz_limits(urans_data)
    xz_lim = (
        min(x0s, x0u), max(x1s, x1u),
        min(z0s, z0u), max(z1s, z1u),
    )
    x0, x1, z0, z1 = xz_lim

    simple_p98 = float(np.nanpercentile(simple_data["wind_speed"], 98))
    urans_p98 = float(np.nanpercentile(urans_data["wind_speed"], 98))
    wrf_p98 = max(simple_p98, urans_p98)
    if wrf_data is not None:
        xz.wrf_lon_to_local_x(wrf_data, y_slice)
        in_view = (wrf_data["height"] >= z0) & (wrf_data["height"] <= z1)
        wrf_ws_view = wrf_data["wind_speed"][in_view]
        wrf_p98 = float(np.nanpercentile(
            wrf_ws_view if np.any(np.isfinite(wrf_ws_view))
            else wrf_data["wind_speed"],
            98,
        ))

    shared_vmax = min(
        xz._nice_vmax(max(simple_p98, urans_p98, wrf_p98)),
        vmax_cap,
    )
    print(
        f"  Shared color  : horizontal WS  |  "
        f"simple p98={simple_p98:.2f}  URANS p98={urans_p98:.2f}  "
        f"WRF p98={wrf_p98:.2f}  →  vmax={shared_vmax:g} m/s "
        f"(cap={vmax_cap:g})"
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
    ax_urans = fig.add_axes([lefts[2], panel_bottom, panel_w, panel_h])
    cax = fig.add_axes([cbar_left, panel_bottom, cbar_w, panel_h])

    labels = (
        "(a) WRF",
        "(b) W-OF (simpleFoam)",
        "(c) W-OF (URANS, t=1800 s)",
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
    hb_urans = xz.draw_cfd_panel(
        ax_urans, urans_data, vmax=shared_vmax, label=labels[2],
        xz_lim=xz_lim, building_polys=building_polys_x,
    )
    _hide_yticklabels(ax_simple)
    _hide_yticklabels(ax_urans)

    if mappable is None:
        mappable = hb_simple
    mappable.set_clim(0, shared_vmax)
    hb_simple.set_clim(0, shared_vmax)
    hb_urans.set_clim(0, shared_vmax)
    xz.add_wind_speed_colorbar(fig, mappable, cax=cax, label="Wind Speed (m/s)")

    for ax, left in zip((ax_wrf, ax_simple, ax_urans), lefts):
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


def main():
    os.chdir(_REPO_ROOT)
    p = argparse.ArgumentParser()
    p.add_argument("tag", help="20250901_1000 or 20250908_0000")
    p.add_argument("--output", default=None)
    p.add_argument("--y-slice", type=float, default=xz.DEFAULT_Y_SLICE)
    args = p.parse_args()
    tag = args.tag.strip().rstrip("/")
    if tag.endswith("_pimplefoam"):
        tag = tag[: -len("_pimplefoam")]
    if tag not in VMAX_CAP:
        raise SystemExit(f"unknown tag {tag}")

    urans_dir = cp.case_dir(tag)
    simple_dir = cp.simplefoam_dir(tag)
    y_slice = args.y_slice
    output_path = args.output or str(
        cp.figs_dir() / f"xz_wrf_simpleFoam_urans_{tag}_y800.png"
    )

    wrf_nc_path, _, wrf_time = xz.infer_paths(str(urans_dir), y_slice)
    wrf_nc_path = xz.resolve_existing_wrf_nc_path(wrf_nc_path)
    simple_csv = xz.resolve_cfd_csv(str(simple_dir), y_slice)
    urans_csv = xz.resolve_cfd_csv(str(urans_dir), y_slice)
    shp_path = os.path.join(_REPO_ROOT, xz.DEFAULT_SHP)

    print("=" * 64)
    print("  WRF / simpleFoam / URANS X-Z Comparison (3-panel)")
    print(f"  Y-slice      : {y_slice:g} m")
    print(f"  simpleFoam   : {simple_csv}")
    print(f"  URANS        : {urans_csv}")
    print(f"  WRF nc file  : {wrf_nc_path}  ({wrf_time})")
    print(f"  Output       : {output_path}")
    print("=" * 64)

    simple_data = xz.load_cfd_csv(simple_csv)
    urans_data = xz.load_cfd_csv(urans_csv)

    x0, x1, _, _ = xz.cfd_xz_limits(urans_data)
    lon0, lon1 = xz.lon_limits_from_x_range(x0, x1, y_slice)
    target_lat = xz.lat_of_y_slice(y_slice)
    target_lon = 0.5 * (lon0 + lon1)
    lon_tol = 0.55 * (lon1 - lon0)

    wrf_data = None
    if os.path.exists(wrf_nc_path):
        wrf_data = xz.extract_wrf_xz(
            wrf_nc_path,
            target_lat=target_lat, target_lon=target_lon,
            lat_tol=xz.LAT_TOL, lon_tol=lon_tol,
            max_height=xz.MAX_HEIGHT,
        )
    else:
        warnings.warn(f"WRF file not found: {wrf_nc_path}")

    compose_three_panel(
        wrf_data, simple_data, urans_data,
        case_label=urans_dir.name,
        output_path=os.path.abspath(output_path),
        y_slice=y_slice,
        shp_path=shp_path,
        vmax_cap=VMAX_CAP[tag],
    )


if __name__ == "__main__":
    main()
