"""
WRF-CFD X-Y Horizontal Wind Field (3x2 multi-height)
====================================================
Three heights x WRF|CFD:

    +--------------+--------------+
    | (a) WRF 30 m | (b) CFD 30 m |
    +--------------+--------------+
    | (c) WRF 60 m | (d) CFD 60 m |
    +--------------+--------------+
    | (e) WRF 120m | (f) CFD 120m |
    +--------------+--------------+
              shared colorbar on the right

Usage
-----
    python visualize_WRF_CFD_xy_three_height.py  /path/to/CFD_run_directory

Example
-------
    python visualize_WRF_CFD_xy_three_height.py \\
        steady_experiments_finer_ABL/20250901_0000_two_boundaries_as_outlet

Path inference follows visualize_WRF_CFD_xy_two_panel.py.
PNG out -> results/wrf_openfoam/xy_wrf_cfd/multi_height/xy_wrf_cfd_<stamp>_30m_60m_120m.png
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator

import visualize_WRF_CFD_xy_two_panel as base

DEFAULT_HEIGHTS = (30, 60, 120)
# Not a horizon/vertical layout choice — dedicated multi-height product dir
MULTI_HEIGHT_DIR = "multi_height"
# Analysis module holding the local wind ratio / acceleration-zone definition
ANALYSIS_SUBDIR = os.path.join("analysis", "260923")


def default_output_path(cfd_dir: str, heights: tuple[int, ...]) -> str:
    """``results/wrf_openfoam/xy_wrf_cfd/multi_height/xy_wrf_cfd_<stamp>_<heights>.png``"""
    cfd_dir = cfd_dir.rstrip(os.sep)
    case = os.path.basename(cfd_dir)
    m = re.match(r"(\d{8}_\d{4})", case)
    stamp = m.group(1) if m else case
    htag = "_".join(f"{h}m" for h in heights)
    return os.path.join(
        base._repo_root(), base.RESULTS_XY_DIR, MULTI_HEIGHT_DIR,
        f"xy_wrf_cfd_{stamp}_{htag}.png",
    )


def title_timestamp(case: str) -> str:
    """
    Format case stamp for the figure title.

    Case IDs store UTC time (``YYYYMMDD_HHMM``); the title shows local
    Beijing time as ``YYYY-mm-dd HH:MM (UTC+8)``.
    """
    from datetime import datetime, timedelta, timezone

    m = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})", case)
    if not m:
        m_short = re.match(r"(\d{8}_\d{4})", case)
        return m_short.group(1) if m_short else case
    yr, mo, dy, hh, mm = (int(x) for x in m.groups())
    utc = datetime(yr, mo, dy, hh, mm, tzinfo=timezone.utc)
    local = utc.astimezone(timezone(timedelta(hours=8)))
    return local.strftime("%Y-%m-%d %H:%M (UTC+8)")


def _panel_letter(row: int, col: int, ncols: int = 2) -> str:
    return chr(ord("a") + row * ncols + col)


def _trim_row_xlabels(ax, row: int, n_rows: int) -> None:
    """Hide x labels on non-bottom rows; keep all y labels (WRF/CFD axes differ)."""
    if row < n_rows - 1:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)


def _add_panel_quiver_key(ax, key_u: float, xy=(0.80, 1.05)) -> None:
    """
    Reference arrow that looks identical on WRF (lon/lat) and CFD (m) panels.

    Uses a dedicated quiver with scale_units='width' so key length does not
    depend on the panel's data units / parent-field quiver.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # Seed outside the visible window (never drawn as a field arrow)
    qv = ax.quiver(
        [x0 - 10 * (x1 - x0)], [y0 - 10 * (y1 - y0)], [1.0], [0.0],
        scale=base.QUIVER_SCALE, width=base.QUIVER_WIDTH * 1.15,
        scale_units="width", angles="xy",
        color="black", zorder=10,
    )
    ax.quiverkey(
        qv, X=xy[0], Y=xy[1], U=key_u,
        label=f"{key_u:g} m/s",
        labelpos="E", coordinates="axes",
        fontproperties={"family": "serif", "size": 14, "weight": "bold"},
        labelsep=0.05,
    )


def _format_wrf_lonlat_ticks(ax) -> None:
    """Left-column only: 2-decimal lon/lat; longitude ticks at 113.30 and 113.34."""
    ax.xaxis.set_major_locator(FixedLocator([113.30, 113.34]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def compose_three_height_figure(
    wrf_by_h: dict[int, dict | None],
    cfd_by_h: dict[int, dict],
    heights: tuple[int, ...],
    case_label: str,
    output_path: str,
    shp_path: str | None = None,
    zone_masks: dict[int, np.ndarray] | None = None,
    zone_axis: np.ndarray | None = None,
) -> None:
    """3x2 grid: rows = heights, cols = WRF | CFD; one shared colorbar on the right."""
    base._apply_global_style()

    p98_vals: list[float] = []
    for h in heights:
        cfd = cfd_by_h[h]
        p98_vals.append(float(np.nanpercentile(cfd["wind_speed"], 98)))
        wrf = wrf_by_h.get(h)
        if wrf is not None:
            p98_vals.append(float(np.nanpercentile(wrf["wind_speed"], 98)))
    shared_vmax = base._nice_vmax(max(p98_vals))
    key_u = base._quiver_key_speed(shared_vmax)

    # Shared geographic footprint from first CFD slice (same XY for all heights)
    ref_cfd = cfd_by_h[heights[0]]
    xy_lim = base.cfd_xy_square_limits(ref_cfd)
    lon0, lon1, lat0, lat1 = base.lonlat_limits_from_xy_square(*xy_lim)
    lonlat_lim = (lon0, lon1, lat0, lat1)
    print(
        f"  Shared domain : XY [{xy_lim[0]:.0f},{xy_lim[1]:.0f}]x"
        f"[{xy_lim[2]:.0f},{xy_lim[3]:.0f}] m  ->  "
        f"lon/lat [{lon0:.5f},{lon1:.5f}]x[{lat0:.5f},{lat1:.5f}]"
    )

    building_rings = []
    if any(wrf_by_h.get(h) is not None for h in heights):
        shp = shp_path or os.path.join(base._repo_root(), base.DEFAULT_SHP)
        building_rings = base.load_building_rings_lonlat(
            shp, lon_lim=(lon0, lon1), lat_lim=(lat0, lat1),
        )

    n_rows = len(heights)
    fig_w, fig_h = 12.5, 15.5
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Square panel boxes (same approach as two_panel): panel_w from panel_h
    top, bottom = 0.93, 0.06
    hspace = 0.075          # room for panel title (left) + quiverkey (right)
    left0 = 0.09
    wspace = 0.14
    cbar_gap, cbar_w = 0.035, 0.022

    usable_h = top - bottom - (n_rows - 1) * hspace
    panel_h = usable_h / n_rows
    panel_w = panel_h * fig_h / fig_w

    panel_pos = {}
    axes = np.empty((n_rows, 2), dtype=object)
    for row in range(n_rows):
        y = top - (row + 1) * panel_h - row * hspace
        for col in range(2):
            x = left0 + col * (panel_w + wspace)
            axes[row, col] = fig.add_axes([x, y, panel_w, panel_h])
            panel_pos[(row, col)] = [x, y, panel_w, panel_h]

    stack_right = left0 + 2 * panel_w + wspace
    full_stack_h = top - bottom
    cbar_h = full_stack_h * 0.4
    cbar_bottom = bottom + (full_stack_h - cbar_h) / 2.0
    cax = fig.add_axes([stack_right + cbar_gap, cbar_bottom, cbar_w, cbar_h])

    # Above each axes, top-right (outside data; sits in the row gap)
    quiver_key_xy = (0.78, 1.05)

    titles: dict[tuple[int, int], str] = {}
    mappable = None
    for row, height in enumerate(heights):
        letter_wrf = _panel_letter(row, 0)
        letter_cfd = _panel_letter(row, 1)
        ax_wrf = axes[row, 0]
        ax_cfd = axes[row, 1]

        titles[(row, 0)] = f"({letter_wrf}) WRF at {height:g} m"
        titles[(row, 1)] = f"({letter_cfd}) WRF-to-\nOpenFOAM at {height:g} m"

        wrf_data = wrf_by_h.get(height)
        if wrf_data is not None:
            m = base.draw_wrf_panel(
                ax_wrf, wrf_data, vmax=shared_vmax, label="",
                quiver_key_speed=key_u, show_quiver_key=False,
                add_panel_label=False,
                building_rings=building_rings, lonlat_lim=lonlat_lim,
            )
            if mappable is None:
                mappable = m
            _format_wrf_lonlat_ticks(ax_wrf)
        else:
            ax_wrf.text(
                0.5, 0.5, "WRF data unavailable\n(file not found)",
                ha="center", va="center", transform=ax_wrf.transAxes,
                fontsize=16, color="grey",
            )
            lon_a, lon_b, lat_a, lat_b = base._square_pad_limits(lon0, lon1, lat0, lat1)
            ax_wrf.set_xlim(lon_a, lon_b)
            ax_wrf.set_ylim(lat_a, lat_b)

        hb = base.draw_cfd_panel(
            ax_cfd, cfd_by_h[height], vmax=shared_vmax, label="",
            quiver_key_speed=key_u, show_quiver_key=False,
            add_panel_label=False, xy_lim=xy_lim,
            zone_mask=(zone_masks or {}).get(height), zone_axis=zone_axis,
        )
        if mappable is None:
            mappable = hb

        _add_panel_quiver_key(ax_wrf, key_u, xy=quiver_key_xy)
        _add_panel_quiver_key(ax_cfd, key_u, xy=quiver_key_xy)

        _trim_row_xlabels(ax_wrf, row, n_rows)
        _trim_row_xlabels(ax_cfd, row, n_rows)

    # Lock boxes after equal-aspect datalim so labels stay in the figure
    for (row, col), pos in panel_pos.items():
        axes[row, col].set_position(pos)

    # Panel tags in figure coordinates above each axes — never under vectors
    for (row, col), pos in panel_pos.items():
        x, y, w, h = pos
        fig.text(
            x, y + h + 0.006, titles[(row, col)],
            ha="left", va="bottom", fontsize=16, fontweight="bold",
            transform=fig.transFigure,
        )

    base.add_wind_speed_colorbar(fig, mappable, cax=cax)

    stamp = title_timestamp(case_label)
    fig.suptitle(
        f"X-Y Horizontal Wind Field — {stamp}",
        fontsize=13, fontweight="bold", y=0.985,
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"\nDONE: Figure saved -> {output_path}  (300 DPI)\n")
    plt.close(fig)


def acceleration_zone_masks(
    cfd_dir: str, heights: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Recompute the localized acceleration zones exactly as defined in Section 2.6:
    local wind ratio above 1.2 after 25 m smoothing, minimum area 625 m².

    The analysis module is reused rather than reimplemented, so the zones drawn
    on the figure cannot drift away from the statistics reported in the text.
    Returns the per-height zone masks and the common grid axis in metres.
    """
    ana_dir = os.path.join(base._repo_root(), ANALYSIS_SUBDIR)
    if ana_dir not in sys.path:
        sys.path.insert(0, ana_dir)
    import local_wind_ratio as lwr

    inside_building = lwr.load_masks(lwr.MASK_NPZ)
    masks: dict[int, np.ndarray] = {}
    for h in heights:
        grid = lwr.load_cfd_grid(base.resolve_cfd_csv(cfd_dir, h))
        fluid = grid["sampled"] & ~inside_building[h]
        v_cfd = grid["wind_speed"]
        v_wrf = lwr.wrf_speed_on_grid(cfd_dir, h)
        usable = (
            fluid
            & np.isfinite(v_cfd)
            & np.isfinite(v_wrf)
            & (v_wrf >= lwr.WRF_SPEED_FLOOR)
        )
        ratio = np.full(v_cfd.shape, np.nan, dtype=np.float64)
        ratio[usable] = v_cfd[usable] / v_wrf[usable]
        smoothed = lwr.smooth_fluid(ratio, usable, lwr.BASE_SMOOTH_M)
        kept, zones = lwr.label_zones(
            smoothed, lwr.BASE_THRESHOLD, lwr.BASE_MIN_AREA_M2, above=True,
        )
        masks[h] = kept
        area_km2 = sum(z["area_m2"] for z in zones) / 1e6
        print(f"      {h:g} m : {len(zones)} zones, {area_km2:.1f} km²")
    return masks, lwr.grid_axis()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WRF-CFD X-Y wind field 3x2 multi-height comparison (shared colorbar)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("cfd_dir", help="Path to the CFD run directory")
    p.add_argument(
        "--heights", type=int, nargs="+", default=list(DEFAULT_HEIGHTS),
        help=f"Slice heights in m (default: {' '.join(map(str, DEFAULT_HEIGHTS))})",
    )
    p.add_argument("--wrf-nc", default=None,
                   help="Override auto-detected WRF NetCDF file path")
    p.add_argument("--output", default=None,
                   help="Output PNG path (default: results/wrf_openfoam/xy_wrf_cfd/"
                        "multi_height/xy_wrf_cfd_<stamp>_<heights>.png)")
    p.add_argument("--shp", default=None,
                   help="Building shapefile for WRF panel white basemap "
                        f"(default: {base.DEFAULT_SHP})")
    p.add_argument("--lat", type=float, default=None,
                   help="WRF crop centre latitude (default: from CFD XY domain)")
    p.add_argument("--lon", type=float, default=None,
                   help="WRF crop centre longitude (default: from CFD XY domain)")
    p.add_argument("--lat-tol", type=float, default=None,
                   help="WRF crop half-height in deg (default: from CFD XY domain)")
    p.add_argument("--lon-tol", type=float, default=None,
                   help="WRF crop half-width in deg (default: from CFD XY domain)")
    p.add_argument("--no-wrf", action="store_true",
                   help="Skip WRF panels even if the file is available")
    p.add_argument("--zones", action="store_true",
                   help="Overlay the localized acceleration zones of Section 2.6 "
                        "on the WRF-to-OpenFOAM panels")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfd_dir = args.cfd_dir.rstrip("/")
    heights = tuple(int(h) for h in args.heights)
    if len(heights) < 1:
        raise SystemExit("--heights must contain at least one value")

    wrf_nc_path, _, wrf_time = base.infer_paths(cfd_dir, heights[0])
    if args.wrf_nc:
        wrf_nc_path = args.wrf_nc
    wrf_nc_path = base.resolve_existing_wrf_nc_path(wrf_nc_path)

    basename = os.path.basename(cfd_dir)
    output_path = args.output or default_output_path(cfd_dir, heights)
    shp_path = args.shp or os.path.join(base._repo_root(), base.DEFAULT_SHP)
    wrf_exp = "W_myExp05" if "W_myExp05" in wrf_nc_path else "W_myExp03"

    print("=" * 64)
    print("  WRF-CFD X-Y Comparison (3x2 multi-height)")
    print("=" * 64)
    print(f"  Heights      : {', '.join(f'{h} m' for h in heights)}")
    print(f"  WRF source   : {wrf_exp}")
    print(f"  WRF nc file  : {wrf_nc_path}")
    print(f"  Building SHP : {shp_path}")
    print(f"  Output       : {output_path}")
    print("=" * 64)

    print("\n[1] Loading CFD CSVs ...")
    cfd_by_h: dict[int, dict] = {}
    for h in heights:
        cfd_csv = base.resolve_cfd_csv(cfd_dir, h)
        print(f"  {h:g} m -> {cfd_csv}")
        cfd_by_h[h] = base.load_cfd_csv(cfd_csv)
        valid = cfd_by_h[h].get('valid')
        ws_valid = cfd_by_h[h]['wind_speed'][valid] if valid is not None else cfd_by_h[h]['wind_speed']
        print(
            f"      {len(cfd_by_h[h]['x']):,} points  |  "
            f"fluid {len(ws_valid):,}  |  "
            f"WS [{ws_valid.min():.2f}, {ws_valid.max():.2f}] m/s",
        )

    # WRF crop centred on CFD XY footprint (same as two_panel)
    xy_lim = base.cfd_xy_square_limits(cfd_by_h[heights[0]])
    lon0, lon1, lat0, lat1 = base.lonlat_limits_from_xy_square(*xy_lim)
    target_lat = args.lat if args.lat is not None else 0.5 * (lat0 + lat1)
    target_lon = args.lon if args.lon is not None else 0.5 * (lon0 + lon1)
    lat_tol = args.lat_tol if args.lat_tol is not None else 0.55 * (lat1 - lat0)
    lon_tol = args.lon_tol if args.lon_tol is not None else 0.55 * (lon1 - lon0)

    wrf_by_h: dict[int, dict | None] = {h: None for h in heights}
    if not args.no_wrf:
        if not os.path.exists(wrf_nc_path):
            warnings.warn(
                f"WRF file not found: {wrf_nc_path}\n"
                "WRF panels will show placeholders. Use --wrf-nc to override.",
            )
        else:
            print(f"\n[2] Loading WRF planes ...  ({wrf_time})")
            print(
                f"      Crop centre ({target_lon:.6f} E, {target_lat:.6f} N), "
                f"tol +/-{lon_tol:.5f}deg/+/-{lat_tol:.5f}deg",
            )
            for h in heights:
                wrf_by_h[h] = base.extract_wrf_xy(
                    wrf_nc_path, h,
                    target_lat=target_lat, target_lon=target_lon,
                    lat_tol=lat_tol, lon_tol=lon_tol,
                )

    print("\nRendering figure ...")

    zone_masks = None
    zone_axis = None
    if args.zones:
        print("\n[3] Recomputing localized acceleration zones (Section 2.6) ...")
        zone_masks, zone_axis = acceleration_zone_masks(cfd_dir, heights)

    compose_three_height_figure(
        wrf_by_h, cfd_by_h, heights,
        case_label=basename,
        output_path=output_path,
        shp_path=shp_path,
        zone_masks=zone_masks,
        zone_axis=zone_axis,
    )


if __name__ == "__main__":
    main()
