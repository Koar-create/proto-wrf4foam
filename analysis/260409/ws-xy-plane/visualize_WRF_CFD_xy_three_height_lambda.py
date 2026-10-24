"""
WRF–CFD X–Y wind field with a third column of Λ (Figs. 7 and 9).

Same 30 / 60 / 120 m rows and the same WRF | WRF-OpenFOAM panels as
``archive/visualize_WRF_CFD_xy_three_height.py``. The third column is Λ = V_CFD / V_WRF
on the interior (|x|, |y| <= 2000 m), 25 m smoothed, coolwarm with white at
Λ = 1. That field is ``smoothed_lambda_field`` from ``plot_interior_lambda.py``
(the field drawn by ``plot_interior_lambda_row.py``), not Λ'.

    +----------------+------------------+----------------+
    | (a) WRF 30 m   | (b) OF 30 m      | (c) Λ 30 m     |
    +----------------+------------------+----------------+
    | (d) WRF 60 m   | (e) OF 60 m      | (f) Λ 60 m     |
    +----------------+------------------+----------------+
    | (g) WRF 120 m  | (h) OF 120 m     | (i) Λ 120 m    |
    +----------------+------------------+----------------+
         shared wind-speed colorbar          Λ colorbar

All three columns share the interior window |x|, |y| <= 2000 m
(WRF via the matching lon/lat box). Buildings, and cells with WRF
speed below the floor, stay white.

Usage
-----
    python analysis/260409/ws-xy-plane/visualize_WRF_CFD_xy_three_height_lambda.py \\
        steady_experiments_finer_ABL/20250901_1000_two_boundaries_as_outlet

    python analysis/260409/ws-xy-plane/visualize_WRF_CFD_xy_three_height_lambda.py \\
        steady_experiments_finer_ABL/20250908_0000_two_boundaries_as_outlet

PNG -> results/wrf_openfoam/xy_wrf_cfd/multi_height/xy_wrf_cfd_<stamp>_30m_60m_120m_lambda.png
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FormatStrFormatter, NullLocator

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ARCHIVE_DIR = os.path.join(_SCRIPT_DIR, "archive")
for _p in (_SCRIPT_DIR, _ARCHIVE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import visualize_WRF_CFD_xy_three_height as tri
import visualize_WRF_CFD_xy_two_panel as base

_ANA = os.path.join(base._repo_root(), "analysis", "260923")
if _ANA not in sys.path:
    sys.path.insert(0, _ANA)

import local_wind_ratio as L  # noqa: E402
from plot_interior_lambda import (  # noqa: E402
    LAMBDA_VCENTER,
    LAMBDA_VMAX,
    LAMBDA_VMIN,
    lambda_norm,
    smoothed_lambda_field,
)

# Shared display square for WRF, WRF-OpenFOAM, and Λ (local metres).
XY_VIEW_M = 2000.0
XY_VIEW_LIM = (-XY_VIEW_M, XY_VIEW_M, -XY_VIEW_M, XY_VIEW_M)


def default_output_path(cfd_dir: str, heights: tuple[int, ...]) -> str:
    """Sibling of the 3×2 product, with a ``_lambda`` suffix so it does not overwrite it."""
    stem = tri.default_output_path(cfd_dir, heights)
    root, ext = os.path.splitext(stem)
    return f"{root}_lambda{ext}"


def lambda_colorbar_ticks(vmin: float, vcenter: float, vmax: float) -> np.ndarray:
    """Ticks that always include Λ = 1 when that value lies inside the scale."""
    if (
        abs(vmin - LAMBDA_VMIN) < 1e-9
        and abs(vcenter - LAMBDA_VCENTER) < 1e-9
        and abs(vmax - LAMBDA_VMAX) < 1e-9
    ):
        return np.array([0.4, 0.7, 1.0, 1.4, 1.85])
    ticks = [vmin, vcenter, vmax]
    return np.asarray(sorted(set(np.round(ticks, 5))), dtype=float)


def _add_lambda_colorbar(fig, mappable, cax, vmin: float, vcenter: float, vmax: float):
    ticks = lambda_colorbar_ticks(vmin, vcenter, vmax)
    cb = fig.colorbar(mappable, cax=cax, label=r"$\Lambda$")
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    cb.ax.tick_params(labelsize=14)
    return cb


def _format_wrf_lonlat_ticks(ax) -> None:
    """Longitude ticks that remain inside the ±2000 m lon/lat box."""
    ax.xaxis.set_major_locator(FixedLocator([113.31, 113.33]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def _draw_lambda_panel(ax, axis: np.ndarray, field: np.ndarray, norm, xy_lim) -> object:
    """Λ on the ±2000 m window. Buildings stay white."""
    x0, x1, y0, y1 = xy_lim
    ax.set_facecolor("white")
    mesh = ax.pcolormesh(
        axis,
        axis,
        np.ma.masked_invalid(field),
        cmap="coolwarm",
        norm=norm,
        shading="auto",
        rasterized=True,
        zorder=2,
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.margins(0)
    ax.set_xlabel("X (m)", fontweight="bold")
    ax.set_ylabel("Y (m)", fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25, ls="--", lw=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    return mesh


def compose_three_column_figure(
    wrf_by_h: dict[int, dict | None],
    cfd_by_h: dict[int, dict],
    lambda_by_h: dict[int, tuple[np.ndarray, np.ndarray]],
    heights: tuple[int, ...],
    case_label: str,
    output_path: str,
    shp_path: str | None,
    vmin: float,
    vcenter: float,
    vmax: float,
) -> None:
    """3×3 grid: rows = heights; columns = WRF | WRF-OpenFOAM | Λ."""
    base._apply_global_style()
    norm = lambda_norm(vmin, vcenter, vmax)

    p98_vals: list[float] = []
    for h in heights:
        cfd = cfd_by_h[h]
        valid = cfd.get("valid")
        ws = cfd["wind_speed"][valid] if valid is not None else cfd["wind_speed"]
        p98_vals.append(float(np.nanpercentile(ws, 98)))
        wrf = wrf_by_h.get(h)
        if wrf is not None:
            p98_vals.append(float(np.nanpercentile(wrf["wind_speed"], 98)))
    shared_vmax = base._nice_vmax(max(p98_vals))
    key_u = base._quiver_key_speed(shared_vmax)

    xy_lim = XY_VIEW_LIM
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
    n_cols = 3
    # Inches, then converted to figure fractions so the three panels stay square
    # and the two colorbar labels do not collide.
    panel_in = 3.65
    hspace_in = 1.20
    wspace_in = 1.20
    left_in = 1.20
    bottom_in = 1.00
    top_in = 1.15
    cbar_w_in = 0.28
    cbar_gap_in = 0.16
    label_gap_in = 0.62
    right_in = 0.55
    fig_h = top_in + n_rows * panel_in + (n_rows - 1) * hspace_in + bottom_in
    fig_w = (
        left_in
        + n_cols * panel_in
        + (n_cols - 1) * wspace_in
        + cbar_gap_in + cbar_w_in
        + label_gap_in + cbar_w_in
        + right_in
    )
    fig = plt.figure(figsize=(fig_w, fig_h))

    def fx(xin: float) -> float:
        return xin / fig_w

    def fy(yin: float) -> float:
        return yin / fig_h

    panel_pos: dict[tuple[int, int], list[float]] = {}
    axes = np.empty((n_rows, n_cols), dtype=object)
    for row in range(n_rows):
        y_in = bottom_in + (n_rows - 1 - row) * (panel_in + hspace_in)
        for col in range(n_cols):
            x_in = left_in + col * (panel_in + wspace_in)
            pos = [fx(x_in), fy(y_in), fx(panel_in), fy(panel_in)]
            axes[row, col] = fig.add_axes(pos)
            panel_pos[(row, col)] = pos

    stack_top_in = bottom_in + n_rows * panel_in + (n_rows - 1) * hspace_in
    stack_h_in = stack_top_in - bottom_in
    cbar_h_in = stack_h_in * 0.42
    cbar_bottom_in = bottom_in + (stack_h_in - cbar_h_in) / 2.0
    panels_right_in = left_in + n_cols * panel_in + (n_cols - 1) * wspace_in
    cax_wind = fig.add_axes([
        fx(panels_right_in + cbar_gap_in),
        fy(cbar_bottom_in),
        fx(cbar_w_in),
        fy(cbar_h_in),
    ])
    cax_lam = fig.add_axes([
        fx(panels_right_in + cbar_gap_in + cbar_w_in + label_gap_in),
        fy(cbar_bottom_in),
        fx(cbar_w_in),
        fy(cbar_h_in),
    ])

    quiver_key_xy = (0.78, 1.05)
    titles: dict[tuple[int, int], str] = {}
    wind_mappable = None
    lambda_mappable = None

    for row, height in enumerate(heights):
        ax_wrf = axes[row, 0]
        ax_cfd = axes[row, 1]
        ax_lam = axes[row, 2]
        titles[(row, 0)] = f"({tri._panel_letter(row, 0, n_cols)}) WRF at {height:g} m"
        titles[(row, 1)] = (
            f"({tri._panel_letter(row, 1, n_cols)}) WRF-\nOpenFOAM at {height:g} m"
        )
        titles[(row, 2)] = (
            f"({tri._panel_letter(row, 2, n_cols)}) "
            + rf"$\Lambda$ at {height:g} m"
        )

        wrf_data = wrf_by_h.get(height)
        if wrf_data is not None:
            mapped = base.draw_wrf_panel(
                ax_wrf, wrf_data, vmax=shared_vmax, label="",
                quiver_key_speed=key_u, show_quiver_key=False,
                add_panel_label=False,
                building_rings=building_rings, lonlat_lim=lonlat_lim,
            )
            if wind_mappable is None:
                wind_mappable = mapped
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

        mapped_cfd = base.draw_cfd_panel(
            ax_cfd, cfd_by_h[height], vmax=shared_vmax, label="",
            quiver_key_speed=key_u, show_quiver_key=False,
            add_panel_label=False, xy_lim=xy_lim,
        )
        if wind_mappable is None:
            wind_mappable = mapped_cfd

        axis, field = lambda_by_h[height]
        lambda_mappable = _draw_lambda_panel(ax_lam, axis, field, norm, xy_lim)

        tri._add_panel_quiver_key(ax_wrf, key_u, xy=quiver_key_xy)
        tri._add_panel_quiver_key(ax_cfd, key_u, xy=quiver_key_xy)
        for ax in (ax_wrf, ax_cfd, ax_lam):
            tri._trim_row_xlabels(ax, row, n_rows)

    for (row, col), pos in panel_pos.items():
        axes[row, col].set_position(pos)

    for (row, col), pos in panel_pos.items():
        x, y, _w, h = pos
        fig.text(
            x, y + h + 0.004, titles[(row, col)],
            ha="left", va="bottom", fontsize=16, fontweight="bold",
            transform=fig.transFigure,
        )

    if wind_mappable is None:
        raise SystemExit("No wind field to colour.")
    base.add_wind_speed_colorbar(fig, wind_mappable, cax=cax_wind)
    _add_lambda_colorbar(fig, lambda_mappable, cax_lam, vmin, vcenter, vmax)

    stamp = tri.title_timestamp(case_label)
    fig.suptitle(
        f"X-Y Horizontal Wind Field — {stamp}",
        fontsize=13, fontweight="bold", y=1.0 - 0.28 / fig_h,
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"\nDONE: Figure saved -> {output_path}  (300 DPI)\n")
    plt.close(fig)


def _lambda_fields(
    cfd_dir: str, heights: tuple[int, ...],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    masks = L.load_masks(L.MASK_NPZ)
    for h in heights:
        if h not in masks:
            raise SystemExit(f"No building mask for {h} m in {L.MASK_NPZ}")
    cfd_dir = os.path.abspath(cfd_dir)
    root = os.path.dirname(cfd_dir)
    name = os.path.basename(cfd_dir)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for h in heights:
        axis, field = smoothed_lambda_field(name, h, masks, root)
        finite = field[np.isfinite(field)]
        print(
            f"      Λ {h:g} m : n={finite.size:,}  "
            f"[{finite.min():.2f}, {finite.max():.2f}]  "
            f"median {np.median(finite):.2f}"
        )
        out[h] = (axis, field)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WRF | WRF-OpenFOAM | Λ, three heights (Figs. 7 and 9 with a third column)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("cfd_dir", help="Path to the CFD run directory")
    p.add_argument(
        "--heights", type=int, nargs="+", default=list(tri.DEFAULT_HEIGHTS),
        help=f"Slice heights in m (default: {' '.join(map(str, tri.DEFAULT_HEIGHTS))})",
    )
    p.add_argument("--wrf-nc", default=None, help="Override auto-detected WRF NetCDF file path")
    p.add_argument("--output", default=None, help="Output PNG path")
    p.add_argument(
        "--shp", default=None,
        help=f"Building shapefile for the WRF panels (default: {base.DEFAULT_SHP})",
    )
    p.add_argument("--lat", type=float, default=None, help="WRF crop centre latitude")
    p.add_argument("--lon", type=float, default=None, help="WRF crop centre longitude")
    p.add_argument("--lat-tol", type=float, default=None, help="WRF crop half-height in deg")
    p.add_argument("--lon-tol", type=float, default=None, help="WRF crop half-width in deg")
    p.add_argument("--no-wrf", action="store_true", help="Skip WRF panels even if the file is available")
    p.add_argument("--vmin", type=float, default=LAMBDA_VMIN, help="Λ colour-scale minimum")
    p.add_argument("--vcenter", type=float, default=LAMBDA_VCENTER, help="Λ drawn white (default 1)")
    p.add_argument("--vmax", type=float, default=LAMBDA_VMAX, help="Λ colour-scale maximum")
    return p


def _console_utf8() -> None:
    """Windows consoles default to cp1252 and cannot print Λ."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def main() -> None:
    _console_utf8()
    args = build_parser().parse_args()
    cfd_dir = args.cfd_dir.rstrip("/\\")
    heights = tuple(int(h) for h in args.heights)
    if len(heights) < 1:
        raise SystemExit("--heights must contain at least one value")
    for h in heights:
        if h not in L.HEIGHTS:
            raise SystemExit(f"Unsupported height {h} m; choose from {L.HEIGHTS}")

    wrf_nc_path, _, wrf_time = base.infer_paths(cfd_dir, heights[0])
    if args.wrf_nc:
        wrf_nc_path = args.wrf_nc
    wrf_nc_path = base.resolve_existing_wrf_nc_path(wrf_nc_path)

    basename = os.path.basename(cfd_dir)
    output_path = args.output or default_output_path(cfd_dir, heights)
    shp_path = args.shp or os.path.join(base._repo_root(), base.DEFAULT_SHP)
    wrf_exp = "W_myExp05" if "W_myExp05" in wrf_nc_path else "W_myExp03"

    print("=" * 64)
    print("  WRF | WRF-OpenFOAM | Λ   (3×3)")
    print("=" * 64)
    print(f"  Heights      : {', '.join(f'{h} m' for h in heights)}")
    print(f"  WRF source   : {wrf_exp}")
    print(f"  WRF nc file  : {wrf_nc_path}")
    print(f"  Building SHP : {shp_path}")
    print(f"  Λ scale      : [{args.vmin:g}, {args.vcenter:g}, {args.vmax:g}]")
    print(f"  Output       : {output_path}")
    print("=" * 64)

    print("\n[1] Loading CFD CSVs ...")
    cfd_by_h: dict[int, dict] = {}
    for h in heights:
        cfd_csv = base.resolve_cfd_csv(cfd_dir, h)
        print(f"  {h:g} m -> {cfd_csv}")
        cfd_by_h[h] = base.load_cfd_csv(cfd_csv)
        valid = cfd_by_h[h].get("valid")
        ws_valid = (
            cfd_by_h[h]["wind_speed"][valid]
            if valid is not None else cfd_by_h[h]["wind_speed"]
        )
        print(
            f"      {len(cfd_by_h[h]['x']):,} points  |  "
            f"fluid {len(ws_valid):,}  |  "
            f"WS [{ws_valid.min():.2f}, {ws_valid.max():.2f}] m/s",
        )

    xy_lim = XY_VIEW_LIM
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

    print("\n[3] Smoothed interior Λ ...")
    lambda_by_h = _lambda_fields(cfd_dir, heights)

    print("\nRendering figure ...")
    compose_three_column_figure(
        wrf_by_h, cfd_by_h, lambda_by_h, heights,
        case_label=basename,
        output_path=output_path,
        shp_path=shp_path,
        vmin=args.vmin,
        vcenter=args.vcenter,
        vmax=args.vmax,
    )


if __name__ == "__main__":
    main()
