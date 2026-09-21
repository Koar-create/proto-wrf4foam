#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV 低空航线多高度对比（2×1 叠画）：WRF vs WRF-to-OpenFOAM (CFD)

基于 uav_route_wind_shear_analysis.py，一次提取 z=30/60/120 m：
  - fig1：2(Route) 叠画沿轨水平风速（颜色=模式蓝/红，线型=高度 -/--/:）
  - fig2：2(Route) 叠画沿轨垂直风切变 |∂V/∂Z|（同编码）
不绘制 route_overview_map 等其它图。

默认快照：2025-09-03 12:00:00 UTC
垂直风切变：Δz=40 m，中心差分
  |∂V/∂Z| = sqrt(du²+dv²+dw²) / Δz，V=(U,V,W)

输出目录（区别于原脚本 results/uav_route_wind_shear/）：
  results/uav_route_wind_shear_multi_z/<YYYYMMDD_HHMM>/

用法:
  python analysis/260409/uav_route_wind_shear_multi_z_analysis.py
  python analysis/260409/uav_route_wind_shear_multi_z_analysis.py --datetime "2025-09-03 12:00:00"
  python analysis/260409/uav_route_wind_shear_multi_z_analysis.py --heights 30,60,120 --shear-dz 40
  python analysis/260409/uav_route_wind_shear_multi_z_analysis.py --replot-from-csv \\
      --out-dir results/uav_route_wind_shear_multi_z/20250903_1200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Reuse extraction / style helpers from the single-height analysis script
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import uav_route_wind_shear_analysis as base  # noqa: E402

REPO_ROOT = base.REPO_ROOT
DEFAULT_DATETIME = base.DEFAULT_DATETIME
DEFAULT_SAMPLE_STEP = base.DEFAULT_SAMPLE_STEP
DEFAULT_SHEAR_DZ = base.DEFAULT_SHEAR_DZ
DEFAULT_STL = base.DEFAULT_STL
DEFAULT_CELL_CENTRES = base.DEFAULT_CELL_CENTRES
ROUTE_SPECS = base.ROUTE_SPECS
DEFAULT_HEIGHTS = (30.0, 60.0, 120.0)

# Keep paper-consistent encoding: color = model, linestyle = height
COLOR_WRF = "#4C78A8"
COLOR_CFD = "#E45756"
# (linestyle, linewidth) per height index — solid/dash/dot for z low→high
HEIGHT_LINESTYLES = ("-", "--", ":")
HEIGHT_LINEWIDTHS_WRF = (2.6, 2.2, 1.8)
HEIGHT_LINEWIDTHS_CFD = (1.6, 1.4, 1.2)


# ---------------------------------------------------------------------------
# Heights
# ---------------------------------------------------------------------------
def parse_heights(text: str) -> tuple[float, ...]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("--heights must contain at least one value")
    return tuple(vals)


def union_extract_heights(
    target_heights: tuple[float, ...],
    shear_dz: float,
) -> tuple[float, ...]:
    """Mid levels plus ±Δz/2 neighbours needed for central-difference shear."""
    hs: set[float] = set()
    half = float(shear_dz) / 2.0
    for h in target_heights:
        hs.add(float(h))
        hs.add(float(h) - half)
        hs.add(float(h) + half)
    return tuple(sorted(hs))


def add_shear_for_heights(
    df: pd.DataFrame,
    target_heights: tuple[float, ...],
    shear_dz: float,
    prefix: str,
) -> pd.DataFrame:
    """Compute Vector_Shear_<prefix>_<z> (and refresh WS_<prefix>_<z>) for each height."""
    out = df
    for h in target_heights:
        out = base.add_vertical_shear(out, float(h), float(shear_dz), prefix)
        tag = int(round(h))
        out[f"Vector_Shear_{prefix}_{tag}"] = out[f"Vector_Shear_{prefix}"]
        # WS_<prefix>_<tag> already from extract; keep untagged as last height only
    return out


# ---------------------------------------------------------------------------
# Plotting: 2 (route) overlay — color=model (blue/red), linestyle=height
# ---------------------------------------------------------------------------
def _draw_building_profile_overlay(
    ax,
    distance_km: np.ndarray,
    building_height_m: np.ndarray,
    *,
    ref_heights: tuple[float, ...],
    label: str = "Building height",
):
    """Building fill once; subtle gray z-refs on the twin Hb axis (not competing with curves)."""
    h = np.asarray(building_height_m, dtype=float)
    d = np.asarray(distance_km, dtype=float)
    if h.size == 0:
        return None

    h_plot = np.nan_to_num(h, nan=0.0)
    h_max = float(np.nanmax(h_plot)) if h_plot.size else 0.0
    refs = [float(r) for r in ref_heights if r is not None and r > 0]
    ymax = max(h_max * 1.15, max(refs, default=0.0) * 1.25, 40.0)

    ax_b = ax.twinx()
    ax_b.fill_between(
        d,
        0.0,
        h_plot,
        step="mid",
        color="#8D6E4A",
        alpha=0.35,
        linewidth=0.0,
        zorder=0,
        label=label if h_max > 0 else None,
    )
    if h_max > 0:
        ax_b.plot(d, h_plot, color="#5D4037", lw=0.6, alpha=0.65, zorder=1, solid_capstyle="butt")
    # Single-style gray refs so they do not steal the blue/red model encoding
    for rh in refs:
        ax_b.axhline(rh, color="0.45", ls=":", lw=0.8, alpha=0.55, zorder=1)
    ax_b.set_ylim(0.0, ymax)
    ax_b.set_ylabel(r"$H_\mathrm{b}$ (m)", color="#5D4037", fontsize=10)
    ax_b.tick_params(axis="y", colors="#5D4037", labelsize=9)
    ax_b.grid(False)

    ax.set_zorder(ax_b.get_zorder() + 1)
    ax.patch.set_visible(False)
    return ax_b


def _overlay_legend_handles(target_heights: tuple[float, ...]) -> list:
    """Legend handles packed for ncol=3 columns: WRF | CFD | Building.

    Matplotlib fills column-major and pads earlier columns when N is not
    divisible by ncol. With 7 items that becomes 3|2|2 and pushes
    ``WRF-to-OpenFOAM z=120 m`` into column 3. Pad to 9 (=3×3) so columns
    stay WRF (3) | CFD (3) | Building + blanks (3).
    """
    handles: list = []
    for i, h in enumerate(target_heights):
        tag = int(round(h))
        ls = HEIGHT_LINESTYLES[i % len(HEIGHT_LINESTYLES)]
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLOR_WRF,
                lw=HEIGHT_LINEWIDTHS_WRF[i % len(HEIGHT_LINEWIDTHS_WRF)],
                ls=ls,
                label=f"WRF z={tag} m",
            )
        )
    for i, h in enumerate(target_heights):
        tag = int(round(h))
        ls = HEIGHT_LINESTYLES[i % len(HEIGHT_LINESTYLES)]
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLOR_CFD,
                lw=HEIGHT_LINEWIDTHS_CFD[i % len(HEIGHT_LINEWIDTHS_CFD)],
                ls=ls,
                label=f"WRF-to-OpenFOAM z={tag} m",
            )
        )
    handles.append(Patch(facecolor="#8D6E4A", alpha=0.35, edgecolor="none", label="Building height"))
    # Invisible spacers so ncol=3 yields equal-height columns (3|3|3)
    n_pad = (-len(handles)) % 3
    for _ in range(n_pad):
        handles.append(Line2D([], [], color="none", lw=0, label=" "))
    return handles


def _place_overlay_legend(fig, target_heights: tuple[float, ...]) -> None:
    fig.legend(
        handles=_overlay_legend_handles(target_heights),
        loc="lower center",
        ncol=3,
        fontsize=8.5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.005),
        columnspacing=1.4,
        handlelength=2.4,
    )


def plot_figure1_wind_speed_overlay(
    route_dfs: dict[str, pd.DataFrame],
    out_path: Path,
    target_heights: tuple[float, ...],
    datetime_utc: pd.Timestamp | None = None,
) -> None:
    base.configure_matplotlib_style()
    route_order = list(ROUTE_SPECS.keys())
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=False)

    for ax, rname in zip(axes, route_order):
        df = route_dfs[rname]
        spec = ROUTE_SPECS[rname]
        dist_km = df["distance_m"].to_numpy() / 1000.0
        bh = (
            df["building_height_m"].to_numpy()
            if "building_height_m" in df.columns
            else np.zeros(len(df))
        )
        _draw_building_profile_overlay(ax, dist_km, bh, ref_heights=target_heights)

        for i, h in enumerate(target_heights):
            tag = int(round(h))
            ls = HEIGHT_LINESTYLES[i % len(HEIGHT_LINESTYLES)]
            ax.plot(
                dist_km,
                df[f"WS_wrf_{tag}"],
                color=COLOR_WRF,
                lw=HEIGHT_LINEWIDTHS_WRF[i % len(HEIGHT_LINEWIDTHS_WRF)],
                ls=ls,
                alpha=0.92,
                zorder=2,
            )
            ax.plot(
                dist_km,
                df[f"WS_cfd_{tag}"],
                color=COLOR_CFD,
                lw=HEIGHT_LINEWIDTHS_CFD[i % len(HEIGHT_LINEWIDTHS_CFD)],
                ls=ls,
                alpha=0.95,
                zorder=3,
            )

        ax.set_ylabel(r"$U$ (m/s)")
        ax.set_title(spec["label"])
        ax.set_xlim(dist_km.min(), dist_km.max())
        ax.set_xlabel("Distance along route (km)")

    _place_overlay_legend(fig, target_heights)

    z_txt = "/".join(str(int(round(h))) for h in target_heights)
    time_txt = base.format_datetime_utc8(datetime_utc) if datetime_utc is not None else None
    title = f"Along-track horizontal wind speed  (z={z_txt} m)"
    if time_txt:
        title = f"{title}\n{time_txt}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Plot] Saved {out_path}", flush=True)


def plot_figure2_shear_overlay(
    route_dfs: dict[str, pd.DataFrame],
    out_path: Path,
    target_heights: tuple[float, ...],
    shear_dz: float,
    datetime_utc: pd.Timestamp | None = None,
) -> None:
    base.configure_matplotlib_style()
    route_order = list(ROUTE_SPECS.keys())
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=False)

    for ax, rname in zip(axes, route_order):
        df = route_dfs[rname]
        spec = ROUTE_SPECS[rname]
        dist_km = df["distance_m"].to_numpy() / 1000.0
        bh = (
            df["building_height_m"].to_numpy()
            if "building_height_m" in df.columns
            else np.zeros(len(df))
        )
        _draw_building_profile_overlay(ax, dist_km, bh, ref_heights=target_heights)

        for i, h in enumerate(target_heights):
            tag = int(round(h))
            ls = HEIGHT_LINESTYLES[i % len(HEIGHT_LINESTYLES)]
            ax.plot(
                dist_km,
                df[f"Vector_Shear_wrf_{tag}"],
                color=COLOR_WRF,
                lw=HEIGHT_LINEWIDTHS_WRF[i % len(HEIGHT_LINEWIDTHS_WRF)],
                ls=ls,
                alpha=0.92,
                zorder=2,
            )
            ax.plot(
                dist_km,
                df[f"Vector_Shear_cfd_{tag}"],
                color=COLOR_CFD,
                lw=HEIGHT_LINEWIDTHS_CFD[i % len(HEIGHT_LINEWIDTHS_CFD)],
                ls=ls,
                alpha=0.95,
                zorder=3,
            )

        ax.axhline(0.0, color="0.4", lw=0.6, zorder=1)
        ax.set_ylabel("Vertical Wind Shear Magnitude (1/s)")
        ax.set_title(f"{spec['label']}  (Δz={int(shear_dz)} m)")
        ax.set_xlim(dist_km.min(), dist_km.max())
        ax.set_xlabel("Distance along route (km)")

    _place_overlay_legend(fig, target_heights)

    z_txt = "/".join(str(int(round(h))) for h in target_heights)
    time_txt = base.format_datetime_utc8(datetime_utc) if datetime_utc is not None else None
    title = f"Along-track vertical wind shear  (z={z_txt} m)"
    if time_txt:
        title = f"{title}\n{time_txt}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Plot] Saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--datetime", default=DEFAULT_DATETIME, help="UTC datetime YYYY-MM-DD HH:MM:SS")
    p.add_argument(
        "--heights",
        default="30,60,120",
        help="Comma-separated analysis heights (m), default 30,60,120",
    )
    p.add_argument("--sample-step", type=float, default=DEFAULT_SAMPLE_STEP, help="Route sampling step (m)")
    p.add_argument("--shear-dz", type=float, default=DEFAULT_SHEAR_DZ, help="Shear layer thickness Δz (m)")
    p.add_argument("--wrf-nc", type=Path, default=None, help="Explicit WRF cartesian NetCDF path")
    p.add_argument("--cfd-case-dir", type=Path, default=None, help="Explicit OpenFOAM case directory")
    p.add_argument("--cell-centres", type=Path, default=DEFAULT_CELL_CENTRES, help="Fallback cellCentres / 0/C")
    p.add_argument("--stl-path", type=Path, default=DEFAULT_STL, help="Buildings binary STL (obstruction)")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default under results/uav_route_wind_shear_multi_z/)")
    p.add_argument(
        "--replot-from-csv",
        action="store_true",
        help="Skip extraction; replot fig1/fig2 from existing along-track CSVs in --out-dir",
    )
    return p.parse_args()


def default_out_dir(dt: pd.Timestamp) -> Path:
    tag = dt.strftime("%Y%m%d_%H%M")
    return REPO_ROOT / "results" / "uav_route_wind_shear_multi_z" / tag


def csv_paths(out_dir: Path, z_tag: str) -> dict[str, Path]:
    return {
        "Route1_open_river": out_dir / f"route1_along_track_multi_z{z_tag}.csv",
        "Route2_urban_canyon": out_dir / f"route2_along_track_multi_z{z_tag}.csv",
    }


def load_routes_from_csv(out_dir: Path, z_tag: str) -> dict[str, pd.DataFrame]:
    paths = csv_paths(out_dir, z_tag)
    routes: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing CSV for replot: {path}")
        routes[name] = pd.read_csv(path)
        print(f"[CSV] Loaded {path}  shape={routes[name].shape}", flush=True)
    return routes


def save_figures(
    routes: dict[str, pd.DataFrame],
    out_dir: Path,
    target_heights: tuple[float, ...],
    shear_dz: float,
    dt: pd.Timestamp,
) -> None:
    z_tag = "-".join(str(int(round(h))) for h in target_heights)
    plot_figure1_wind_speed_overlay(
        routes,
        out_dir / f"fig1_wind_speed_along_route_multi_z{z_tag}.png",
        target_heights,
        datetime_utc=dt,
    )
    plot_figure2_shear_overlay(
        routes,
        out_dir / f"fig2_vertical_wind_shear_along_route_multi_z{z_tag}.png",
        target_heights,
        shear_dz,
        datetime_utc=dt,
    )


def main() -> int:
    args = parse_args()
    dt = pd.Timestamp(args.datetime)
    target_heights = parse_heights(args.heights)
    shear_dz = float(args.shear_dz)
    extract_heights = union_extract_heights(target_heights, shear_dz)

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(dt)
    out_dir.mkdir(parents=True, exist_ok=True)

    z_tag = "-".join(str(int(round(h))) for h in target_heights)

    print("=" * 64)
    print("UAV route wind / shear multi-z analysis (2×1 overlay)")
    print(f"  datetime       : {dt}")
    print(f"  target heights : {target_heights} m")
    print(f"  shear Δz       : {shear_dz} m")
    print(f"  out_dir        : {out_dir}")
    print("=" * 64)

    if args.replot_from_csv:
        routes = load_routes_from_csv(out_dir, z_tag)
        save_figures(routes, out_dir, target_heights, shear_dz, dt)
        print("\nDone.")
        return 0

    print(f"  extract zs     : {extract_heights}")
    print(f"  sample step    : {args.sample_step} m")

    # --- Routes ---
    routes = base.build_all_routes(args.sample_step)
    for name, df in routes.items():
        print(
            f"[Route] {name}: {len(df)} pts, total distance={df['distance_m'].iloc[-1]:.0f} m"
        )

    # --- Buildings ---
    roof_index = base.load_building_roof_index(Path(args.stl_path))
    for name in list(routes.keys()):
        routes[name] = base.annotate_obstruction(
            routes[name], roof_index, (30.0, 60.0, 120.0)
        )

    # --- WRF ---
    wrf_nc = base.resolve_wrf_nc(dt, args.wrf_nc)
    for name in list(routes.keys()):
        routes[name] = base.extract_wrf_along_route(wrf_nc, routes[name], extract_heights)
        routes[name] = add_shear_for_heights(
            routes[name], target_heights, shear_dz, "wrf"
        )

    # --- CFD ---
    cfd_case = base.resolve_cfd_case(dt, args.cfd_case_dir)
    tree, u_band = base.load_cfd_velocity_band(
        cfd_case, extract_heights, cell_centres_fallback=args.cell_centres
    )
    for name in list(routes.keys()):
        routes[name] = base.extract_cfd_along_route(
            routes[name], extract_heights, tree, u_band
        )
        routes[name] = add_shear_for_heights(
            routes[name], target_heights, shear_dz, "cfd"
        )

    # --- CSV (reproducibility; no overview map) ---
    paths = csv_paths(out_dir, z_tag)
    keep_cols_base = [
        "route",
        "leg",
        "seq",
        "x",
        "y",
        "distance_m",
        "building_height_m",
        "obstructed_30",
        "obstructed_60",
        "obstructed_120",
    ]
    extra: list[str] = []
    for h in extract_heights:
        tag = int(round(h))
        for prefix in ("wrf", "cfd"):
            for var in ("U", "V", "W", "WS"):
                extra.append(f"{var}_{prefix}_{tag}")
    for h in target_heights:
        tag = int(round(h))
        for prefix in ("wrf", "cfd"):
            extra.append(f"Vector_Shear_{prefix}_{tag}")

    for name, path in paths.items():
        df = routes[name]
        cols = [c for c in keep_cols_base + extra if c in df.columns]
        df[cols].to_csv(path, index=False)
        print(f"[CSV] {path}  shape={df[cols].shape}")

    # --- Figures only (no route_overview_map) ---
    save_figures(routes, out_dir, target_heights, shear_dz, dt)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
