#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-down UAV route overview: building footprints coloured by height.

Writes results/uav_route_wind_shear/route_overview_map.png.
Routes are the same local segments as the wind-shear analysis (one-way).

用法:
  python analysis/260409/uav-route_wind-exposure/plot_uav_route_overview_map.py
  python analysis/260409/uav-route_wind-exposure/plot_uav_route_overview_map.py \\
      --out results/uav_route_wind_shear/route_overview_map.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SHP = REPO_ROOT / "data" / "Guangzhou_shp_file" / "project_UTM49" / "Export_Output.shp"
DEFAULT_ORIGIN_LON = 113.3218197
DEFAULT_ORIGIN_LAT = 23.1133057
DEFAULT_OUT = (
    REPO_ROOT
    / "results"
    / "uav_route_wind_shear"
    / "route_overview_map.png"
)
DEFAULT_CLIP_XY = 1800.0
DEFAULT_SAMPLE_STEP = 20.0

ROUTE_SPECS = {
    "Route1_open_river": {
        "label": "Route 1 (open river)",
        "start": (-1500.0, 0.0),
        "end": (1500.0, 0.0),
    },
    "Route2_urban_canyon": {
        "label": "Route 2 (urban canyon)",
        "start": (400.0, -1000.0),
        "end": (400.0, 1000.0),
    },
}


def configure_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": ":",
            "grid.color": "black",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def build_route(
    name: str,
    p_start: tuple[float, float],
    p_end: tuple[float, float],
    step: float,
) -> pd.DataFrame:
    """One-way samples along a straight segment. Columns: route, x, y."""
    x0, y0 = p_start
    x1, y1 = p_end
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length <= 0:
        raise ValueError(f"Route {name} has zero length")
    n_seg = max(1, int(round(length / step)))
    t = np.linspace(0.0, 1.0, n_seg + 1)
    return pd.DataFrame(
        {
            "route": name,
            "x": x0 + t * (x1 - x0),
            "y": y0 + t * (y1 - y0),
        }
    )


def build_all_routes(sample_step: float) -> dict[str, pd.DataFrame]:
    return {
        name: build_route(name, spec["start"], spec["end"], sample_step)
        for name, spec in ROUTE_SPECS.items()
    }


def origin_utm49n(lon: float, lat: float) -> tuple[float, float]:
    """WGS84 lon/lat (deg) → UTM Zone 49N easting/northing (m)."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    ox, oy = transformer.transform(lon, lat)
    return float(ox), float(oy)


def _rings_from_shape(shp) -> list[np.ndarray]:
    """Extract exterior rings from a pyshp polygon (skip hole rings)."""
    pts = np.asarray(shp.points, dtype=float)
    if pts.size == 0:
        return []
    parts = list(shp.parts) + [len(pts)]
    rings: list[np.ndarray] = []
    for i in range(len(parts) - 1):
        if i > 0:
            break
        ring = pts[parts[i] : parts[i + 1]]
        if ring.shape[0] >= 2 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if ring.shape[0] < 3:
            continue
        x, y = ring[:, 0], ring[:, 1]
        area = abs(0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))
        if area < 5.0:
            continue
        rings.append(ring)
    return rings


def load_building_footprints_local(
    shp_path: Path,
    origin_xy: tuple[float, float],
    clip_xy: float,
    height_field: str = "jzgd",
    encoding: str = "gbk",
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load building rings in local XY, clipped to ±clip_xy."""
    import shapefile

    ox, oy = origin_xy
    reader = shapefile.Reader(str(shp_path), encoding=encoding)
    field_names = [f[0] for f in reader.fields[1:]]
    if height_field not in field_names:
        raise ValueError(f"Height field '{height_field}' not in {field_names}")
    hi = field_names.index(height_field)

    rings: list[np.ndarray] = []
    heights: list[float] = []
    pad = 50.0

    for shp, rec in zip(reader.shapes(), reader.records()):
        if shp.shapeType not in (5, 15, 25):
            continue
        try:
            h = float(rec[hi])
        except (TypeError, ValueError):
            h = float("nan")
        if not np.isfinite(h):
            continue
        for ring_utm in _rings_from_shape(shp):
            ring = np.column_stack([ring_utm[:, 0] - ox, ring_utm[:, 1] - oy])
            if (
                ring[:, 0].max() < -clip_xy - pad
                or ring[:, 0].min() > clip_xy + pad
                or ring[:, 1].max() < -clip_xy - pad
                or ring[:, 1].min() > clip_xy + pad
            ):
                continue
            rings.append(ring)
            heights.append(h)

    print(
        f"[SHP] {len(rings)} footprint rings within ±{clip_xy:.0f} m "
        f"from {shp_path.name}",
        flush=True,
    )
    return rings, np.asarray(heights, dtype=float)


def plot_route_overview(
    route_dfs: dict[str, pd.DataFrame],
    shp_path: Path,
    out_path: Path,
    clip_xy: float = DEFAULT_CLIP_XY,
    origin_lon: float = DEFAULT_ORIGIN_LON,
    origin_lat: float = DEFAULT_ORIGIN_LAT,
) -> None:
    """Top-down map of routes on SHP building footprints (height-coloured)."""
    configure_matplotlib_style()
    font_family = "DejaVu Serif"
    ox, oy = origin_utm49n(origin_lon, origin_lat)
    rings, heights = load_building_footprints_local(Path(shp_path), (ox, oy), clip_xy)

    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#e6ebf0")

    cbar = None
    if len(rings) == 0:
        print("[SHP] WARNING: no footprints in view; drawing routes only", flush=True)
    else:
        vmax = float(np.nanpercentile(heights, 96))
        vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = max(float(np.nanmax(heights)), vmin + 1.0)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "urban_ht",
            ["#f0d5a8", "#e09a3e", "#c45c26", "#7a2e12"],
            N=256,
        )
        patches = [MplPolygon(r, closed=True) for r in rings]
        coll = PatchCollection(
            patches,
            cmap=cmap,
            norm=norm,
            array=heights,
            edgecolors="#3d342c",
            linewidths=0.18,
            alpha=0.95,
            zorder=1,
        )
        ax.add_collection(coll)
        cbar = fig.colorbar(coll, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Building height (m)", fontsize=20, fontfamily=font_family)
        cbar.ax.tick_params(labelsize=16)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontfamily(font_family)

    route_colors = {"Route1_open_river": "#0D47A1", "Route2_urban_canyon": "#B71C1C"}
    endpoint_ann = {
        "Route1_open_river": {
            "start": {"xytext": (0, 22), "ha": "center", "va": "bottom"},
            "end": {"xytext": (0, 22), "ha": "center", "va": "bottom"},
        },
        "Route2_urban_canyon": {
            "start": {"xytext": (22, 0), "ha": "left", "va": "center"},
            "end": {"xytext": (22, 0), "ha": "left", "va": "center"},
        },
    }
    for rname, df in route_dfs.items():
        color = route_colors.get(rname, "k")
        ax.plot(
            df["x"],
            df["y"],
            color=color,
            lw=2.6,
            solid_capstyle="round",
            label=ROUTE_SPECS[rname]["label"],
            zorder=3,
        )
        x0, y0 = float(df["x"].iloc[0]), float(df["y"].iloc[0])
        x1, y1 = float(df["x"].iloc[-1]), float(df["y"].iloc[-1])
        for x, y in ((x0, y0), (x1, y1)):
            ax.scatter(
                x,
                y,
                c="k",
                s=40,
                zorder=4,
                edgecolors="white",
                linewidths=0.7,
            )
        ann = endpoint_ann.get(
            rname,
            {
                "start": {"xytext": (0, 22), "ha": "center", "va": "bottom"},
                "end": {"xytext": (0, 22), "ha": "center", "va": "bottom"},
            },
        )
        for label, (x, y), key in (
            ("Start", (x0, y0), "start"),
            ("End", (x1, y1), "end"),
        ):
            sty = ann[key]
            ax.annotate(
                label,
                xy=(x, y),
                xytext=sty["xytext"],
                textcoords="offset points",
                color=color,
                fontsize=16,
                fontfamily=font_family,
                fontweight="bold",
                ha=sty["ha"],
                va=sty["va"],
                zorder=5,
            )

    ax.set_xlim(-clip_xy, clip_xy)
    ax.set_ylim(-clip_xy, clip_xy)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=20, fontfamily=font_family)
    ax.set_ylabel("y (m)", fontsize=20, fontfamily=font_family)
    ax.tick_params(axis="both", labelsize=18)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily(font_family)
    ax.grid(True, color="0.45", alpha=0.22, linestyle=":", linewidth=0.7)
    ax.legend(
        loc="upper left",
        fontsize=16,
        framealpha=0.92,
        prop={"family": font_family, "size": 16},
    )
    fig.tight_layout()
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily(font_family)
    if cbar is not None:
        cbar.ax.yaxis.label.set_fontfamily(font_family)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontfamily(font_family)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Saved {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shp-path", type=Path, default=DEFAULT_SHP, help="Building footprints shapefile")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output PNG path")
    p.add_argument("--clip-xy", type=float, default=DEFAULT_CLIP_XY, help="Half-width of the map window (m)")
    p.add_argument("--sample-step", type=float, default=DEFAULT_SAMPLE_STEP, help="Route sampling step (m)")
    p.add_argument("--origin-lon", type=float, default=DEFAULT_ORIGIN_LON)
    p.add_argument("--origin-lat", type=float, default=DEFAULT_ORIGIN_LAT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    routes = build_all_routes(float(args.sample_step))
    plot_route_overview(
        routes,
        Path(args.shp_path),
        Path(args.out),
        clip_xy=float(args.clip_xy),
        origin_lon=float(args.origin_lon),
        origin_lat=float(args.origin_lat),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
