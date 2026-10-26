#!/usr/bin/env python3
"""Fig. 11: along-track wind speed, hourly median.

WRF-OpenFOAM is a solid median with an interquartile band.
Standalone WRF is a dashed median without a band.
Hue and linestyle follow Fig. 4 (orange dashed WRF, teal solid WRF-OpenFOAM).
Regular-weather lines are lightened so the two regimes stay separable.
Both regimes share one wind-speed axis so the absolute contrast is visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import uav_route_wind_shear_analysis as base  # noqa: E402

OUT = _REPO / "results" / "uav_route_exposure_hourly"
FIG = _REPO / "docs" / "scs-wrf-of-manuscript" / "images" / "scs-fig11.png"
HEIGHTS = (30, 60, 120)
ROUTES = ("Route1_open_river", "Route2_urban_canyon")
ROUTE_TITLES = {
    "Route1_open_river": "Route 1 (open river)",
    "Route2_urban_canyon": "Route 2 (urban canyon)",
}
# Same model colors and line styles as Fig. 4 (plot_timeseries_three_sites_two_heights.py).
COLOR_WRF = "#e07b39"
COLOR_CFD = "#2196a5"
NCOLS = 3


def _lighten(hex_color: str, mix: float) -> str:
    """Mix a hex color toward white. mix=0 keeps the color, mix=1 is white."""
    rgb = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    out = [int(round(c + (255 - c) * mix)) for c in rgb]
    return "#{:02x}{:02x}{:02x}".format(*out)


# Darker lines are typhoon-affected; lighter lines are regular-weather.
LINE = {
    "regular": {"wrf": _lighten(COLOR_WRF, 0.68), "cfd": _lighten(COLOR_CFD, 0.68)},
    "typhoon": {"wrf": COLOR_WRF, "cfd": COLOR_CFD},
}


def _panel_letter(row: int, col: int, ncols: int = NCOLS) -> str:
    return chr(ord("a") + row * ncols + col)


def _add_panel_label(ax, row: int, col: int, ncols: int = NCOLS) -> None:
    ax.text(
        0.04, 0.97, f"({_panel_letter(row, col, ncols)})",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=12, fontweight="bold", zorder=10,
    )


def _regime_of_case(case: str) -> str:
    ts = pd.Timestamp(case[:8] + " " + case[9:11] + ":00:00", tz="UTC")
    start = pd.Timestamp("2025-09-07 03:00:00", tz="UTC")
    end = pd.Timestamp("2025-09-08 19:00:00", tz="UTC")
    return "typhoon" if start <= ts <= end else "regular"


def load_wind() -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """route -> regime -> {cfd, wrf} array (n_hours, n_height, n_points)."""
    buckets: dict[str, dict[str, dict[str, list]]] = {
        r: {"regular": {"cfd": [], "wrf": []}, "typhoon": {"cfd": [], "wrf": []}}
        for r in ROUTES
    }
    for path in sorted((OUT / "along").glob("*.npz")):
        regime = _regime_of_case(path.stem)
        with np.load(path) as data:
            for route in ROUTES:
                buckets[route][regime]["cfd"].append(data[f"{route}__ws_cfd"])
                buckets[route][regime]["wrf"].append(data[f"{route}__ws_wrf"])
    stacked = {}
    for route in ROUTES:
        stacked[route] = {}
        for regime in ("regular", "typhoon"):
            stacked[route][regime] = {
                key: np.stack(val, axis=0) for key, val in buckets[route][regime].items() if val
            }
    return stacked


def main() -> int:
    base.configure_matplotlib_style()
    points = pd.read_csv(OUT / "route_points.csv")
    stacked = load_wind()
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6), sharey=True)
    band_hi = 0.0
    for r_i, route in enumerate(ROUTES):
        df = points[points["route"] == route]
        dist = df["distance_m"].to_numpy() / 1000.0
        roof = df["building_height_m"].to_numpy()
        for c_i, height in enumerate(HEIGHTS):
            ax = axes[r_i, c_i]
            ax_b = ax.twinx()
            ax_b.fill_between(dist, 0.0, roof, step="mid", color="#8D6E4A", alpha=0.28, linewidth=0)
            ax_b.set_ylim(0.0, max(40.0, float(np.nanmax(roof)) * 1.25))
            ax_b.set_ylabel(r"$H_b$ (m)" if c_i == 2 else "", color="#5D4037", fontsize=9)
            ax_b.tick_params(axis="y", labelsize=8, colors="#5D4037")
            if c_i != 2:
                ax_b.set_yticklabels([])
            for regime in ("regular", "typhoon"):
                cfd = stacked[route][regime]["cfd"][:, c_i, :]
                wrf = stacked[route][regime]["wrf"][:, c_i, :]
                med = np.nanmedian(cfd, axis=0)
                q25 = np.nanpercentile(cfd, 25, axis=0)
                q75 = np.nanpercentile(cfd, 75, axis=0)
                wrf_med = np.nanmedian(wrf, axis=0)
                band_hi = max(band_hi, float(np.nanmax(q75)), float(np.nanmax(wrf_med)))
                ax.fill_between(dist, q25, q75, color=LINE[regime]["cfd"], alpha=0.28, linewidth=0, zorder=2)
                ax.plot(dist, wrf_med, color=LINE[regime]["wrf"], lw=1.15, ls="--", zorder=3)
                ax.plot(dist, med, color=LINE[regime]["cfd"], lw=1.7, ls="-", zorder=4)
            ax.set_xlim(dist.min(), dist.max())
            ax.set_title(f"{ROUTE_TITLES[route]}\n{height} m", fontsize=11)
            if r_i == 1:
                ax.set_xlabel("Distance along route (km)")
            if c_i == 0:
                ax.set_ylabel(r"$U$ (m s$^{-1}$)")
            ax.set_zorder(ax_b.get_zorder() + 1)
            ax.patch.set_visible(False)
            _add_panel_label(ax, r_i, c_i)
    # One axis for every panel. 8 m/s was the planned top; raise it only if a
    # plotted median or upper quartile would otherwise be clipped.
    ymax = 8.0 if band_hi <= 8.0 else float(np.ceil(band_hi))
    axes[0, 0].set_ylim(0.0, ymax)
    print(f"[ylim] band/median max {band_hi:.2f} -> 0–{ymax:.0f}", flush=True)
    fig.legend(
        handles=[
            Line2D([0], [0], color=LINE["regular"]["wrf"], lw=1.15, ls="--", label="WRF, regular"),
            Line2D([0], [0], color=LINE["typhoon"]["wrf"], lw=1.15, ls="--", label="WRF, typhoon"),
            Line2D([0], [0], color=LINE["regular"]["cfd"], lw=1.7, label="WRF-OpenFOAM, regular"),
            Line2D([0], [0], color=LINE["typhoon"]["cfd"], lw=1.7, label="WRF-OpenFOAM, typhoon"),
            Patch(facecolor=LINE["regular"]["cfd"], alpha=0.28, label="WRF-OpenFOAM, regular, interquartile"),
            Patch(facecolor=LINE["typhoon"]["cfd"], alpha=0.28, label="WRF-OpenFOAM, typhoon, interquartile"),
            Patch(facecolor="#8D6E4A", alpha=0.28, label="Building height"),
        ],
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)
    fig.savefig(OUT / "route_along_track_wind.png")
    plt.close(fig)
    print(f"[Plot] {FIG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
