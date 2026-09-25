#!/usr/bin/env python3
"""Supplementary along-track wind speed and shear at two representative hours.

Regular: 2025-09-01 18:00 BJT (case 20250901_1000).
Typhoon: 2025-09-08 08:00 BJT (case 20250908_0000), same hour as Figs. 9–10.
Building interiors are already NaN in the sampled arrays.
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
IMG = _REPO / "docs" / "scs-wrf-of-manuscript" / "images"
HEIGHTS = (30, 60, 120)
ROUTES = ("Route1_open_river", "Route2_urban_canyon")
TITLES = {
    "Route1_open_river": "Route 1 (open river)",
    "Route2_urban_canyon": "Route 2 (urban canyon)",
}
CASES = (
    ("20250901_1000_two_boundaries_as_outlet", "regular", "2025-09-01 18:00 BJT"),
    ("20250908_0000_two_boundaries_as_outlet", "typhoon", "2025-09-08 08:00 BJT"),
)
COLOR_WRF = "#4C78A8"
COLOR_CFD = "#E45756"
LSTYLES = ("-", "--", ":")


def _building(ax, dist, roof, heights):
    ax_b = ax.twinx()
    ax_b.fill_between(dist, 0.0, roof, step="mid", color="#8D6E4A", alpha=0.35, linewidth=0)
    ymax = max(float(np.nanmax(roof)) * 1.15, max(heights) * 1.25, 40.0)
    ax_b.set_ylim(0.0, ymax)
    ax_b.set_ylabel(r"$H_b$ (m)", color="#5D4037", fontsize=10)
    ax_b.tick_params(axis="y", colors="#5D4037", labelsize=9)
    ax.set_zorder(ax_b.get_zorder() + 1)
    ax.patch.set_visible(False)
    return ax_b


def _legend(fig):
    handles = []
    for i, h in enumerate(HEIGHTS):
        handles.append(Line2D([0], [0], color=COLOR_WRF, lw=2.2, ls=LSTYLES[i], label=f"WRF z={h} m"))
    for i, h in enumerate(HEIGHTS):
        handles.append(Line2D([0], [0], color=COLOR_CFD, lw=1.5, ls=LSTYLES[i], label=f"WRF-OpenFOAM z={h} m"))
    handles.append(Patch(facecolor="#8D6E4A", alpha=0.35, label="Building height"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))


def plot_case(case: str, kind: str, when: str, quantity: str) -> Path:
    base.configure_matplotlib_style()
    points = pd.read_csv(OUT / "route_points.csv")
    wind = np.load(OUT / "along" / f"{case}.npz")
    shear = np.load(OUT / "shear_along" / f"{case}.npz") if quantity == "shear" else None
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.0), sharex=False)
    for ax, route in zip(axes, ROUTES):
        df = points[points["route"] == route]
        dist = df["distance_m"].to_numpy() / 1000.0
        _building(ax, dist, df["building_height_m"].to_numpy(), HEIGHTS)
        for i, _h in enumerate(HEIGHTS):
            if quantity == "wind":
                y_wrf = wind[f"{route}__ws_wrf"][i]
                y_cfd = wind[f"{route}__ws_cfd"][i]
            else:
                y_wrf = shear[f"{route}__shear_wrf"][i]
                y_cfd = shear[f"{route}__shear_cfd"][i]
            ax.plot(dist, y_wrf, color=COLOR_WRF, lw=2.2, ls=LSTYLES[i])
            ax.plot(dist, y_cfd, color=COLOR_CFD, lw=1.5, ls=LSTYLES[i])
        ax.set_xlim(dist.min(), dist.max())
        ax.set_title(TITLES[route])
        ax.set_xlabel("Distance along route (km)")
        if quantity == "wind":
            ax.set_ylabel(r"$U$ (m/s)")
        else:
            ax.axhline(0.0, color="0.4", lw=0.6)
            ax.set_ylabel(r"Vertical wind shear (s$^{-1}$)")
    _legend(fig)
    label = "horizontal wind speed" if quantity == "wind" else "vertical wind shear"
    fig.suptitle(f"Along-track {label}\n{when}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    name = f"outline-figS1-{kind}-wind.png" if quantity == "wind" else f"outline-figS1-{kind}-shear.png"
    path = IMG / name
    fig.savefig(path)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"[Plot] {path}", flush=True)
    return path


def main() -> int:
    for case, kind, when in CASES:
        plot_case(case, kind, when, "wind")
        plot_case(case, kind, when, "shear")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
