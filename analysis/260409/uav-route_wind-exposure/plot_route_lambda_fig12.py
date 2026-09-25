#!/usr/bin/env python3
"""Fig. 12: along-track Lambda, hourly median and interquartile band."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import uav_route_wind_shear_analysis as base  # noqa: E402

OUT = _REPO / "results" / "uav_route_exposure_hourly"
FIG = _REPO / "docs" / "scs-wrf-of-manuscript" / "images" / "outline-fig12.png"
HEIGHTS = (30, 60, 120)
ROUTES = ("Route1_open_river", "Route2_urban_canyon")
ROUTE_TITLES = {
    "Route1_open_river": "Route 1 (open river)",
    "Route2_urban_canyon": "Route 2 (urban canyon)",
}
COLOR = {"regular": "#4C78A8", "typhoon": "#F58518"}


def _regime_of_case(case: str) -> str:
    # Case name is UTC. Typhoon window matches local_wind_ratio.regime_of.
    ts = pd.Timestamp(case[:8] + " " + case[9:11] + ":00:00", tz="UTC")
    start = pd.Timestamp("2025-09-07 03:00:00", tz="UTC")
    end = pd.Timestamp("2025-09-08 19:00:00", tz="UTC")
    return "typhoon" if start <= ts <= end else "regular"


def load_lambda() -> dict[str, dict[str, np.ndarray]]:
    """route -> regime -> array (n_hours, n_height, n_points)."""
    buckets: dict[str, dict[str, list]] = {r: {"regular": [], "typhoon": []} for r in ROUTES}
    for path in sorted((OUT / "along").glob("*.npz")):
        regime = _regime_of_case(path.stem)
        with np.load(path) as data:
            for route in ROUTES:
                buckets[route][regime].append(data[f"{route}__lam"])
    stacked = {}
    for route in ROUTES:
        stacked[route] = {reg: np.stack(v, axis=0) for reg, v in buckets[route].items() if v}
    return stacked


def main() -> int:
    base.configure_matplotlib_style()
    points = pd.read_csv(OUT / "route_points.csv")
    stacked = load_lambda()
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6), sharey=True)
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
                arr = stacked[route][regime][:, c_i, :]
                med = np.nanmedian(arr, axis=0)
                q25 = np.nanpercentile(arr, 25, axis=0)
                q75 = np.nanpercentile(arr, 75, axis=0)
                ax.fill_between(dist, q25, q75, color=COLOR[regime], alpha=0.22, linewidth=0)
                ax.plot(dist, med, color=COLOR[regime], lw=1.6)
            ax.axhline(1.0, color="0.35", lw=0.8, ls="--")
            ax.set_xlim(dist.min(), dist.max())
            ax.set_title(f"{ROUTE_TITLES[route]}\n{height} m", fontsize=11)
            if r_i == 1:
                ax.set_xlabel("Distance along route (km)")
            if c_i == 0:
                ax.set_ylabel(r"$\Lambda = V_{CFD}/V_{WRF}$")
            ax.set_zorder(ax_b.get_zorder() + 1)
            ax.patch.set_visible(False)
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color=COLOR["regular"], lw=1.8, label="Regular, median"),
            plt.Line2D([0], [0], color=COLOR["typhoon"], lw=1.8, label="Typhoon, median"),
            Patch(facecolor=COLOR["regular"], alpha=0.22, label="Regular, interquartile"),
            Patch(facecolor=COLOR["typhoon"], alpha=0.22, label="Typhoon, interquartile"),
            plt.Line2D([0], [0], color="0.35", lw=0.8, ls="--", label=r"$\Lambda = 1$"),
            Patch(facecolor="#8D6E4A", alpha=0.28, label="Building height"),
        ],
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)
    fig.savefig(OUT / "outline-fig12.png")
    plt.close(fig)
    print(f"[Plot] {FIG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
