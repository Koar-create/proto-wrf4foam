#!/usr/bin/env python3
"""One row of three panels: interior Λ' at 30, 60 and 120 m for one hour.

Uses the same field definition as plot_interior_lp.py and local_wind_ratio.py.
Default cases match the two manuscript snapshot hours. Output:

  results/wrf_openfoam/local_wind_ratio/interior_lp_<YYYYMMDD_HHMM>_row.png

    python analysis/260923/plot_interior_lp_row.py
    python analysis/260923/plot_interior_lp_row.py 20250901_1000_two_boundaries_as_outlet
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_wind_ratio as L  # noqa: E402
from plot_interior_lp import (  # noqa: E402
    _DEFAULT_CASES,
    _DEFAULT_OUT_DIR,
    _REPO,
    annotate_lidar_stations,
    case_stamp,
    configure_style,
)

_LP_VMIN, _LP_VMAX = 0.6, 1.8


def row_figure_path(out_dir: str, case_name: str) -> str:
    return os.path.join(out_dir, f"interior_lp_{case_stamp(case_name)}_row.png")


def smoothed_lp_and_zones(
    case_name: str,
    height: int,
    masks: dict[int, np.ndarray],
    root: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case_dir = os.path.join(root, case_name)
    csv_path = os.path.join(case_dir, "postProcessing", f"{height}m.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"Missing slice: {csv_path}")
    grid = L.load_cfd_grid(csv_path)
    fluid = grid["sampled"] & ~masks[height] & L.interior_mask()
    v = grid["wind_speed"]
    v_wrf = L.wrf_speed_on_grid(case_dir, height)
    usable = fluid & np.isfinite(v) & np.isfinite(v_wrf) & (v_wrf >= L.WRF_SPEED_FLOOR)
    finite_fluid = v[fluid & np.isfinite(v)]
    if finite_fluid.size == 0:
        raise SystemExit(f"No interior fluid speed in {csv_path}")
    mean = float(np.mean(finite_fluid))
    ratio_p = np.full(v.shape, np.nan)
    ratio_p[usable] = v[usable] / mean
    smoothed = L.smooth_fluid(ratio_p, usable, L.BASE_SMOOTH_M)
    kept, _zones = L.label_zones(smoothed, L.BASE_THRESHOLD, L.BASE_MIN_AREA_M2, above=True)
    return L.grid_axis(), smoothed, kept


def plot_row(name: str, title: str, out_dir: str, heights: tuple[int, ...]) -> None:
    masks = L.load_masks(L.MASK_NPZ)
    for h in heights:
        if h not in masks:
            raise SystemExit(f"No building mask for {h} m in {L.MASK_NPZ}")
    root = os.path.join(_REPO, "steady_experiments_finer_ABL")

    fig, axes = plt.subplots(1, len(heights), figsize=(5.2 * len(heights), 4.8), constrained_layout=True)
    if len(heights) == 1:
        axes = [axes]

    mappable = None
    for ax, height in zip(axes, heights):
        axis, smoothed, kept = smoothed_lp_and_zones(name, height, masks, root)
        mappable = ax.pcolormesh(
            axis,
            axis,
            np.ma.masked_invalid(smoothed),
            cmap="coolwarm",
            vmin=_LP_VMIN,
            vmax=_LP_VMAX,
            shading="auto",
        )
        ax.contour(axis, axis, kept.astype(float), levels=[0.5], colors="k", linewidths=0.6)
        annotate_lidar_stations(ax)
        ax.set_xlim(-2000, 2000)
        ax.set_ylim(-2000, 2000)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_title(f"$z$ = {height} m")
    axes[0].set_ylabel("y (m)")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.colorbar(mappable, ax=axes, shrink=0.88, label=r"$\Lambda'$")
    out_path = row_figure_path(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        default=_DEFAULT_OUT_DIR,
        help=f"Output directory (default {_DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--heights",
        type=int,
        nargs="+",
        default=list(L.HEIGHTS),
        help=f"Heights left to right (default: {' '.join(map(str, L.HEIGHTS))})",
    )
    parser.add_argument(
        "cases",
        nargs="*",
        help="Case names under steady_experiments_finer_ABL; default: manuscript hours.",
    )
    args = parser.parse_args()
    heights = tuple(int(h) for h in args.heights)
    for h in heights:
        if h not in L.HEIGHTS:
            raise SystemExit(f"Unsupported height {h} m; choose from {L.HEIGHTS}")

    if args.cases:
        cases = [(name, name) for name in args.cases]
    else:
        cases = list(_DEFAULT_CASES)

    configure_style()
    for name, title in cases:
        plot_row(name, title, args.out_dir, heights)


if __name__ == "__main__":
    main()
