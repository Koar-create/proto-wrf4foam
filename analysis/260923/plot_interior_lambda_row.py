#!/usr/bin/env python3
"""One row of three panels: interior Λ at 30, 60 and 120 m for one hour.

Same field as plot_interior_lambda.py. Output:

  results/wrf_openfoam/local_wind_ratio/interior_lambda_<YYYYMMDD_HHMM>_row.png

    python analysis/260923/plot_interior_lambda_row.py
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
from plot_interior_lambda import (  # noqa: E402
    LAMBDA_VCENTER,
    LAMBDA_VMAX,
    LAMBDA_VMIN,
    lambda_norm,
    smoothed_lambda_field,
)
from plot_interior_lp import (  # noqa: E402
    _DEFAULT_CASES,
    _DEFAULT_OUT_DIR,
    _REPO,
    annotate_lidar_stations,
    case_stamp,
    configure_style,
)


def row_figure_path(out_dir: str, case_name: str) -> str:
    return os.path.join(out_dir, f"interior_lambda_{case_stamp(case_name)}_row.png")


def plot_row(
    name: str,
    title: str,
    out_dir: str,
    heights: tuple[int, ...],
    vmin: float,
    vcenter: float,
    vmax: float,
) -> None:
    norm = lambda_norm(vmin, vcenter, vmax)
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
        axis, smoothed = smoothed_lambda_field(name, height, masks, root)
        mappable = ax.pcolormesh(
            axis,
            axis,
            np.ma.masked_invalid(smoothed),
            cmap="coolwarm",
            norm=norm,
            shading="auto",
        )
        annotate_lidar_stations(ax)
        ax.set_xlim(-2000, 2000)
        ax.set_ylim(-2000, 2000)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_title(f"$z$ = {height} m")
    axes[0].set_ylabel("y (m)")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.colorbar(mappable, ax=axes, shrink=0.88, label=r"$\Lambda$")
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
    parser.add_argument("--vmin", type=float, default=LAMBDA_VMIN)
    parser.add_argument("--vcenter", type=float, default=LAMBDA_VCENTER)
    parser.add_argument("--vmax", type=float, default=LAMBDA_VMAX)
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
        plot_row(name, title, args.out_dir, heights, args.vmin, args.vcenter, args.vmax)


if __name__ == "__main__":
    main()
