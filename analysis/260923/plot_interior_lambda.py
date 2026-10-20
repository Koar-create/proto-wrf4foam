#!/usr/bin/env python3
"""Plot smoothed interior Λ = V_CFD / V_WRF, one figure per hour and height.

Same interior mask and smoothing as plot_interior_lp.py and local_wind_ratio.py:
|x| and |y| <= 2000 m, building footprints removed, WRF speed at least 0.2 m/s,
25 m box mean on fluid points. No Λ'>1.2 patch contours (those apply only to Λ').

Output: results/wrf_openfoam/local_wind_ratio/interior_lambda_<YYYYMMDD_HHMM>_<height>m.png

    python analysis/260923/plot_interior_lambda.py
    python analysis/260923/plot_interior_lambda.py --height 60
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
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

# Diverging scale: white at Λ = 1 (equal to WRF); ends tightened around 1.0.
LAMBDA_VMIN = 0.4
LAMBDA_VCENTER = 1.0
LAMBDA_VMAX = 1.85


def lambda_norm(vmin: float, vcenter: float, vmax: float) -> Normalize:
    if not (vmin < vcenter < vmax):
        raise ValueError(f"Need vmin < vcenter < vmax, got {vmin}, {vcenter}, {vmax}")
    return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


def figure_path(out_dir: str, case_name: str, height: int) -> str:
    return os.path.join(out_dir, f"interior_lambda_{case_stamp(case_name)}_{height}m.png")


def smoothed_lambda_field(
    case_name: str,
    height: int,
    masks: dict[int, np.ndarray],
    root: str,
) -> tuple[np.ndarray, np.ndarray]:
    case_dir = os.path.join(root, case_name)
    csv_path = os.path.join(case_dir, "postProcessing", f"{height}m.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"Missing slice: {csv_path}")
    grid = L.load_cfd_grid(csv_path)
    fluid = grid["sampled"] & ~masks[height] & L.interior_mask()
    v = grid["wind_speed"]
    v_wrf = L.wrf_speed_on_grid(case_dir, height)
    usable = fluid & np.isfinite(v) & np.isfinite(v_wrf) & (v_wrf >= L.WRF_SPEED_FLOOR)
    if not np.any(usable):
        raise SystemExit(f"No usable interior Λ points in {csv_path}")
    ratio = np.full(v.shape, np.nan)
    ratio[usable] = v[usable] / v_wrf[usable]
    smoothed = L.smooth_fluid(ratio, usable, L.BASE_SMOOTH_M)
    return L.grid_axis(), smoothed


def plot_one(
    name: str,
    title: str,
    height: int,
    masks: dict,
    root: str,
    out_path: str,
    vmin: float,
    vcenter: float,
    vmax: float,
) -> None:
    axis, smoothed = smoothed_lambda_field(name, height, masks, root)
    norm = lambda_norm(vmin, vcenter, vmax)

    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    image = ax.pcolormesh(
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
    ax.set_ylabel("y (m)")
    ax.set_title(f"{title}\n$z$ = {height} m")
    fig.colorbar(image, ax=ax, shrink=0.86, label=r"$\Lambda$")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(out_path)


def plot_cases(
    cases: list[tuple[str, str]],
    height: int,
    out_dir: str,
    vmin: float,
    vcenter: float,
    vmax: float,
) -> None:
    configure_style()
    masks = L.load_masks(L.MASK_NPZ)
    if height not in masks:
        raise SystemExit(f"No building mask for {height} m in {L.MASK_NPZ}")
    root = os.path.join(_REPO, "steady_experiments_finer_ABL")
    for name, title in cases:
        plot_one(
            name, title, height, masks, root,
            figure_path(out_dir, name, height), vmin, vcenter, vmax,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--height", type=int, default=30, choices=L.HEIGHTS,
        help="Slice height in metres (default 30)",
    )
    parser.add_argument(
        "--out-dir", default=_DEFAULT_OUT_DIR,
        help=f"Output directory (default {_DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--vmin", type=float, default=LAMBDA_VMIN, help="Colour scale minimum")
    parser.add_argument(
        "--vcenter", type=float, default=LAMBDA_VCENTER,
        help="Λ at which coolwarm is white (default 1.0)",
    )
    parser.add_argument("--vmax", type=float, default=LAMBDA_VMAX, help="Colour scale maximum")
    parser.add_argument(
        "cases", nargs="*",
        help="Case directory names under steady_experiments_finer_ABL. "
        "Default: the two manuscript hours.",
    )
    args = parser.parse_args()
    if args.cases:
        cases = [(name, name) for name in args.cases]
    else:
        cases = list(_DEFAULT_CASES)
    plot_cases(cases, args.height, args.out_dir, args.vmin, args.vcenter, args.vmax)


if __name__ == "__main__":
    main()
