#!/usr/bin/env python3
"""Plot smoothed interior Λ' and the Λ'>1.2 zones, one figure per hour.

The field and the contours use the same interior rule as local_wind_ratio.py:
|x| and |y| <= 2000 m, building footprints removed, WRF speed at least 0.2 m/s.
Λ' is V_CFD divided by the mean V_CFD on those interior fluid points.
Zones are 8-connected patches of the 25 m smoothed field above 1.2, with area
at least about 625 m². Default hours are the two manuscript snapshots at 30 m.
Each hour is written as interior_lp_<YYYYMMDD_HHMM>_<height>m.png.
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

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_CASES = (
    ("20250901_1000_two_boundaries_as_outlet", "regular 09-01 18:00 (UTC+8)"),
    ("20250908_0000_two_boundaries_as_outlet", "typhoon-affected 09-08 08:00 (UTC+8)"),
)
_DEFAULT_OUT_DIR = os.path.join(_REPO, "results", "wrf_openfoam", "local_wind_ratio")


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def case_stamp(case_name: str) -> str:
    parts = case_name.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 8 and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return case_name


def figure_path(out_dir: str, case_name: str, height: int) -> str:
    return os.path.join(out_dir, f"interior_lp_{case_stamp(case_name)}_{height}m.png")


def plot_one(
    name: str,
    title: str,
    height: int,
    masks: dict,
    root: str,
    axis: np.ndarray,
    out_path: str,
) -> None:
    case_dir = os.path.join(root, name)
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

    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    image = ax.pcolormesh(
        axis, axis, np.ma.masked_invalid(smoothed),
        cmap="coolwarm", vmin=0.6, vmax=1.8, shading="auto",
    )
    ax.contour(axis, axis, kept.astype(float), levels=[0.5], colors="k", linewidths=0.6)
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{title}\n$z$ = {height} m")
    fig.colorbar(image, ax=ax, shrink=0.86, label=r"$\Lambda'$")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(out_path)


def plot_cases(cases: list[tuple[str, str]], height: int, out_dir: str) -> None:
    configure_style()
    masks = L.load_masks(L.MASK_NPZ)
    if height not in masks:
        raise SystemExit(f"No building mask for {height} m in {L.MASK_NPZ}")
    root = os.path.join(_REPO, "steady_experiments_finer_ABL")
    axis = L.grid_axis()
    for name, title in cases:
        plot_one(name, title, height, masks, root, axis, figure_path(out_dir, name, height))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="python analysis/260923/plot_interior_lp.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--height", type=int, default=30, choices=L.HEIGHTS,
        help="Slice height in metres (default 30)",
    )
    parser.add_argument(
        "--out-dir", default=_DEFAULT_OUT_DIR,
        help=f"Directory for one PNG per hour (default {_DEFAULT_OUT_DIR})",
    )
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
    plot_cases(cases, args.height, args.out_dir)


if __name__ == "__main__":
    main()
