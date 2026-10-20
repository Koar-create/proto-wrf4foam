#!/usr/bin/env python3
"""Project Q=1e-3 isosurface points to xy/xz. Output stays in the DFSEM case."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_dfsem_z100 import LIDAR, read_vtp_points_U

CASE = Path("/hpc2hdd/home/zyang248/WRF-OpenFOAM-Coupling/LES/20250901_0000_WN_outlet_DFSEM")
MEAN = Path("/hpc2hdd/home/zyang248/WRF-OpenFOAM-Coupling/LES/20250901_0000_WN_outlet")
OUTDIR = CASE / "figs_dfsem_100s"
os.environ.setdefault("MPLCONFIGDIR", str(CASE / ".mplconfig"))


def iso_path(case: Path, time: str) -> Path:
    return case / "postProcessing" / "isoQ" / time / "qIso1em3.vtp"


def plot_one(pts: np.ndarray, title: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), constrained_layout=True)
    hb0 = axes[0].hexbin(
        pts[:, 0],
        pts[:, 1],
        gridsize=120,
        cmap="inferno",
        mincnt=1,
        extent=(-5000, 5000, -5000, 5000),
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-5000, 5000)
    axes[0].set_ylim(-5000, 5000)
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("xy projection")
    for name, xy in LIDAR.items():
        axes[0].plot(*xy, "c+", ms=7, mew=1.1)
        axes[0].annotate(name, xy, fontsize=7, color="w", xytext=(4, 4), textcoords="offset points")
    fig.colorbar(hb0, ax=axes[0], fraction=0.046, pad=0.04, label="point count")

    hb1 = axes[1].hexbin(
        pts[:, 0],
        pts[:, 2],
        gridsize=120,
        cmap="inferno",
        mincnt=1,
        extent=(-5000, 5000, 0, 2000),
    )
    axes[1].set_xlim(-5000, 5000)
    axes[1].set_ylim(0, 2000)
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("z (m)")
    axes[1].set_title("xz projection")
    fig.colorbar(hb1, ax=axes[1], fraction=0.046, pad=0.04, label="point count")
    fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}  n_points={len(pts)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", default="100")
    ap.add_argument("--mean-time", default="105")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True, mode=0o700)

    dpath = iso_path(CASE, args.time)
    pts_d, _ = read_vtp_points_U(dpath, field="Q")
    plot_one(
        pts_d,
        f"DFSEM Q = 10$^{{-3}}$ isosurface at t = {args.time} s",
        OUTDIR / f"isoQ_1em3_dfsem_t{args.time}.png",
    )

    mpath = iso_path(MEAN, args.mean_time)
    if mpath.exists():
        pts_m, _ = read_vtp_points_U(mpath, field="Q")
        plot_one(
            pts_m,
            f"mean-inlet LES Q = 10$^{{-3}}$ isosurface at t = {args.mean_time} s",
            OUTDIR / f"isoQ_1em3_meaninlet_t{args.mean_time}.png",
        )
        print(f"n_points DFSEM t={args.time}: {len(pts_d)}")
        print(f"n_points mean-inlet t={args.mean_time}: {len(pts_m)}")
        print(f"ratio DFSEM/mean-inlet = {len(pts_d) / max(len(pts_m), 1):.3f}")
        z_d = pts_d[:, 2]
        z_m = pts_m[:, 2]
        print(
            f"DFSEM z min/p50/max {z_d.min():.1f}/{np.median(z_d):.1f}/{z_d.max():.1f}"
        )
        print(
            f"mean-inlet z min/p50/max {z_m.min():.1f}/{np.median(z_m):.1f}/{z_m.max():.1f}"
        )


if __name__ == "__main__":
    main()
