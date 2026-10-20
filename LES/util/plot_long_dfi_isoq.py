#!/usr/bin/env python3
"""Q iso-surface XY projection at t=500 and 1000 for long DFI case."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LES / "20250901_0000_WN_outlet_DFSEM"))
from plot_dfsem_z100 import LIDAR, read_vtp_points_U  # noqa: E402

CASE = LES / "experiments/20250901_1000_long_DFI"
OUT = CASE / "figs_long"
os.environ.setdefault("MPLCONFIGDIR", str(CASE / ".mplconfig"))


def plot_q(t: int):
    p = CASE / "postProcessing" / "isoQ" / str(t) / "qIso1em3.vtp"
    if not p.is_file():
        p = CASE / "postProcessing" / "isoQ" / str(t) / "isoQ_1e-3.vtp"
    pts, q = read_vtp_points_U(p, field="Q")
    m = q.ravel() > 1e-3
    fig, ax = plt.subplots(figsize=(7, 6))
    if m.any():
        ax.scatter(pts[m, 0], pts[m, 1], c=np.log10(q.ravel()[m]), s=0.3, cmap="hot", vmin=-2, vmax=2)
    for name, xy in LIDAR.items():
        ax.plot(*xy, "ko", ms=4)
        ax.annotate(name, xy, fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlim(-2500, 2500)
    ax.set_ylim(-2500, 2500)
    ax.set_title(f"DFI long run Q>1e-3 t={t} s (xy)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"isoQ_xy_t{t}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / f'isoQ_xy_t{t}.png'}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--times", type=int, nargs="+", default=[500, 1000, 1500, 2000])
    args = ap.parse_args()
    for tt in args.times:
        plot_q(tt)
