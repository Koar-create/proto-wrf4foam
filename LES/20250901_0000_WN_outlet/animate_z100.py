#!/usr/bin/env python3
"""z=100 horizontal-speed frames from surfacesU VTPs. Output in this LES case."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

from plot_les_vs_rans_z100 import CASE, HOTSPOT, read_vtp_points_U

os.environ.setdefault("MPLCONFIGDIR", str(CASE / ".mplconfig"))
OUT = CASE / "figs_les_300s" / "z100_frames"
SURF = CASE / "postProcessing" / "surfacesU"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    times = sorted(int(p.name) for p in SURF.iterdir() if p.name.isdigit())
    xi = np.linspace(-2500, 2500, 400)
    yi = np.linspace(-2500, 2500, 400)
    XX, YY = np.meshgrid(xi, yi)
    vmax = 1.4
    norm = Normalize(vmin=0.0, vmax=vmax)
    for t in times:
        path = SURF / str(t) / "z100.vtp"
        pts, vec = read_vtp_points_U(path)
        ws = np.hypot(vec[:, 0], vec[:, 1])
        zz = griddata(pts[:, :2], ws, (XX, YY), method="linear")
        fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
        im = ax.pcolormesh(XX, YY, zz, shading="auto", cmap="viridis", norm=norm)
        ax.set_aspect("equal")
        ax.set_xlim(-2500, 2500)
        ax.set_ylim(-2500, 2500)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"LES WALE  z=100 m  t={t} s  (instantaneous Ws)")
        ax.plot(*HOTSPOT, "r+", ms=8, mew=1.2)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("horizontal wind speed (m s$^{-1}$)")
        fig.savefig(OUT / f"z100_t{t:03d}.png", dpi=110)
        plt.close(fig)
        print(t, f"p50={np.nanmedian(ws):.3f} max={np.nanmax(ws):.3f}")
    print(f"wrote {len(times)} frames -> {OUT}")


if __name__ == "__main__":
    main()
