#!/usr/bin/env python3
"""Step 1 only: score candidate transects on the existing z=60 m map and draw a locator.

Does not read 3-D U fields or write profile CSV/figures.

  python analysis/260909-grid-sensitivity/prepare_five_probes.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CFG_PATH = Path(__file__).with_name("probes_5pt.json")
PV = REPO / "results" / "grid_sensitivity" / "paraview"
OUT_FIG = REPO / "results" / "grid_sensitivity" / "five_probes_locator_z60m.png"
LIDAR = {"GAW103": (975.0, -320.0), "GAW104": (450.0, 350.0), "GAW111": (75.0, 30.0)}


def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _add_panel_label(ax, idx: int) -> None:
    ax.text(
        0.04, 0.97, f"({chr(ord('a') + idx)})",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=12, fontweight="bold", zorder=10,
        color="k",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
    )


def load_mag(key: str):
    p = PV / f"Uz60_{key}.csv"
    u0s, u1s, u2s, mask = [], [], [], []
    with p.open() as f:
        for row in csv.DictReader(f):
            u0s.append(float(row["U:0"]))
            u1s.append(float(row["U:1"]))
            u2s.append(float(row.get("U:2", "0") or 0.0))
            mask.append(float(row.get("vtkValidPointMask", "1")))
    v = np.sqrt(np.asarray(u0s) ** 2 + np.asarray(u1s) ** 2 + np.asarray(u2s) ** 2)
    v = np.where(np.asarray(mask) > 0.5, v, np.nan)
    n = int(np.sqrt(len(v)))
    x1d = np.linspace(-2000.0, 2000.0, n)
    y1d = np.linspace(-2000.0, 2000.0, n)
    return n, x1d, y1d, v.reshape(n, n)


def main() -> None:
    configure_matplotlib_style()
    with CFG_PATH.open(encoding="utf-8") as f:
        cfg = json.load(f)
    _n, x1d, y1d, u2 = load_mag("M2")
    u1 = load_mag("M1")[3]
    u3 = load_mag("M3")[3]
    d12 = u1 - u2
    d32 = u3 - u2
    xx, yy = np.meshgrid(x1d, y1d)
    probes = cfg["probes"]
    xs = [p["x"] for p in probes]
    ys = [p["y"] for p in probes]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for i, (ax, field, title) in enumerate((
        (axes[0], d12, r"$|U|_{\mathrm{M1}}-|U|_{\mathrm{M2}}$"),
        (axes[1], d32, r"$|U|_{\mathrm{M3}}-|U|_{\mathrm{M2}}$"),
    )):
        im = ax.pcolormesh(xx, yy, field, cmap="RdBu_r", vmin=-0.6, vmax=0.6, shading="auto")
        ax.plot(xs, ys, color="k", lw=1.2, ls="--")
        ax.scatter(xs, ys, c="k", s=28, zorder=5)
        for p in probes:
            ax.annotate(p["id"], (p["x"], p["y"]), textcoords="offset points", xytext=(4, 4), fontsize=9)
        for name, (lx, ly) in LIDAR.items():
            ax.scatter([lx], [ly], c="0.35", s=12, marker="o")
            ax.annotate(name, (lx, ly), textcoords="offset points", xytext=(4, -10), fontsize=8, color="0.35")
        ax.set_aspect("equal")
        ax.set_xlim(-2000, 2000)
        ax.set_ylim(-2000, 2000)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        _add_panel_label(ax, i)
    axes[0].set_ylabel("y (m)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(r"$\Delta|U|$ (m s$^{-1}$)")
    fig.suptitle("2025-09-01 18:00 (UTC+8) | z = 60 m", fontweight="bold", y=1.04)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300)
    plt.close(fig)
    print(f"wrote {OUT_FIG}")
    print("frozen probes:")
    for p in probes:
        print(f"  {p['id']:3s}  ({p['x']:7.1f}, {p['y']:7.1f})")


if __name__ == "__main__":
    main()
