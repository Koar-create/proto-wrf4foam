#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-panel z=100 m horizontal wind speed at t=1800 s."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
URANS_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(URANS_ROOT / "util"))
import case_paths as cp  # noqa: E402

PANELS = (
    ("20250901_1000", "(a) 2025-09-01 10:00 UTC", 5.0),
    ("20250908_0000", "(b) 2025-09-08 00:00 UTC", 8.0),
)


def load_slice(tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv = cp.case_dir(tag) / "postProcessing" / "100m.csv"
    if not csv.is_file():
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv, usecols=["Coords:0", "Coords:1", "U:0", "U:1", "vtkValidPointMask"])
    ok = (df["vtkValidPointMask"].to_numpy() == 1)
    x = df["Coords:0"].to_numpy()[ok]
    y = df["Coords:1"].to_numpy()[ok]
    ws = np.sqrt(df["U:0"].to_numpy()[ok] ** 2 + df["U:1"].to_numpy()[ok] ** 2)
    finite = np.isfinite(ws)
    return x[finite], y[finite], ws[finite]


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), constrained_layout=True)
    for ax, (tag, title, vmax) in zip(axes, PANELS):
        x, y, ws = load_slice(tag)
        hb = ax.hexbin(
            x, y, C=ws, gridsize=140, cmap="viridis",
            reduce_C_function=np.mean, vmin=0, vmax=vmax, linewidths=0,
        )
        ax.set_aspect("equal")
        ax.set_title(title, loc="left")
        ax.set_xlabel("X (m)")
        cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Wind speed (m/s)")
        print(f"{tag}: n={len(ws):,}  WS p50={np.nanmedian(ws):.2f}  "
              f"p99={np.nanpercentile(ws, 99):.2f}  max={np.nanmax(ws):.2f} m/s")
    axes[0].set_ylabel("Y (m)")
    fig.suptitle("URANS at t = 1800 s, z = 100 m", fontsize=13, fontweight="bold")
    out = cp.figs_dir() / "urans_xy_z100_t1800.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
