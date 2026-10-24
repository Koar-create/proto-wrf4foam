#!/usr/bin/env python3
"""Fig 6 / F04: station-level virtual DBS error from T5 (no full-domain grid)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import FIGS, OUT
from matplotlib.patches import Patch

from style import INK, INK2, SEQ, apply, panel_label, savefig

apply()


def main():
    df = pd.read_csv(OUT / "T5_virtual_dbs_stations.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.8), sharey=True)
    zs = [30, 60, 100, 120]
    angs = [15.0, 18.62, 30.0]
    cols = [SEQ[2], SEQ[4], SEQ[6]]
    for ax, station in zip(axs, ["GAW103", "GAW104", "GAW111"]):
        pos, data, c = [], [], []
        for i, z in enumerate(zs):
            for j, ang in enumerate(angs):
                e = 100 * df[(df.station == station) & (df.z == z) & np.isclose(df.theta, ang)].rel.to_numpy()
                data.append(e[np.isfinite(e)])
                pos.append(i * 4 + j)
                c.append(cols[j])
        bp = ax.boxplot(
            data, positions=pos, widths=0.7, showfliers=False, patch_artist=True,
            medianprops=dict(color=INK, lw=1.0),
            whiskerprops=dict(color=INK2, lw=0.7),
            capprops=dict(color=INK2, lw=0.7),
        )
        for b, col in zip(bp["boxes"], c):
            b.set_facecolor(col)
            b.set_edgecolor("white")
            b.set_linewidth(0.6)
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_xticks([1, 5, 9, 13])
        ax.set_xticklabels(["30", "60", "100", "120"])
        ax.set_title(station, loc="left")
        ax.set_xlabel("z (m)")
        if ax is axs[0]:
            ax.set_ylabel("relative error (%)")
        panel_label(ax, "abc"[["GAW103", "GAW104", "GAW111"].index(station)])
    fig.legend(
        [Patch(color=c) for c in cols],
        ["15° (elev. 75°)", "18.6° (GAW105 ref.)", "30° (elev. 60°)"],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=6.5,
    )
    fig.suptitle("Virtual DBS − point value, 21 snapshots t ≥ 1000 s (non-uniformity error, lower bound)",
                 x=0.01, ha="left", fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    savefig(fig, FIGS / "F04_virtual_DBS.png")
    plt.close(fig)
    print("wrote", FIGS / "F04_virtual_DBS.png")


if __name__ == "__main__":
    main()
