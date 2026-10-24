"""Shared matplotlib style and WRF / OpenFOAM colours."""

from __future__ import annotations

import matplotlib.pyplot as plt

COLOR_OBS = "#1a1a2e"
COLOR_WRF = "#e07b39"
COLOR_CFD = "#2196a5"
COLOR_LOW = "#2c7fb8"
COLOR_MID = "#d95f5f"
COLOR_LOW_NIGHT = "#7fcdbb"
COLOR_MID_NIGHT = "#f0a48f"


def configure_matplotlib_style(*, font_size: float = 13) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": font_size,
            "axes.labelsize": font_size + 1,
            "axes.titlesize": font_size + 2,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
