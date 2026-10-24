"""Figure style shared with the inbox prototypes (colour-blind safe)."""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib as mpl

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
C_RANS = "#2a78d6"
C_URANS = "#eb6834"
C_LES = "#1baf7a"
C_OBS = INK
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"
SOLID = "#6f6e69"


def apply():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 6.8,
        "savefig.dpi": 220,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
    })


def stamp(fig, text, top=False):
    fig.text(
        0.995, 0.997 if top else 0.004, text,
        ha="right", va="top" if top else "bottom",
        fontsize=6.2, color="#b23a3a", alpha=0.9, style="italic",
    )


def panel_label(ax, s):
    ax.text(
        -0.02, 1.02, s, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.5, fontweight="bold", color=INK,
    )


def savefig(fig, path, **kwargs):
    """Write PNG with retries — Windows previewers often lock the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("bbox_inches", "tight")
    tmp = path.with_name(path.stem + ".__tmp" + path.suffix)
    last = None
    fig.savefig(tmp, **kwargs)
    for i in range(10):
        try:
            tmp.replace(path)
            return
        except OSError as e:
            last = e
            time.sleep(0.5 * (i + 1))
    try:
        fig.savefig(path, **kwargs)
        tmp.unlink(missing_ok=True)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise last


def fig_legend(fig, handles, labels, ncol=4, y=-0.02, fontsize=6.5):
    fig.legend(
        handles, labels, loc="lower center", ncol=ncol,
        bbox_to_anchor=(0.5, y), fontsize=fontsize, handlelength=2.2,
    )
