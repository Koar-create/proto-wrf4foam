from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ─── Configuration ────────────────────────────────────────
COLOR_WRF = "#e07b39"
COLOR_CFD = "#2196a5"

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
CIRCULAR_CSV = ANALYSIS_DIR / "wind_direction_metrics_circular.csv"
VECTOR_CSV = ANALYSIS_DIR / "wind_direction_metrics_vector.csv"
VEER_CSV = ANALYSIS_DIR / "wind_direction_metrics_veer.csv"

OUTPUT_DIR = REPO_ROOT / "results/wind_direction_metrics/260409"


def configure_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
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


def plot_bars(ax: plt.Axes, df: pd.DataFrame, spec: dict) -> None:
    """Helper to plot grouped bars for layer metrics"""
    categories = ["All Times", "Daytime", "Nighttime"]
    layer_offsets = {"Low (52-300 m)": -0.18, "Mid (300-1000 m)": 0.18}
    model_offsets = {"wrf": -0.055, "cfd": 0.055}
    width = 0.10

    for x, category in enumerate(categories):
        sub = df[df["Category"] == category]
        for layer, layer_offset in layer_offsets.items():
            layer_data = sub[sub["Layer"] == layer]
            if layer_data.empty:
                continue
            row = layer_data.iloc[0]
            
            for model, model_offset in model_offsets.items():
                col_name = spec[f"{model}_col"]
                value = row[col_name]
                color = COLOR_WRF if model == "wrf" else COLOR_CFD
                alpha = 0.98 if layer.startswith("Low") else 0.52
                ax.bar(
                    x + layer_offset + model_offset,
                    value,
                    width=width,
                    color=color,
                    alpha=alpha,
                    edgecolor="white",
                    linewidth=0.6,
                )

    if spec.get("zero_line"):
        ax.axhline(0, color="0.35", lw=0.8)
    
    if spec.get("ylim"):
        ax.set_ylim(*spec["ylim"])
        
    ax.set_xticks(range(len(categories)), ["All", "Daytime", "Nighttime"])
    ax.set_ylabel(spec["ylabel"])
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def build_layered_metrics_figure(df_circ: pd.DataFrame, df_vec: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.09, hspace=0.34, wspace=0.22)
    fig.suptitle(
        "Wind Direction Metrics (Low & Mid Layers)",
        y=0.985,
        fontsize=13,
        fontweight="bold",
    )

    # Panel Specs
    specs = [
        {
            "ax": axes[0, 0],
            "df": df_circ,
            "title": "(a) Circular Mean Bias Error",
            "ylabel": r"MBE ($^\circ$)",
            "wrf_col": "WRF_Circ_MBE",
            "cfd_col": "CFD_Circ_MBE",
            "zero_line": True,
        },
        {
            "ax": axes[0, 1],
            "df": df_circ,
            "title": "(b) Circular Root Mean Square Error",
            "ylabel": r"RMSE ($^\circ$)",
            "wrf_col": "WRF_Circ_RMSE",
            "cfd_col": "CFD_Circ_RMSE",
        },
        {
            "ax": axes[1, 0],
            "df": df_circ,
            "title": r"(c) Direction Accuracy ($\leq$ 22.5$^\circ$)",
            "ylabel": "Accuracy (%)",
            "wrf_col": "WRF_Acc_22.5(%)",
            "cfd_col": "CFD_Acc_22.5(%)",
            "ylim": (0, 100),
        },
        {
            "ax": axes[1, 1],
            "df": df_vec,
            "title": "(d) Vector RMSE",
            "ylabel": r"RMSE (m s$^{-1}$)",
            "wrf_col": "WRF_Vector_RMSE",
            "cfd_col": "CFD_Vector_RMSE",
        },
    ]

    for s in specs:
        plot_bars(s["ax"], s["df"], s)
        s["ax"].set_title(s["title"])

    # Legend
    legend_handles = [
        Patch(facecolor=COLOR_WRF, alpha=0.98, label="WRF low (52-300 m)"),
        Patch(facecolor=COLOR_CFD, alpha=0.98, label="CFD low"),
        Patch(facecolor=COLOR_WRF, alpha=0.52, label="WRF mid (300-1000 m)"),
        Patch(facecolor=COLOR_CFD, alpha=0.52, label="CFD mid"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncols=4,
        frameon=False,
        handlelength=0.8,
        columnspacing=1.4,
    )
    return fig


def build_veer_metrics_figure(df_veer: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.75, bottom=0.15, wspace=0.22)
    fig.suptitle(
        "Wind Veer Metrics (Low to Mid Layer)",
        y=0.985,
        fontsize=13,
        fontweight="bold",
    )

    categories = ["All Times", "Daytime", "Nighttime"]
    x = np.arange(len(categories))
    width = 0.25
    offsets = {"wrf": -width/2, "cfd": width/2}

    specs = [
        {
            "ax": axes[0],
            "title": "(a) Veer Mean Bias Error",
            "ylabel": r"MBE ($^\circ$/100m)",
            "wrf_col": "WRF_Veer_MBE",
            "cfd_col": "CFD_Veer_MBE",
            "zero_line": True,
        },
        {
            "ax": axes[1],
            "title": "(b) Veer Root Mean Square Error",
            "ylabel": r"RMSE ($^\circ$/100m)",
            "wrf_col": "WRF_Veer_RMSE",
            "cfd_col": "CFD_Veer_RMSE",
            "zero_line": False,
        },
    ]

    for s in specs:
        ax = s["ax"]
        for i, category in enumerate(categories):
            row = df_veer[df_veer["Category"] == category].iloc[0]
            
            # plot wrf
            ax.bar(
                i + offsets["wrf"],
                row[s["wrf_col"]],
                width=width,
                color=COLOR_WRF,
                alpha=0.98,
                edgecolor="white",
                linewidth=0.6,
            )
            # plot cfd
            ax.bar(
                i + offsets["cfd"],
                row[s["cfd_col"]],
                width=width,
                color=COLOR_CFD,
                alpha=0.98,
                edgecolor="white",
                linewidth=0.6,
            )

        if s["zero_line"]:
            ax.axhline(0, color="0.35", lw=0.8)
            
        ax.set_xticks(x, ["All", "Daytime", "Nighttime"])
        ax.set_ylabel(s["ylabel"])
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        ax.set_title(s["title"])

    # Legend
    legend_handles = [
        Patch(facecolor=COLOR_WRF, alpha=0.98, label="WRF"),
        Patch(facecolor=COLOR_CFD, alpha=0.98, label="CFD"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncols=2,
        frameon=False,
        handlelength=0.8,
        columnspacing=1.4,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot wind direction metrics (excluding High layer).")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args()

    configure_matplotlib_style()

    # Load data
    df_circ = pd.read_csv(CIRCULAR_CSV)
    df_vec = pd.read_csv(VECTOR_CSV)
    df_veer = pd.read_csv(VEER_CSV)

    # Filter out High layer
    high_layer_label = "High (1000-2000 m)"
    df_circ = df_circ[df_circ["Layer"] != high_layer_label]
    df_vec = df_vec[df_vec["Layer"] != high_layer_label]

    # Ensure Category order
    cat_order = ["All Times", "Daytime", "Nighttime"]
    for df in [df_circ, df_vec, df_veer]:
        df["Category"] = pd.Categorical(df["Category"], categories=cat_order, ordered=True)
    
    df_circ = df_circ.sort_values(["Category", "Layer"])
    df_vec = df_vec.sort_values(["Category", "Layer"])
    df_veer = df_veer.sort_values(["Category"])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot layered metrics (Circular & Vector)
    fig_layered = build_layered_metrics_figure(df_circ, df_vec)
    layered_path_png = args.out_dir / "wind_direction_metrics_layered_panel.png"
    layered_path_pdf = args.out_dir / "wind_direction_metrics_layered_panel.pdf"
    fig_layered.savefig(layered_path_png)
    fig_layered.savefig(layered_path_pdf)
    print(f"Saved {layered_path_png}")
    print(f"Saved {layered_path_pdf}")
    plt.close(fig_layered)

    # 2. Plot veer metrics
    fig_veer = build_veer_metrics_figure(df_veer)
    veer_path_png = args.out_dir / "wind_direction_metrics_veer_panel.png"
    veer_path_pdf = args.out_dir / "wind_direction_metrics_veer_panel.pdf"
    fig_veer.savefig(veer_path_png)
    fig_veer.savefig(veer_path_pdf)
    print(f"Saved {veer_path_png}")
    print(f"Saved {veer_path_pdf}")
    plt.close(fig_veer)


if __name__ == "__main__":
    main()
