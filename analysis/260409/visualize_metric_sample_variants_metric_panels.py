from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from visualize_metric_sample_variants_four_metrics import (
    COLOR_CFD,
    COLOR_LOW,
    COLOR_LOW_NIGHT,
    COLOR_MID,
    COLOR_MID_NIGHT,
    COLOR_WRF,
    DATA_PATH,
    METRICS,
    OUTPUT_DIR,
    configure_matplotlib_style,
    daily_rows,
    load_metrics,
    plot_daily_lines,
    plot_summary_bars,
    summary_rows,
)


PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]


def _remove_axis_legend(ax: plt.Axes) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def _bar_legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=COLOR_WRF, alpha=0.98, label="WRF low (52-300 m)"),
        Patch(facecolor=COLOR_CFD, alpha=0.98, label="CFD low"),
        Patch(facecolor=COLOR_WRF, alpha=0.52, label="WRF mid (300-1000 m)"),
        Patch(facecolor=COLOR_CFD, alpha=0.52, label="CFD mid"),
    ]


def _daily_dumbbell_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=COLOR_WRF, lw=1.6, label="WRF"),
        Line2D([0], [0], color=COLOR_CFD, lw=1.6, label="CFD"),
        Line2D([0], [0], color="0.25", lw=1.6, alpha=0.98, label="Low"),
        Line2D([0], [0], color="0.25", lw=1.6, alpha=0.48, label="Mid"),
        Line2D([0], [0], color="0.25", marker="o", lw=0, label="daytime"),
        Line2D([0], [0], color="0.25", marker="^", markerfacecolor="white", lw=0, label="nighttime"),
    ]


def _daily_delta_line_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=COLOR_LOW, lw=1.6, marker="o", label="Low - day"),
        Line2D([0], [0], color=COLOR_LOW_NIGHT, lw=1.6, ls="--", marker="^", label="Low - night"),
        Line2D([0], [0], color=COLOR_MID, lw=1.6, marker="o", label="Mid - day"),
        Line2D([0], [0], color=COLOR_MID_NIGHT, lw=1.6, ls="--", marker="^", label="Mid - night"),
    ]


def _shade_negative_y(ax: plt.Axes) -> None:
    ymin, ymax = ax.get_ylim()
    if ymin < 0:
        ax.axhspan(ymin, min(0, ymax), facecolor="0.92", edgecolor="none", zorder=0)
        ax.set_ylim(ymin, ymax)


def _daily_raw_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=COLOR_WRF, lw=1.6, alpha=0.98, marker="o", label="WRF low"),
        Line2D([0], [0], color=COLOR_CFD, lw=1.6, alpha=0.98, marker="o", label="CFD low"),
        Line2D([0], [0], color=COLOR_WRF, lw=1.6, alpha=0.52, marker="s", label="WRF mid"),
        Line2D([0], [0], color=COLOR_CFD, lw=1.6, alpha=0.52, marker="s", label="CFD mid"),
        Line2D([0], [0], color="0.25", lw=1.4, ls="-", label="daytime"),
        Line2D([0], [0], color="0.25", lw=1.4, ls="--", label="nighttime"),
    ]


def build_bar_panel_figure(df) -> plt.Figure:
    summary = summary_rows(df)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.09, hspace=0.34, wspace=0.22)
    fig.suptitle(
        "WRF vs. OpenFOAM metric summary bars, 1-5 September 2025",
        y=0.985,
        fontsize=13,
        fontweight="bold",
    )

    for ax, spec, panel_label in zip(axes.ravel(), METRICS, PANEL_LABELS):
        plot_summary_bars(ax, summary, spec)
        ax.set_title(f"{panel_label} {spec.title}")

    fig.legend(
        handles=_bar_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncols=4,
        frameon=False,
        handlelength=0.8,
        columnspacing=1.4,
    )
    return fig


def build_daily_panel_figure(df, line_mode: str = "delta") -> plt.Figure:
    daily = daily_rows(df)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.5), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.09, hspace=0.48, wspace=0.22)
    suptitle = (
        "Daily compact comparison and CFD improvement, 1-5 September 2025"
        if line_mode == "delta"
        else "Daily WRF and OpenFOAM metric values, 1-5 September 2025"
    )
    fig.suptitle(
        suptitle,
        y=0.985,
        fontsize=13,
        fontweight="bold",
    )

    for ax, spec, panel_label in zip(axes.ravel(), METRICS, PANEL_LABELS):
        plot_daily_lines(ax, daily, spec, line_mode=line_mode)
        _remove_axis_legend(ax)
        if line_mode == "delta" and spec.key in {"ioa", "ss"}:
            _shade_negative_y(ax)
        ax.set_title(f"{panel_label} {spec.title}")

    if line_mode == "delta":
        fig.legend(
            handles=_daily_dumbbell_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.87),
            ncols=6,
            frameon=False,
            handlelength=1.5,
            columnspacing=1.1,
            fontsize=8.0,
        )
        fig.legend(
            handles=_daily_delta_line_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.43),
            ncols=4,
            frameon=False,
            handlelength=1.5,
            columnspacing=1.4,
            fontsize=8.0,
        )
        return fig
    else:
        legend_handles = _daily_raw_legend_handles()
        ncols = 6

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncols=ncols,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.1,
        fontsize=8.0,
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot two 2x2 metric panel figures: summary bars and daily lines."
    )
    parser.add_argument("--csv", type=Path, default=DATA_PATH, help="Input wide metric CSV.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Figure formats to write.",
    )
    parser.add_argument(
        "--line-mode",
        choices=["delta", "raw", "both"],
        default="both",
        help="Daily line-panel mode: delta improvement, raw WRF/CFD values, or both.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib_style()
    df = load_metrics(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    line_modes = ["delta", "raw"] if args.line_mode == "both" else [args.line_mode]
    figures = {"metric_sample_variants_no_high_metric_bars_panel": build_bar_panel_figure(df)}
    for line_mode in line_modes:
        figures[f"metric_sample_variants_no_high_metric_daily_{line_mode}_panel"] = build_daily_panel_figure(
            df, line_mode=line_mode
        )
    for stem, fig in figures.items():
        for ext in args.formats:
            out_path = args.out_dir / f"{stem}.{ext}"
            fig.savefig(out_path)
            print(f"Wrote {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
