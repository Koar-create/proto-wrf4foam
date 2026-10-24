"""Plot subset wind-speed metrics (summary bars and daily lines) from ws_subset_metrics.csv."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_ANALYSIS = Path(__file__).resolve().parents[2]
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import batch
from metrics.style import (
    COLOR_CFD,
    COLOR_LOW,
    COLOR_LOW_NIGHT,
    COLOR_MID,
    COLOR_MID_NIGHT,
    COLOR_WRF,
    configure_matplotlib_style,
)

DATA_PATH = batch.SUBSET_METRICS_CSV
OUTPUT_DIR = batch.FIGURES_DIR
SUMMARY_CATEGORIES = batch.SUMMARY_CATEGORIES
DAILY_PANEL_START = batch.DAILY_PANEL_START
DAILY_PANEL_END = batch.DAILY_PANEL_END

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    ylabel: str
    filename: str
    zero_line: bool = False
    ylim: tuple[float, float] | None = None


METRICS = [
    MetricSpec("rmse", "Root Mean Square Error", r"m s$^{-1}$", "rmse"),
    MetricSpec("mbe", "Mean Bias Error", r"m s$^{-1}$", "mbe", zero_line=True),
    MetricSpec("ioa", "Index of Agreement", "IoA", "ioa", ylim=(0.0, 1.0)),
    MetricSpec("ss", "Skill Score", "Skill Score", "ss", zero_line=True, ylim=(-0.9, 0.9)),
]


def load_metrics(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    return df


def summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[df["time_category"].isin(SUMMARY_CATEGORIES)].copy()
    out["time_category"] = pd.Categorical(out["time_category"], SUMMARY_CATEGORIES, ordered=True)
    return out.sort_values(["time_category", "layer"])


def daily_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[
        (df["sample_group"] == "lst_day_period")
        & (df["sample_date"] >= DAILY_PANEL_START)
        & (df["sample_date"] <= DAILY_PANEL_END)
    ].copy()
    return out.sort_values(["sample_date", "period", "layer"])


def daily_panel_days() -> list[pd.Timestamp]:
    return [pd.Timestamp(d) for d in pd.date_range(DAILY_PANEL_START, DAILY_PANEL_END, freq="D")]


def _period_metric(sub: pd.DataFrame, day: pd.Timestamp, period: str, column: str) -> float | None:
    day = pd.Timestamp(day).normalize()
    try:
        return float(sub.loc[(day, period), column])
    except KeyError:
        return None


def model_metric_value(row: pd.Series, model: str, metric: str) -> float:
    if metric == "ss":
        return 0.0 if model == "wrf" else row["skill_score"]
    return row[f"{model}_{metric}"]


def daily_improvement_value(row: pd.Series, metric: str) -> float:
    if metric == "rmse":
        return row["wrf_rmse"] - row["cfd_rmse"]
    if metric == "mbe":
        return abs(row["wrf_mbe"]) - abs(row["cfd_mbe"])
    if metric == "ioa":
        return row["cfd_ioa"] - row["wrf_ioa"]
    if metric == "ss":
        return row["skill_score"]
    raise ValueError(f"Unsupported metric: {metric}")


def daily_improvement_ylabel(spec: MetricSpec) -> str:
    labels = {
        "rmse": r"$\Delta$RMSE (WRF - WRF-OpenFOAM, m s$^{-1}$)",
        "mbe": r"$\Delta$|MBE| (WRF - WRF-OpenFOAM, m s$^{-1}$)",
        "ioa": r"$\Delta$IoA (WRF-OpenFOAM - WRF)",
        "ss": "Skill score",
    }
    return labels[spec.key]


def day_labels(days: list[pd.Timestamp]) -> list[str]:
    return [f"Sep {d.day}" for d in days]


def annotate_axis_ylabel(ax: plt.Axes, ylabel: str) -> None:
    ax.text(
        0.98,
        0.04,
        ylabel,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12.0,
        color="0.25",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )


def set_symmetric_delta_ylim(ax: plt.Axes) -> None:
    bottom, top = ax.get_ylim()
    half_range = max(abs(bottom), abs(top), 0.05)
    ax.set_ylim(-half_range * 1.08, half_range * 1.08)


def plot_summary_bars(ax: plt.Axes, df: pd.DataFrame, spec: MetricSpec) -> None:
    categories = [c for c in SUMMARY_CATEGORIES if c in set(df["time_category"].astype(str))]
    layer_offsets = {"Low": -0.18, "Mid": 0.18}
    model_offsets = {"wrf": -0.055, "cfd": 0.055}
    width = 0.10

    for x, category in enumerate(categories):
        sub = df[df["time_category"] == category]
        for layer, layer_offset in layer_offsets.items():
            row = sub[sub["layer"] == layer].iloc[0]
            for model, model_offset in model_offsets.items():
                value = model_metric_value(row, model, spec.key)
                color = COLOR_WRF if model == "wrf" else COLOR_CFD
                alpha = 0.98 if layer == "Low" else 0.52
                ax.bar(
                    x + layer_offset + model_offset,
                    value,
                    width=width,
                    color=color,
                    alpha=alpha,
                    edgecolor="white",
                    linewidth=0.6,
                )

    if spec.zero_line:
        ax.axhline(0, color="0.35", lw=0.8)
    if spec.ylim is not None:
        ax.set_ylim(*spec.ylim)
    ax.set_xticks(range(len(categories)), categories)
    ax.set_ylabel(spec.ylabel)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def plot_daily_dumbbell_lines(
    ax: plt.Axes, df: pd.DataFrame, spec: MetricSpec, *, connect_markers: bool = True
) -> None:
    days = daily_panel_days()
    x = np.arange(len(days))
    offsets = {
        ("wrf", "Low"): -0.18,
        ("cfd", "Low"): -0.06,
        ("wrf", "Mid"): 0.06,
        ("cfd", "Mid"): 0.18,
    }
    model_colors = {"wrf": COLOR_WRF, "cfd": COLOR_CFD}
    layer_alpha = {"Low": 0.98, "Mid": 0.48}

    for model in ["wrf", "cfd"]:
        metric_col = f"{model}_{spec.key}"
        for layer in ["Low", "Mid"]:
            sub = df[df["layer"] == layer].set_index(["sample_date", "period"])
            for i, day in enumerate(days):
                xpos = x[i] + offsets[(model, layer)]
                daytime = _period_metric(sub, day, "daytime", metric_col)
                nighttime = _period_metric(sub, day, "nighttime", metric_col)
                if daytime is None and nighttime is None:
                    continue
                color = model_colors[model]
                alpha = layer_alpha[layer]
                if connect_markers and daytime is not None and nighttime is not None:
                    ax.plot(
                        [xpos, xpos],
                        [daytime, nighttime],
                        color=color,
                        alpha=alpha,
                        lw=1.2,
                        solid_capstyle="round",
                    )
                if daytime is not None:
                    ax.scatter(
                        xpos,
                        daytime,
                        marker="o",
                        s=28,
                        color=color,
                        alpha=alpha,
                        edgecolor="white",
                        linewidth=0.4,
                        zorder=3,
                    )
                if nighttime is not None:
                    ax.scatter(
                        xpos,
                        nighttime,
                        marker="^",
                        s=34,
                        facecolor="white",
                        edgecolor=color,
                        alpha=alpha,
                        linewidth=1.0,
                        zorder=3,
                    )

    legend_handles = [
        Line2D([0], [0], color=COLOR_WRF, lw=1.6, label="WRF"),
        Line2D([0], [0], color=COLOR_CFD, lw=1.6, label="WRF-OpenFOAM"),
        Line2D([0], [0], color="0.25", lw=1.6, alpha=0.98, label="Low"),
        Line2D([0], [0], color="0.25", lw=1.6, alpha=0.48, label="Mid"),
        Line2D([0], [0], color="0.25", marker="o", lw=0, label="daytime"),
        Line2D([0], [0], color="0.25", marker="^", markerfacecolor="white", lw=0, label="nighttime"),
    ]

    if spec.zero_line:
        ax.axhline(0, color="0.35", lw=0.8)
    if spec.ylim is not None:
        ax.set_ylim(*spec.ylim)
    ax.set_xticks(x, day_labels(days))
    ax.set_ylabel(spec.ylabel)
    annotate_axis_ylabel(ax, spec.ylabel)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncols=3,
        frameon=False,
        handlelength=1.5,
        fontsize=8.0,
    )


def plot_daily_delta_lines(
    ax: plt.Axes, df: pd.DataFrame, spec: MetricSpec, *, show_dumbbell_connectors: bool = True
) -> None:
    if spec.key in {"rmse", "mbe"}:
        plot_daily_dumbbell_lines(ax, df, spec, connect_markers=show_dumbbell_connectors)
        return

    days = daily_panel_days()
    x = np.arange(len(days))

    styles = {
        ("Low", "daytime"): (COLOR_LOW, "o", "-"),
        ("Low", "nighttime"): (COLOR_LOW_NIGHT, "^", "--"),
        ("Mid", "daytime"): (COLOR_MID, "o", "-"),
        ("Mid", "nighttime"): (COLOR_MID_NIGHT, "^", "--"),
    }
    labels = {
        ("Low", "daytime"): "Low - day",
        ("Low", "nighttime"): "Low - night",
        ("Mid", "daytime"): "Mid - day",
        ("Mid", "nighttime"): "Mid - night",
    }
    for (layer, period), (color, marker, linestyle) in styles.items():
        sub = df[(df["layer"] == layer) & (df["period"] == period)].set_index("sample_date")
        y = [
            daily_improvement_value(sub.loc[day], spec.key) if day in sub.index else np.nan
            for day in days
        ]
        ax.plot(x, y, marker=marker, linestyle=linestyle, color=color, lw=1.5, ms=5, label=labels[(layer, period)])

    ax.axhline(0, color="0.35", lw=0.8)
    if spec.key == "ss" and spec.ylim is not None:
        ax.set_ylim(*spec.ylim)
    else:
        set_symmetric_delta_ylim(ax)
    ax.set_xticks(x, day_labels(days))
    ylabel = daily_improvement_ylabel(spec)
    ax.set_ylabel(ylabel)
    annotate_axis_ylabel(ax, ylabel)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncols=2, frameon=False, handlelength=1.6, fontsize=8.5)


def plot_daily_raw_lines(ax: plt.Axes, df: pd.DataFrame, spec: MetricSpec) -> None:
    if spec.key == "ss":
        plot_daily_delta_lines(ax, df, spec)
        return

    days = daily_panel_days()
    x = np.arange(len(days))
    color_map = {
        ("wrf", "Low"): COLOR_WRF,
        ("cfd", "Low"): COLOR_CFD,
        ("wrf", "Mid"): COLOR_WRF,
        ("cfd", "Mid"): COLOR_CFD,
    }
    alpha_map = {"Low": 0.98, "Mid": 0.52}
    linestyle_map = {"daytime": "-", "nighttime": "--"}
    marker_map = {"Low": "o", "Mid": "s"}

    for model in ["wrf", "cfd"]:
        for layer in ["Low", "Mid"]:
            for period in ["daytime", "nighttime"]:
                sub = df[(df["layer"] == layer) & (df["period"] == period)].set_index("sample_date")
                metric_col = f"{model}_{spec.key}"
                y = [float(sub.loc[day, metric_col]) if day in sub.index else np.nan for day in days]
                ax.plot(
                    x,
                    y,
                    marker=marker_map[layer],
                    linestyle=linestyle_map[period],
                    color=color_map[(model, layer)],
                    alpha=alpha_map[layer],
                    lw=1.4,
                    ms=4.5,
                )

    legend_handles = [
        Line2D([0], [0], color=COLOR_WRF, lw=1.6, alpha=0.98, marker="o", label="WRF low"),
        Line2D([0], [0], color=COLOR_CFD, lw=1.6, alpha=0.98, marker="o", label="WRF-OpenFOAM low"),
        Line2D([0], [0], color=COLOR_WRF, lw=1.6, alpha=0.52, marker="s", label="WRF mid"),
        Line2D([0], [0], color=COLOR_CFD, lw=1.6, alpha=0.52, marker="s", label="WRF-OpenFOAM mid"),
        Line2D([0], [0], color="0.25", lw=1.4, ls="-", label="daytime"),
        Line2D([0], [0], color="0.25", lw=1.4, ls="--", label="nighttime"),
    ]

    if spec.zero_line:
        ax.axhline(0, color="0.35", lw=0.8)
    if spec.ylim is not None:
        ax.set_ylim(*spec.ylim)
    ax.set_xticks(x, day_labels(days))
    ax.set_ylabel(spec.ylabel)
    annotate_axis_ylabel(ax, spec.ylabel)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncols=3,
        frameon=False,
        handlelength=1.6,
        fontsize=8.0,
    )


def plot_daily_lines(
    ax: plt.Axes,
    df: pd.DataFrame,
    spec: MetricSpec,
    line_mode: str = "delta",
    *,
    show_dumbbell_connectors: bool = True,
) -> None:
    if line_mode == "delta":
        plot_daily_delta_lines(ax, df, spec, show_dumbbell_connectors=show_dumbbell_connectors)
    elif line_mode == "raw":
        plot_daily_raw_lines(ax, df, spec)
    else:
        raise ValueError(f"Unsupported line mode: {line_mode}")


def _remove_axis_legend(ax: plt.Axes) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def _bar_legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=COLOR_WRF, alpha=0.98, label="WRF low (52-300 m)"),
        Patch(facecolor=COLOR_CFD, alpha=0.98, label="WRF-OpenFOAM low"),
        Patch(facecolor=COLOR_WRF, alpha=0.52, label="WRF mid (300-1000 m)"),
        Patch(facecolor=COLOR_CFD, alpha=0.52, label="WRF-OpenFOAM mid"),
    ]


def _daily_dumbbell_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=COLOR_WRF, lw=1.6, label="WRF"),
        Line2D([0], [0], color=COLOR_CFD, lw=1.6, label="WRF-OpenFOAM"),
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
        Line2D([0], [0], color=COLOR_CFD, lw=1.6, alpha=0.98, marker="o", label="WRF-OpenFOAM low"),
        Line2D([0], [0], color=COLOR_WRF, lw=1.6, alpha=0.52, marker="s", label="WRF mid"),
        Line2D([0], [0], color=COLOR_CFD, lw=1.6, alpha=0.52, marker="s", label="WRF-OpenFOAM mid"),
        Line2D([0], [0], color="0.25", lw=1.4, ls="-", label="daytime"),
        Line2D([0], [0], color="0.25", lw=1.4, ls="--", label="nighttime"),
    ]


def build_bar_panel_figure(df) -> plt.Figure:
    summary = summary_rows(df)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.09, hspace=0.34, wspace=0.22)

    for ax, spec, panel_label in zip(axes.ravel(), METRICS, PANEL_LABELS):
        plot_summary_bars(ax, summary, spec)
        ax.set_title(f"{panel_label} {spec.title}")

    fig.legend(
        handles=_bar_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncols=4,
        frameon=False,
        handlelength=0.8,
        columnspacing=1.4,
        fontsize=13.0,
    )
    return fig


def build_daily_panel_figure(df, line_mode: str = "delta") -> plt.Figure:
    daily = daily_rows(df)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.5), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.09, hspace=0.48, wspace=0.22)

    for ax, spec, panel_label in zip(axes.ravel(), METRICS, PANEL_LABELS):
        plot_daily_lines(
            ax,
            daily,
            spec,
            line_mode=line_mode,
            show_dumbbell_connectors=line_mode != "delta",
        )
        _remove_axis_legend(ax)
        if line_mode == "delta" and spec.key in {"ioa", "ss"}:
            _shade_negative_y(ax)
        ax.set_title(f"{panel_label} {spec.title}")

    if line_mode == "delta":
        fig.legend(
            handles=_daily_dumbbell_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncols=6,
            frameon=False,
            handlelength=1.5,
            columnspacing=1.1,
            fontsize=13.0,
        )
        fig.legend(
            handles=_daily_delta_line_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.46),
            ncols=4,
            frameon=False,
            handlelength=1.5,
            columnspacing=1.4,
            fontsize=13.0,
        )
        return fig
    else:
        legend_handles = _daily_raw_legend_handles()
        ncols = 6

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncols=ncols,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.1,
        fontsize=13.0,
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot subset wind-speed metrics: summary bars and daily lines."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DATA_PATH,
        help="Subset-metric CSV from compute_subset_metrics.py.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Figure formats to write.",
    )
    parser.add_argument(
        "--line-mode",
        choices=["delta", "raw", "both"],
        default="both",
        help="Daily line-panel mode: delta improvement, raw WRF/WRF-OpenFOAM values, or both.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.csv.exists():
        raise FileNotFoundError(
            f"Subset-metric CSV not found: {args.csv}\n"
            "Run `python analysis/260409/metrics/compute_subset_metrics.py` first."
        )
    df = load_metrics(args.csv)
    figures = {"ws_subset_bars": build_bar_panel_figure(df)}
    line_modes = ["delta", "raw"] if args.line_mode == "both" else [args.line_mode]
    for line_mode in line_modes:
        figures[f"ws_subset_daily_{line_mode}"] = build_daily_panel_figure(df, line_mode=line_mode)
    for stem, fig in figures.items():
        for ext in args.formats:
            out_path = args.out_dir / f"{stem}.{ext}"
            fig.savefig(out_path)
            print(f"Wrote {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
