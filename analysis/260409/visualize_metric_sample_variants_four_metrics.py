from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "analysis/260409/metric_sample_variants_no_high_wide.csv"
OUTPUT_DIR = REPO_ROOT / "results/metric_sample_variants/260409"


COLOR_WRF = "#e07b39"
COLOR_CFD = "#2196a5"
COLOR_LOW = "#2c7fb8"
COLOR_MID = "#d95f5f"
COLOR_LOW_NIGHT = "#7fcdbb"
COLOR_MID_NIGHT = "#f0a48f"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    ylabel: str
    filename: str
    zero_line: bool = False
    ylim: tuple[float, float] | None = None


METRICS = [
    MetricSpec("rmse", "Root mean square error", r"m s$^{-1}$", "rmse"),
    MetricSpec("mbe", "Mean bias error", r"m s$^{-1}$", "mbe", zero_line=True),
    MetricSpec("ioa", "Index of agreement", "IoA", "ioa", ylim=(0.0, 1.0)),
    MetricSpec("ss", "Skill score", "Skill score", "ss", zero_line=True, ylim=(-0.9, 0.9)),
]


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


def load_metrics(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    return df


def summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    keep = (
        (df["sample_group"] == "baseline")
        | ((df["sample_group"] == "utc_hour_group") & df["period"].isin(["daytime", "nighttime"]))
    )
    out = df.loc[keep].copy()
    out["time_category"] = out["period"].map(
        {
            "all_times": "All",
            "daytime": "Daytime",
            "nighttime": "Nighttime",
        }
    )
    out["time_category"] = pd.Categorical(
        out["time_category"], categories=["All", "Daytime", "Nighttime"], ordered=True
    )
    return out.sort_values(["time_category", "layer"])


def daily_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[
        (df["sample_group"] == "lst_day_period")
        & (df["sample_date"] >= pd.Timestamp("2025-09-01"))
        & (df["sample_date"] <= pd.Timestamp("2025-09-05"))
    ].copy()
    return out.sort_values(["sample_date", "period", "layer"])


def model_metric_value(row: pd.Series, model: str, metric: str) -> float:
    if metric == "ss":
        return 0.0 if model == "wrf" else row["skill_score"]
    return row[f"{model}_{metric}"]


def daily_improvement_value(row: pd.Series, metric: str) -> float:
    """Positive values mean CFD performs better than WRF."""
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
        "rmse": r"$\Delta$RMSE (WRF - CFD, m s$^{-1}$)",
        "mbe": r"$\Delta$|MBE| (WRF - CFD, m s$^{-1}$)",
        "ioa": r"$\Delta$IoA (CFD - WRF)",
        "ss": "Skill score",
    }
    return labels[spec.key]


def plot_summary_bars(ax: plt.Axes, df: pd.DataFrame, spec: MetricSpec) -> None:
    categories = ["All", "Daytime", "Nighttime"]
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
        fontsize=8.0,
        color="0.25",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )


def set_symmetric_delta_ylim(ax: plt.Axes) -> None:
    bottom, top = ax.get_ylim()
    half_range = max(abs(bottom), abs(top), 0.05)
    ax.set_ylim(-half_range * 1.08, half_range * 1.08)


def plot_daily_dumbbell_lines(ax: plt.Axes, df: pd.DataFrame, spec: MetricSpec) -> None:
    days = [pd.Timestamp(d) for d in sorted(df["sample_date"].dropna().unique())]
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
        for layer in ["Low", "Mid"]:
            sub = df[df["layer"] == layer].set_index(["sample_date", "period"])
            for i, day in enumerate(days):
                xpos = x[i] + offsets[(model, layer)]
                daytime = sub.loc[(day, "daytime"), f"{model}_{spec.key}"]
                nighttime = sub.loc[(day, "nighttime"), f"{model}_{spec.key}"]
                color = model_colors[model]
                alpha = layer_alpha[layer]
                ax.plot(
                    [xpos, xpos],
                    [daytime, nighttime],
                    color=color,
                    alpha=alpha,
                    lw=1.2,
                    solid_capstyle="round",
                )
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
        Line2D([0], [0], color=COLOR_CFD, lw=1.6, label="CFD"),
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


def plot_daily_delta_lines(ax: plt.Axes, df: pd.DataFrame, spec: MetricSpec) -> None:
    if spec.key in {"rmse", "mbe"}:
        plot_daily_dumbbell_lines(ax, df, spec)
        return

    days = [pd.Timestamp(d) for d in sorted(df["sample_date"].dropna().unique())]
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
        y = [daily_improvement_value(sub.loc[day], spec.key) for day in days]
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

    days = [pd.Timestamp(d) for d in sorted(df["sample_date"].dropna().unique())]
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
                y = [sub.loc[day, f"{model}_{spec.key}"] for day in days]
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
        Line2D([0], [0], color=COLOR_CFD, lw=1.6, alpha=0.98, marker="o", label="CFD low"),
        Line2D([0], [0], color=COLOR_WRF, lw=1.6, alpha=0.52, marker="s", label="WRF mid"),
        Line2D([0], [0], color=COLOR_CFD, lw=1.6, alpha=0.52, marker="s", label="CFD mid"),
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


def plot_daily_lines(ax: plt.Axes, df: pd.DataFrame, spec: MetricSpec, line_mode: str = "delta") -> None:
    if line_mode == "delta":
        plot_daily_delta_lines(ax, df, spec)
    elif line_mode == "raw":
        plot_daily_raw_lines(ax, df, spec)
    else:
        raise ValueError(f"Unsupported line mode: {line_mode}")


def build_figure(df: pd.DataFrame, spec: MetricSpec, line_mode: str = "delta") -> plt.Figure:
    summary = summary_rows(df)
    daily = daily_rows(df)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.78, bottom=0.14, wspace=0.22)
    fig.suptitle(
        f"{spec.title}, 1-5 September 2025",
        y=0.98,
        fontsize=13,
        fontweight="bold",
    )

    plot_summary_bars(axes[0], summary, spec)
    axes[0].set_title("(a) All / daytime / nighttime")

    plot_daily_lines(axes[1], daily, spec, line_mode=line_mode)
    if line_mode == "delta" and spec.key in {"rmse", "mbe"}:
        daily_title = "Daily compact WRF and CFD values"
    elif line_mode == "delta":
        daily_title = "Daily CFD improvement"
    else:
        daily_title = "Daily WRF and CFD values"
    axes[1].set_title(f"(b) {daily_title}")

    if spec.key != "ss":
        bar_legend = [
            Patch(facecolor=COLOR_WRF, alpha=0.98, label="WRF low (52-300 m)"),
            Patch(facecolor=COLOR_CFD, alpha=0.98, label="CFD low"),
            Patch(facecolor=COLOR_WRF, alpha=0.52, label="WRF mid (300-1000 m)"),
            Patch(facecolor=COLOR_CFD, alpha=0.52, label="CFD mid"),
        ]
    else:
        bar_legend = [
            Patch(facecolor=COLOR_CFD, alpha=0.98, label="CFD low (52-300 m)"),
            Patch(facecolor=COLOR_CFD, alpha=0.52, label="CFD mid (300-1000 m)"),
        ]
    
    fig.legend(
        handles=bar_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncols=4,
        frameon=False,
        handlelength=0.8,
        columnspacing=1.4,
    )

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one two-panel figure per metric: summary bars and daily metric lines."
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
        help="Daily line-plot mode: delta improvement, raw WRF/CFD values, or both.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib_style()
    df = load_metrics(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    line_modes = ["delta", "raw"] if args.line_mode == "both" else [args.line_mode]
    for spec in METRICS:
        for line_mode in line_modes:
            fig = build_figure(df, spec, line_mode=line_mode)
            for ext in args.formats:
                out_path = args.out_dir / f"metric_sample_variants_no_high_{spec.filename}_{line_mode}_daily_summary.{ext}"
                fig.savefig(out_path)
                print(f"Wrote {out_path}")
            plt.close(fig)


if __name__ == "__main__":
    main()
