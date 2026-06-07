from __future__ import annotations

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
    out["day_label"] = out["sample_date"].dt.strftime("Sep %-d")
    if out["day_label"].str.contains("%-d").any():
        out["day_label"] = out["sample_date"].dt.strftime("Sep %#d")
    return out.sort_values(["sample_date", "period", "layer"])


def plot_grouped_metric(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str) -> None:
    categories = ["All", "Daytime", "Nighttime"]
    layer_offsets = {"Low": -0.18, "Mid": 0.18}
    model_offsets = {"wrf": -0.055, "cfd": 0.055}
    width = 0.10

    for x, category in enumerate(categories):
        sub = df[df["time_category"] == category]
        for layer, layer_offset in layer_offsets.items():
            row = sub[sub["layer"] == layer].iloc[0]
            for model, model_offset in model_offsets.items():
                value = row[f"{model}_{metric}"]
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

    if metric == "mbe":
        ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(range(len(categories)), categories)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def plot_delta_ioa(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = []
    values = []
    for layer in ["Low", "Mid"]:
        for category in ["All", "Daytime", "Nighttime"]:
            row = df[(df["layer"] == layer) & (df["time_category"] == category)].iloc[0]
            labels.append(f"{layer} - {category}")
            values.append(row["cfd_ioa"] - row["wrf_ioa"])

    y = np.arange(len(labels))
    colors = [COLOR_CFD if v >= 0 else COLOR_WRF for v in values]
    ax.barh(y, values, color=colors, alpha=0.78, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="0.35", lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("IoA(CFD) - IoA(WRF)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Patch(facecolor=COLOR_CFD, alpha=0.78, label="CFD better"),
            Patch(facecolor=COLOR_WRF, alpha=0.78, label="WRF better"),
        ],
        loc="lower right",
        frameon=False,
        ncols=2,
        handlelength=1.0,
    )


def plot_daily_skill(ax: plt.Axes, df: pd.DataFrame) -> None:
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

    days = sorted(df["sample_date"].dropna().unique())
    x = np.arange(len(days))
    day_labels = [pd.Timestamp(d).strftime("Sep %#d") for d in days]

    for key, (color, marker, linestyle) in styles.items():
        layer, period = key
        sub = df[(df["layer"] == layer) & (df["period"] == period)].set_index("sample_date")
        y = [sub.loc[pd.Timestamp(d), "skill_score"] for d in days]
        ax.plot(
            x,
            y,
            marker=marker,
            linestyle=linestyle,
            color=color,
            lw=1.5,
            ms=5,
            label=labels[key],
        )

    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(x, day_labels)
    ax.set_ylabel("Skill score")
    ax.set_ylim(-0.9, 0.9)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    period_legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=2,
        frameon=False,
        handlelength=1.6,
        fontsize=8.5,
    )
    ax.add_artist(period_legend)


def build_figure(df: pd.DataFrame) -> plt.Figure:
    summary = summary_rows(df)
    daily = daily_rows(df)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.3), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.11, hspace=0.42, wspace=0.22)
    fig.suptitle(
        "WRF vs. OpenFOAM wind-speed evaluation, 1-5 September 2025",
        y=0.985,
        fontsize=13,
        fontweight="bold",
    )

    plot_grouped_metric(axes[0, 0], summary, "rmse", r"m s$^{-1}$")
    axes[0, 0].set_title("(a) Root mean square error")

    plot_grouped_metric(axes[0, 1], summary, "mbe", r"m s$^{-1}$")
    axes[0, 1].set_title("(b) Mean bias error")

    plot_delta_ioa(axes[1, 0], summary)
    axes[1, 0].set_title("(c) Index of agreement change")

    plot_daily_skill(axes[1, 1], daily)
    axes[1, 1].set_title("(d) Daily skill score")

    bar_legend = [
        Patch(facecolor=COLOR_WRF, alpha=0.98, label="WRF low (52-300 m)"),
        Patch(facecolor=COLOR_CFD, alpha=0.98, label="CFD low"),
        Patch(facecolor=COLOR_WRF, alpha=0.52, label="WRF mid (300-1000 m)"),
        Patch(facecolor=COLOR_CFD, alpha=0.52, label="CFD mid"),
    ]
    fig.legend(
        handles=bar_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncols=4,
        frameon=False,
        handlelength=0.8,
        columnspacing=1.4,
    )

    # note_handles = [
    #     Line2D([0], [0], color="0.25", ls="-", lw=1.2, label="solid = daytime"),
    #     Line2D([0], [0], color="0.25", ls="--", lw=1.2, label="dashed = nighttime"),
    # ]
    # axes[1, 1].add_artist(
    #     axes[1, 1].legend(
    #         handles=note_handles,
    #         loc="lower left",
    #         frameon=False,
    #         handlelength=1.8,
    #     )
    # )

    # fig.text(
    #     0.5,
    #     0.005,
    #     "UTC calendar-day and conservative subsets are omitted; Sep 6 is excluded because the LST day is incomplete.",
    #     ha="center",
    #     va="bottom",
    #     fontsize=8.5,
    #     color="0.35",
    # )
    return fig


def main() -> None:
    configure_matplotlib_style()
    df = load_metrics()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure(df)
    fig.savefig(OUTPUT_DIR / "metric_sample_variants_no_high_summary.png")
    fig.savefig(OUTPUT_DIR / "metric_sample_variants_no_high_summary.pdf")
    plt.close(fig)
    print(f"Wrote {OUTPUT_DIR / 'metric_sample_variants_no_high_summary.png'}")
    print(f"Wrote {OUTPUT_DIR / 'metric_sample_variants_no_high_summary.pdf'}")


if __name__ == "__main__":
    main()
