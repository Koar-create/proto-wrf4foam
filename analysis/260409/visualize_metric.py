from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
LAYER_SUMMARY_CSV = ANALYSIS_DIR / "layer_metrics_summary.csv"
OUTPUT_DIR = REPO_ROOT / "results/metric/260409"
OUTPUT_PNG = OUTPUT_DIR / "layer_metrics_comparison_optimized.png"


def load_layer_metrics(csv_path: Path = LAYER_SUMMARY_CSV) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Metrics CSV not found: {csv_path}\n"
            "Run `python analysis/260409/print_metric.py` first."
        )
    return pd.read_csv(csv_path)


def main() -> None:
    df = load_layer_metrics()

    layers = df["Layer"].tolist()
    wrf_mbe = df["WRF_MBE"].tolist()
    cfd_mbe = df["CFD_MBE"].tolist()
    wrf_rmse = df["WRF_RMSE"].tolist()
    cfd_rmse = df["CFD_RMSE"].tolist()
    wrf_ioa = df["WRF_IoA"].tolist()
    cfd_ioa = df["CFD_IoA"].tolist()
    ss = df["SS"].tolist()

    # 严谨的商务配色：深蓝色与深灰色（避开任何黄色/橙色系）
    color_wrf = "#1f77b4"
    color_cfd = "#4d4d4d"
    color_ss = "#2ca02c"

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    x = np.arange(len(layers))
    width = 0.3

    def add_labels(ax, rects, fontsize=11):
        for rect in rects:
            height = rect.get_height()
            if height >= 0:
                ax.annotate(
                    f"{height:+.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    fontweight="bold",
                )
            else:
                ax.annotate(
                    f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, -14),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    fontweight="bold",
                )

    # --- 子图 1: MBE ---
    rects1 = axes[0].bar(
        x - width / 2, wrf_mbe, width, label="WRF",
        color=color_wrf, edgecolor="black", linewidth=0.7,
    )
    rects2 = axes[0].bar(
        x + width / 2, cfd_mbe, width, label="CFD",
        color=color_cfd, edgecolor="black", linewidth=0.7,
    )
    axes[0].set_title("Mean Bias Error (MBE)", fontsize=15, fontweight="bold", pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layers, fontsize=13)
    axes[0].tick_params(axis="y", labelsize=12)
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
    mbe_lim = max(abs(min(wrf_mbe + cfd_mbe)), abs(max(wrf_mbe + cfd_mbe))) * 1.25
    axes[0].set_ylim(-mbe_lim, mbe_lim)
    add_labels(axes[0], rects1, fontsize=11)
    add_labels(axes[0], rects2, fontsize=11)
    axes[0].legend(fontsize=12, loc="upper right")

    # --- 子图 2: RMSE ---
    rects1 = axes[1].bar(
        x - width / 2, wrf_rmse, width, label="WRF",
        color=color_wrf, edgecolor="black", linewidth=0.7,
    )
    rects2 = axes[1].bar(
        x + width / 2, cfd_rmse, width, label="CFD",
        color=color_cfd, edgecolor="black", linewidth=0.7,
    )
    axes[1].set_title("Root Mean Squared Error (RMSE)", fontsize=15, fontweight="bold", pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layers, fontsize=13)
    axes[1].tick_params(axis="y", labelsize=12)
    axes[1].set_ylim(0, max(wrf_rmse + cfd_rmse) * 1.2)
    add_labels(axes[1], rects1, fontsize=11)
    add_labels(axes[1], rects2, fontsize=11)
    axes[1].legend(fontsize=12, loc="upper right")

    # --- 子图 3: IoA ---
    rects1 = axes[2].bar(
        x - width / 2, wrf_ioa, width, label="WRF",
        color=color_wrf, edgecolor="black", linewidth=0.7,
    )
    rects2 = axes[2].bar(
        x + width / 2, cfd_ioa, width, label="CFD",
        color=color_cfd, edgecolor="black", linewidth=0.7,
    )
    axes[2].set_title("Index of Agreement (IoA)", fontsize=15, fontweight="bold", pad=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(layers, fontsize=13)
    axes[2].tick_params(axis="y", labelsize=12)
    axes[2].set_ylim(0, 1.05)
    add_labels(axes[2], rects1, fontsize=11)
    add_labels(axes[2], rects2, fontsize=11)
    axes[2].legend(fontsize=12, loc="lower right")

    # --- 子图 4: SS ---
    rects1 = axes[3].bar(
        x, ss, width * 1.2, color=color_ss,
        edgecolor="black", linewidth=0.7, label="Skill Score",
    )
    axes[3].set_title("Skill Score (SS)", fontsize=15, fontweight="bold", pad=12)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(layers, fontsize=13)
    axes[3].tick_params(axis="y", labelsize=12)
    axes[3].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[3].set_ylim(0, max(ss) * 1.25 if ss else 0.2)
    add_labels(axes[3], rects1, fontsize=11)
    axes[3].legend(fontsize=12, loc="upper right")

    plt.suptitle(
        "Model Performance Metrics Comparison across Layers",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"[PNG] Saved to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
