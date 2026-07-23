#!/usr/bin/env python3
"""Summarize turbulence statistics across all enhanced hourly states."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.synthetic_wind.config import (  # noqa: E402
    OUTPUT_ROOT,
    STATS_SUBDIR,
    enhanced_hdf5_path,
    list_case_ids,
)
from analysis.synthetic_wind.grid import load_enhanced_hdf5  # noqa: E402
from analysis.synthetic_wind.mean_field import case_id_to_datetime  # noqa: E402
from analysis.synthetic_wind.statistics import (  # noqa: E402
    classify_stability,
    integral_scales,
    summarize_case_stats,
    turbulence_intensity,
)


def build_summary_table() -> pd.DataFrame:
    rows = []
    for cid in list_case_ids():
        path = enhanced_hdf5_path(cid)
        if not path.is_file():
            continue
        data = load_enhanced_hdf5(cid)
        k, eps, U = data["k"], data["epsilon"], data["U"]
        z = data["coords_z"]
        L, T = integral_scales(k, eps)
        summary = summarize_case_stats(k, eps, U, z)
        ti = float(np.nanmean(turbulence_intensity(data["R"], U)))
        dt = case_id_to_datetime(cid)
        rows.append({
            "case_id": cid,
            "datetime": dt.isoformat(),
            "hour_utc": dt.hour,
            "stability": classify_stability(summary),
            "llj": summary["llj"],
            "k_aloft": summary["k_aloft"],
            "k_max": summary["k_max"],
            "L_median": float(np.nanmedian(L)),
            "T_median": float(np.nanmedian(T)),
            "TI_mean": ti,
        })
    return pd.DataFrame(rows)


def plot_daily(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["hour_utc"] = df["hour_utc"].astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col, ylab in zip(
        axes.ravel(),
        ["TI_mean", "k_aloft", "L_median", "T_median"],
        ["Mean TI (-)", "k aloft (m²/s²)", "L median (m)", "T median (s)"],
    ):
        for stab, sub in df.groupby("stability"):
            ax.plot(sub["hour_utc"], sub[col], "o", alpha=0.5, label=stab, markersize=3)
        ax.set_xlabel("Hour (UTC)")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("120-hour RANS turbulence statistics (diurnal)")
    fig.tight_layout()
    fig.savefig(out_dir / "daily_turbulence_stats.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    df = build_summary_table()
    out_dir = OUTPUT_ROOT / STATS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "turbulence_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} cases)")
    if args.plot and len(df) > 0:
        plot_daily(df, out_dir)


if __name__ == "__main__":
    main()
