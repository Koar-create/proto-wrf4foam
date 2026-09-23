#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import re
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]

# Color specifications from metrics/visualize_metric_sample_variants_four_metrics.py
COLOR_WRF = "#e07b39"
COLOR_CFD = "#2196a5"

# Same finished-case rule as scripts/plot_experiment_hourly_status.py:
# a UTC hour has data only when <YYYYMMDD_HHMM>_two_boundaries_as_outlet/5000 exists.
CASES_ROOT = REPO_ROOT / "steady_experiments_finer_ABL"
CASE_RE = re.compile(r"^(\d{8})_(\d{4})_two_boundaries_as_outlet$")
COMPLETION_MARKER = "5000"
LATER_MERGED_CSV = REPO_ROOT / "data/260707/processed/merged_lidar_simulation_final.csv"
X_RIGHT = (2025, 9, 9, 23)


def configure_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
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


def _fmt_height_tag(h: float) -> str:
    if float(h).is_integer():
        return f"{int(h)}m"
    s = f"{h:.1f}".rstrip("0").rstrip(".")
    return f"{s}m"


def _pick_nearest_height(df_ob: pd.DataFrame, h_req: float) -> float:
    heights = pd.to_numeric(df_ob["Height"], errors="coerce").dropna().unique()
    if len(heights) == 0:
        raise ValueError("No valid Height values for this obtid.")
    heights = np.asarray(heights, dtype=float)
    idx = int(np.argmin(np.abs(heights - float(h_req))))
    return float(heights[idx])


def _ensure_ws_cfd(df: pd.DataFrame) -> None:
    if "ws_cfd" in df.columns:
        return
    if {"u_cfd", "v_cfd"}.issubset(df.columns):
        u = pd.to_numeric(df["u_cfd"], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(df["v_cfd"], errors="coerce").to_numpy(dtype=float)
        df["ws_cfd"] = np.sqrt(u * u + v * v)
        return
    raise ValueError("CSV missing ws_cfd and u_cfd/v_cfd; cannot compute CFD wind speed.")


def _pad_ymax_for_metrics(ymax: float) -> float:
    """Bump ymax to the next even integer (e.g. 6.x -> 8) for metric headroom."""
    nice = float(math.ceil(float(ymax) / 2.0) * 2)
    if nice <= float(ymax) + 1e-6:
        nice += 2.0
    return nice


def finished_utc_timestamps(cases_root: Path) -> set[pd.Timestamp]:
    """UTC hours whose production case has a ``5000`` directory."""
    if not cases_root.is_dir():
        raise FileNotFoundError(f"cases root not found: {cases_root}")
    finished: set[pd.Timestamp] = set()
    for path in cases_root.iterdir():
        if not path.is_dir():
            continue
        match = CASE_RE.match(path.name)
        if match is None or not (path / COMPLETION_MARKER).is_dir():
            continue
        day, hhmm = match.group(1), match.group(2)
        finished.add(
            pd.Timestamp(
                f"{day[:4]}-{day[4:6]}-{day[6:8]} {hhmm[:2]}:{hhmm[2:]}:00",
                tz="UTC",
            )
        )
    return finished


def _read_merged(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.floor("h")
    return df


def load_merged_frames(primary: Path, later: Path | None) -> pd.DataFrame:
    """Stack merged CSVs. Duplicate keys keep the primary file."""
    frames = [_read_merged(primary)]
    if later is not None and later.resolve() != primary.resolve():
        if later.exists():
            frames.append(_read_merged(later))
        else:
            print(f"Warning: later merged CSV not found, skipped: {later}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["datetime", "obtid", "Height"], keep="first")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot 3-station 2-height time series (LiDAR vs WRF vs OpenFOAM) in a 3x2 layout."
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv",
        help="Primary merged CSV (wind-speed values). Duplicate keys win over --csv-later.",
    )
    ap.add_argument(
        "--csv-later",
        type=Path,
        default=LATER_MERGED_CSV,
        help="Merged CSV for hours after the primary file (same columns). "
        "Values still come only from these merged tables.",
    )
    ap.add_argument(
        "--cases-root",
        type=Path,
        default=CASES_ROOT,
        help="OpenFOAM case root. A timestamp is drawn only if <case>/5000 exists.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results/timeseries_3x2/260409",
        help="Output directory.",
    )
    ap.add_argument(
        "--3h-rolling",
        dest="rolling_3h",
        action="store_true",
        help="Apply 3-hour rolling mean (time-based rolling window).",
    )
    ap.add_argument(
        "--tz",
        choices=["utc", "lst"],
        default="lst",
        help="Timezone for x-axis: utc (default) or lst (UTC+8 local standard time).",
    )
    ap.add_argument(
        "--show-metrics",
        action="store_true",
        help="Calculate and display R, MBE, and RMSE metrics on the plot (default: off).",
    )
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    configure_matplotlib_style()

    csv_path: Path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = load_merged_frames(csv_path, args.csv_later)
    _ensure_ws_cfd(df)

    finished = finished_utc_timestamps(args.cases_root)
    in_csv = set(df["datetime"].unique())
    kept = in_csv & finished
    df = df[df["datetime"].isin(kept)].copy()
    x_right_utc = pd.Timestamp(year=X_RIGHT[0], month=X_RIGHT[1], day=X_RIGHT[2], hour=X_RIGHT[3], tz="UTC")
    missing = sorted(t for t in finished if t <= x_right_utc and t not in in_csv and t >= pd.Timestamp("2025-09-01", tz="UTC"))
    print(
        f"[Info] merged hours drawn: {len(kept)}  |  "
        f"finished cases through {x_right_utc.strftime('%Y-%m-%d %H:%M')} UTC missing from merged CSV: {len(missing)}"
    )
    for t in missing:
        print(f"       no merged row: {t.strftime('%Y-%m-%d %H:%M')} UTC")

    tz_mode = str(args.tz).lower()
    rolling_3h = bool(args.rolling_3h)

    sites = ["GAW103", "GAW104", "GAW111"]
    height_reqs = [120.0, 500.0]

    if tz_mode == "lst":
        tzinfo = timezone(timedelta(hours=8))
        x_label = "Time (UTC+8)"
    else:
        tzinfo = timezone.utc
        x_label = "Time (UTC)"

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(17, 10.5), sharex=True, sharey="row")
    fig.subplots_adjust(top=0.91, bottom=0.08, left=0.06, right=0.98, hspace=0.28, wspace=0.12)

    for i, site in enumerate(sites):
        df_site = df[df["obtid"].astype(str) == site].copy()
        if len(df_site) == 0:
            print(f"Warning: No rows for {site}, skipping.")
            continue
            
        for j, h_req in enumerate(height_reqs):
            ax = axes[i, j]
            panel_label = f"({chr(ord('a') + i * len(height_reqs) + j)})"
            try:
                h_used = _pick_nearest_height(df_site, h_req)
            except ValueError:
                print(f"Warning: No valid height for {site} near {h_req}m.")
                continue
                
            df_plot = df_site[np.isclose(pd.to_numeric(df_site["Height"], errors="coerce"), h_used)].copy()
            df_plot = df_plot.sort_values("datetime").set_index("datetime")
            
            if rolling_3h:
                for c in ["ws_obs", "ws_wrf", "ws_cfd"]:
                    df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce").rolling("3h", center=True, min_periods=1).mean()
            else:
                for c in ["ws_obs", "ws_wrf", "ws_cfd"]:
                    df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")
                    
            if tz_mode == "lst":
                df_plot = df_plot.tz_convert(tzinfo)

            ax.plot(
                df_plot.index,
                df_plot["ws_obs"],
                linestyle="-",
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                color="black",
                alpha=0.85,
            )
            ax.plot(
                df_plot.index,
                df_plot["ws_wrf"],
                linestyle="--",
                linewidth=2.0,
                color=COLOR_WRF,
                alpha=0.95,
            )
            ax.plot(
                df_plot.index,
                df_plot["ws_cfd"],
                linestyle="-",
                linewidth=2.0,
                color=COLOR_CFD,
                alpha=0.95,
            )

            # Calculate and annotate metrics
            if args.show_metrics:
                valid_wrf = df_plot[["ws_obs", "ws_wrf"]].dropna()
                if len(valid_wrf) > 1:
                    r_w = valid_wrf["ws_obs"].corr(valid_wrf["ws_wrf"])
                    d_w = valid_wrf["ws_wrf"] - valid_wrf["ws_obs"]
                    mbe_w = d_w.mean()
                    rmse_w = np.sqrt((d_w ** 2).mean())
                    wrf_text = f"WRF | R: {r_w:.2f}  MBE: {mbe_w:.2f}  RMSE: {rmse_w:.2f}"
                else:
                    wrf_text = "WRF | N/A"

                valid_cfd = df_plot[["ws_obs", "ws_cfd"]].dropna()
                if len(valid_cfd) > 1:
                    r_c = valid_cfd["ws_obs"].corr(valid_cfd["ws_cfd"])
                    d_c = valid_cfd["ws_cfd"] - valid_cfd["ws_obs"]
                    mbe_c = d_c.mean()
                    rmse_c = np.sqrt((d_c ** 2).mean())
                    cfd_text = f"W-OF | R: {r_c:.2f}  MBE: {mbe_c:.2f}  RMSE: {rmse_c:.2f}"
                else:
                    cfd_text = "W-OF | N/A"

                bbox_props = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=1.0)
                ax.text(0.015, 0.97, wrf_text, transform=ax.transAxes, color=COLOR_WRF, fontsize=11, fontweight='bold', va='top', ha='left', bbox=bbox_props, zorder=5)
                ax.text(0.015, 0.86, cfd_text, transform=ax.transAxes, color=COLOR_CFD, fontsize=11, fontweight='bold', va='top', ha='left', bbox=bbox_props, zorder=5)

            # Subplot titles and labels (panel id outside axes, in title)
            layer_tag = "(Low)" if int(h_req) == 120 else "(Mid)"
            ax.set_title(f"{panel_label} {site} | {int(h_req)} m {layer_tag}")
            
            # Y-axis label only on the left column
            if j == 0:
                ax.set_ylabel(r"Wind speed (m s$^{-1}$)")
                
            # X-axis logic (sharex is True, so ticklabels are only on the bottom row automatically)
            # Use DayLocator to tick at 00:00 each day
            ax.xaxis.set_major_locator(mdates.DayLocator(tz=tzinfo))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=tzinfo))
            
            if i == 2:
                ax.set_xlabel(x_label)

    # X-limits in the plot timezone. Right edge is 2025-09-09 23:00.
    x_left = pd.Timestamp(year=2025, month=9, day=1, hour=0, tz=tzinfo)
    x_right = pd.Timestamp(year=X_RIGHT[0], month=X_RIGHT[1], day=X_RIGHT[2], hour=X_RIGHT[3], tz=tzinfo)
    axes[0, 0].set_xlim(x_left, x_right)

    # Leave headroom at top for metric annotations (sharey='row').
    if args.show_metrics:
        for i in range(axes.shape[0]):
            ymax = max(float(ax.get_ylim()[1]) for ax in axes[i, :])
            ymin = min(float(ax.get_ylim()[0]) for ax in axes[i, :])
            axes[i, 0].set_ylim(ymin, _pad_ymax_for_metrics(ymax))

    # Common Legend
    legend_handles = [
        Line2D([0], [0], color="black", marker="o", markersize=7, lw=1.5, label="LiDAR Obs"),
        Line2D([0], [0], color=COLOR_WRF, lw=2.5, linestyle="--", label="WRF"),
        Line2D([0], [0], color=COLOR_CFD, lw=2.5, linestyle="-", label="WRF-OpenFOAM")
    ]
    fig.legend(
        handles=legend_handles, 
        loc="upper center", 
        bbox_to_anchor=(0.5, 0.98), 
        ncols=3, 
        frameon=False,
        handlelength=2.5,
        fontsize=17
    )
    
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    roll_tag = "roll3h" if rolling_3h else "raw"
    metrics_tag = "_metrics" if args.show_metrics else ""
    out_path = out_dir / f"timeseries_3x2_sites_heights_{roll_tag}_tz-{tz_mode}{metrics_tag}.png"
    
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved figure to: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
