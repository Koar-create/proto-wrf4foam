#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-time-slice validation figure: LiDAR vs WRF vs OpenFOAM (merged table).

Left:  obs vs CFD (colored by height) + obs vs WRF (grey), 1:1 line, ±20% band,
       text box with N, R², RMSE, MBE for CFD and WRF vs obs.
Right: Vertical wind-speed profiles (ws_obs, ws_wrf, ws_cfd) at one station.

Usage (from repo root):
  python util/plot_validation_scatter_profile.py \\
    --csv data/260409/processed/merged_lidar_simulation_final.csv \\
    --datetime "2025-09-03 14:00:00" \\
    --station GAW111 \\
    --out results/wrf_openfoam/snapshot_20250903_1400_validation.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LiDAR–WRF–CFD validation scatter + profile (one UTC time).")
    p.add_argument(
        "--csv",
        type=Path,
        default=_repo_root() / "data" / "260409" / "processed" / "merged_lidar_simulation_final.csv",
        help="Merged LiDAR + WRF + CFD CSV.",
    )
    p.add_argument(
        "--datetime",
        type=str,
        default=None,
        help='UTC timestamp string, e.g. "2025-09-03 14:00:00".',
    )
    p.add_argument("--date", type=str, default=None, help="Alternative: YYYY-MM-DD (use with --hour).")
    p.add_argument("--hour", type=int, default=None, help="UTC hour 0–23 (use with --date).")
    p.add_argument(
        "--station",
        type=str,
        default="GAW111",
        help="Station obtid for the right-hand profile panel (default GAW111).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_repo_root()
        / "results"
        / "wrf_openfoam"
        / "snapshot_20250903_1400_validation.png",
        help="Output PNG path.",
    )
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _resolve_time_str(args: argparse.Namespace) -> str:
    if args.datetime:
        return args.datetime.strip()
    if args.date is not None and args.hour is not None:
        return f"{args.date} {args.hour:02d}:00:00"
    raise SystemExit("Provide either --datetime or both --date and --hour.")


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def main() -> None:
    args = _parse_args()
    root = _repo_root()
    time_str = _resolve_time_str(args)
    csv_path = args.csv if args.csv.is_absolute() else (root / args.csv)
    out_path = args.out if args.out.is_absolute() else (root / args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns:
        print("ERROR: expected column 'datetime' in merged CSV.", file=sys.stderr)
        sys.exit(1)

    df["datetime"] = pd.to_datetime(df["datetime"])
    ts = pd.to_datetime(time_str)
    sub = df[df["datetime"] == ts].copy()
    if sub.empty:
        print(f"ERROR: no rows for datetime == {ts}", file=sys.stderr)
        sys.exit(1)

    # Horizontal wind speed from CFD components (plan)
    sub["ws_cfd"] = np.sqrt(sub["u_cfd"].astype(float) ** 2 + sub["v_cfd"].astype(float) ** 2)

    # Scatter datasets (four stations merged)
    m_cfd = sub["ws_obs"].notna() & sub["ws_cfd"].notna()
    m_wrf = sub["ws_obs"].notna() & sub["ws_wrf"].notna()

    obs_cfd = sub.loc[m_cfd, "ws_obs"].to_numpy(dtype=float)
    prd_cfd = sub.loc[m_cfd, "ws_cfd"].to_numpy(dtype=float)
    h_cfd = sub.loc[m_cfd, "Height"].to_numpy(dtype=float)

    obs_wrf = sub.loc[m_wrf, "ws_obs"].to_numpy(dtype=float)
    prd_wrf = sub.loc[m_wrf, "ws_wrf"].to_numpy(dtype=float)

    n_cfd = int(obs_cfd.size)
    n_wrf = int(obs_wrf.size)
    r2_cfd = _r2_score(obs_cfd, prd_cfd) if n_cfd >= 2 else float("nan")
    r2_wrf = _r2_score(obs_wrf, prd_wrf) if n_wrf >= 2 else float("nan")
    rmse_cfd = _rmse(obs_cfd, prd_cfd) if n_cfd else float("nan")
    rmse_wrf = _rmse(obs_wrf, prd_wrf) if n_wrf else float("nan")
    mbe_cfd = _mbe(obs_cfd, prd_cfd) if n_cfd else float("nan")
    mbe_wrf = _mbe(obs_wrf, prd_wrf) if n_wrf else float("nan")

    # Right panel: one station profile
    st = args.station.strip()
    prof = sub[sub["obtid"] == st].sort_values("Height")
    if prof.empty:
        print(f"ERROR: no rows for station {st} at {ts}", file=sys.stderr)
        sys.exit(1)

    lon0 = float(prof["lon"].iloc[0])
    lat0 = float(prof["lat"].iloc[0])

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "figure.dpi": 120,
        }
    )

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)

    # --- Left: scatter ---
    lim_hi = float(
        np.nanmax(
            np.concatenate(
                [
                    obs_cfd,
                    prd_cfd,
                    obs_wrf,
                    prd_wrf,
                ]
            )
        )
    )
    lim = max(0.5, lim_hi * 1.05)

    xx = np.linspace(0.0, lim, 300)
    axL.plot(xx, xx, color="0.35", ls="--", lw=1.2, label="1:1")
    axL.fill_between(xx, 0.8 * xx, 1.2 * xx, color="0.85", alpha=0.35, linewidth=0.0, label="±20%")

    axL.scatter(
        obs_wrf,
        prd_wrf,
        s=10,
        alpha=0.22,
        color="0.45",
        edgecolors="none",
        rasterized=True,
        label="WRF vs obs (ref.)",
    )
    sc_cfd = axL.scatter(
        obs_cfd,
        prd_cfd,
        c=h_cfd,
        cmap="viridis",
        vmin=0.0,
        vmax=2000.0,
        s=22,
        alpha=0.78,
        edgecolors="none",
        rasterized=True,
        label="CFD vs obs (color = height)",
    )

    axL.set_aspect("equal", adjustable="box")
    axL.set_xlim(0.0, lim)
    axL.set_ylim(0.0, lim)
    axL.set_xlabel(r"LiDAR $|\mathbf{u}|$ (m/s)")
    axL.set_ylabel(r"Model $|\mathbf{u}|$ (m/s)")
    axL.set_title(f"(a) Scatter — UTC {ts.strftime('%Y-%m-%d %H:%M')}, all stations")
    axL.grid(True, alpha=0.25, ls="--", lw=0.5)

    cb = fig.colorbar(sc_cfd, ax=axL, fraction=0.046, pad=0.02)
    cb.set_label("Height (m)")

    txt = (
        f"CFD vs obs: N={n_cfd}, R²={r2_cfd:.3f}, RMSE={rmse_cfd:.3f}, MBE={mbe_cfd:+.3f}\n"
        f"WRF vs obs: N={n_wrf}, R²={r2_wrf:.3f}, RMSE={rmse_wrf:.3f}, MBE={mbe_wrf:+.3f}"
    )
    axL.text(
        0.03,
        0.97,
        txt,
        transform=axL.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.92),
    )

    # --- Right: profile ---
    z = prof["Height"].to_numpy(dtype=float)
    axR.plot(prof["ws_obs"], z, color="0.1", lw=1.6, marker="o", ms=3.5, label=r"LiDAR $|\mathbf{u}|$")
    axR.plot(prof["ws_wrf"], z, color="0.45", lw=1.4, ls="--", label=r"WRF $|\mathbf{u}|$")
    axR.plot(prof["ws_cfd"], z, color="C0", lw=1.4, ls=":", marker="^", ms=3.5, label=r"CFD $|\mathbf{u}|$")

    axR.set_xlabel(r"Wind speed $|\mathbf{u}|$ (m/s)")
    axR.set_ylabel("Height (m)")
    axR.set_ylim(0.0, 2000.0)
    axR.set_title(f"(b) Vertical profile — {st} ({lon0:.4f}°E, {lat0:.4f}°N)")
    axR.grid(True, alpha=0.25, ls="--", lw=0.5)

    axR.text(
        0.03,
        0.03,
        "Nocturnal ABL (UTC evening; local night)\nStable / neutral layers common",
        transform=axR.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.92),
    )
    axR.legend(loc="upper right", frameon=True)

    fig.suptitle(
        f"WRF–OpenFOAM downscaling validation snapshot — {ts.strftime('%Y-%m-%d %H:%M')} UTC",
        fontsize=13,
        fontweight="bold",
    )

    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)

    print("OK:", out_path)
    print("STATS_CFD", {"N": n_cfd, "R2": r2_cfd, "RMSE": rmse_cfd, "MBE": mbe_cfd})
    print("STATS_WRF", {"N": n_wrf, "R2": r2_wrf, "RMSE": rmse_wrf, "MBE": mbe_wrf})


if __name__ == "__main__":
    main()
