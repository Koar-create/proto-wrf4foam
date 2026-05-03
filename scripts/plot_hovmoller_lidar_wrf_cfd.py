#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _infer_time_freq(times_utc: pd.DatetimeIndex) -> pd.Timedelta:
    if len(times_utc) < 3:
        return pd.Timedelta(hours=1)
    diffs = np.diff(times_utc.asi8)  # ns
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return pd.Timedelta(hours=1)
    # mode of diffs (ns)
    vals, counts = np.unique(diffs, return_counts=True)
    dt_ns = int(vals[np.argmax(counts)])
    dt = pd.to_timedelta(dt_ns, unit="ns")
    # guard against weirdly tiny dt
    if dt < pd.Timedelta(minutes=1):
        return pd.Timedelta(hours=1)
    return dt


def _make_full_daily_time_index(times_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Expand to full 00:00–23:xx each day at inferred frequency.
    This makes missing windows (e.g. 00:00–10:00 UTC) show as blank.
    """
    tmin = times_utc.min()
    tmax = times_utc.max()
    dt = _infer_time_freq(times_utc)
    days = pd.date_range(tmin.floor("D"), tmax.ceil("D"), freq="D", tz="UTC")
    full = []
    for d in days:
        # inclusive end for one-day range
        end = d + pd.Timedelta(days=1) - dt
        full.append(pd.date_range(d, end, freq=dt, tz="UTC"))
    return full[0].append(full[1:]) if len(full) > 1 else full[0]


def _pivot_hovmoller(
    df: pd.DataFrame,
    value_col: str,
    *,
    full_times_utc: pd.DatetimeIndex | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (times, heights, Z[height, time]) suitable for pcolormesh.
    """
    piv = (
        df.pivot_table(index="Height", columns="datetime", values=value_col, aggfunc="mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if full_times_utc is not None:
        piv = piv.reindex(columns=full_times_utc, copy=False)
    heights = piv.index.to_numpy(dtype=float)
    times = piv.columns.to_numpy()
    z = piv.to_numpy(dtype=float)
    return times, heights, z


def _default_out_path(*, obtid: str, ymax: float, out_dir: Path, sensitivity: bool) -> Path:
    ymax_tag = int(round(float(ymax)))
    prefix = "hovmoller_ws_obs_wrf_cfd_ref_sen" if sensitivity else "hovmoller_ws_obs_wrf_cfd"
    return out_dir / f"{prefix}_{obtid}_y{ymax_tag}.png"


def _add_sunrise_sunset_guides(*, axes: list[plt.Axes], tmin_utc: pd.Timestamp, tmax_utc: pd.Timestamp):
    # sunrise/sunset guides (UTC): sunrise=22:00, sunset=10:00 (match scripts/task5_sample_space_map.py)
    days = pd.date_range(tmin_utc.floor("D"), tmax_utc.ceil("D"), freq="D", tz="UTC")
    sunset_handle = None
    sunrise_handle = None
    for d in days:
        dt_sunset = d + pd.Timedelta(hours=10)
        dt_sunrise = d + pd.Timedelta(hours=22)
        if tmin_utc <= dt_sunset <= tmax_utc:
            x = mdates.date2num(dt_sunset.to_pydatetime())
            for ax in axes:
                h = ax.axvline(x, color="#d62728", linestyle="--", linewidth=1.3, alpha=0.75)
                if sunset_handle is None:
                    sunset_handle = h
        if tmin_utc <= dt_sunrise <= tmax_utc:
            x = mdates.date2num(dt_sunrise.to_pydatetime())
            for ax in axes:
                h = ax.axvline(x, color="#1f77b4", linestyle="-.", linewidth=1.3, alpha=0.75)
                if sunrise_handle is None:
                    sunrise_handle = h
    if sunset_handle is not None:
        sunset_handle.set_label("Sunset (UTC 10:00)")
    if sunrise_handle is not None:
        sunrise_handle.set_label("Sunrise (UTC 22:00)")
    return sunset_handle, sunrise_handle


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot Hovmöller with shared viridis scale (standard or sensitivity mode)."
    )
    ap.add_argument(
        "--sensitivity",
        action="store_true",
        help="Sensitivity mode: use data/260413 processed CSV and output dir; plot 4 panels.",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Input merged CSV path.",
    )
    ap.add_argument(
        "--obtid",
        type=str,
        default=None,
        help="Station id (obtid). Default: first unique obtid in CSV.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output figure path.",
    )
    ap.add_argument(
        "--ymax",
        type=float,
        default=2000.0,
        help="Limit y-axis upper bound (meters). This is also reflected in the default output filename.",
    )
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--vmin", type=float, default=None, help="Override shared colormap vmin.")
    ap.add_argument(
        "--vmax",
        type=float,
        default=8.0,
        help="Override shared colormap vmax. Default: 8 (recommended for this dataset).",
    )
    args = ap.parse_args()

    sensitivity = bool(args.sensitivity)
    default_csv = (
        Path("data") / "260413" / "processed" / "merged_lidar_simulation_final_nighttime_only.csv"
        if sensitivity
        else Path("data") / "260409" / "processed" / "merged_lidar_simulation_final.csv"
    )
    out_dir = (
        Path("results") / "hovmoller" / "260413-sensitivity"
        if sensitivity
        else Path("results") / "hovmoller" / "260409"
    )

    csv_path = args.csv if args.csv is not None else default_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns or "Height" not in df.columns:
        raise ValueError("CSV must contain columns: datetime, Height")

    # treat input time as UTC (matches existing WRF/CFD processing in this repo)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    if args.obtid is None:
        if "obtid" not in df.columns:
            raise ValueError("CSV has no obtid column; please add one or pass a filtered CSV.")
        obtid = str(df["obtid"].dropna().unique()[0])
    else:
        obtid = str(args.obtid)

    if "obtid" in df.columns:
        df = df[df["obtid"].astype(str) == obtid].copy()

    ymax = float(args.ymax)
    if not np.isfinite(ymax) or ymax <= 0:
        raise ValueError(f"Invalid --ymax: {args.ymax}")
    df = df[np.asarray(df["Height"], dtype=float) <= ymax].copy()

    # ensure CFD wind-speed columns exist (compute from u/v if needed)
    if sensitivity:
        if "ws_cfd_ref" not in df.columns:
            if {"u_cfd_ref", "v_cfd_ref"}.issubset(df.columns):
                df["ws_cfd_ref"] = np.sqrt(
                    np.asarray(df["u_cfd_ref"], dtype=float) ** 2 + np.asarray(df["v_cfd_ref"], dtype=float) ** 2
                )
            else:
                raise ValueError("Sensitivity CSV missing ws_cfd_ref and u_cfd_ref/v_cfd_ref.")
        if "ws_cfd_sen" not in df.columns:
            if {"u_cfd_sen", "v_cfd_sen"}.issubset(df.columns):
                df["ws_cfd_sen"] = np.sqrt(
                    np.asarray(df["u_cfd_sen"], dtype=float) ** 2 + np.asarray(df["v_cfd_sen"], dtype=float) ** 2
                )
            else:
                raise ValueError("Sensitivity CSV missing ws_cfd_sen and u_cfd_sen/v_cfd_sen.")
        required = {"ws_obs", "ws_wrf", "ws_cfd_ref", "ws_cfd_sen"}
    else:
        if "ws_cfd" not in df.columns:
            if {"u_cfd", "v_cfd"}.issubset(df.columns):
                df["ws_cfd"] = np.sqrt(np.asarray(df["u_cfd"], dtype=float) ** 2 + np.asarray(df["v_cfd"], dtype=float) ** 2)
            else:
                raise ValueError("Standard CSV missing ws_cfd and u_cfd/v_cfd.")
        required = {"ws_obs", "ws_wrf", "ws_cfd"}

    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # expand time axis to full-day grid to keep 00:00–10:00 UTC blank in sensitivity mode
    t_unique = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True).dropna().unique()).sort_values()
    full_times = _make_full_daily_time_index(t_unique) if sensitivity else None

    t_obs, z_obs, ws_obs = _pivot_hovmoller(df, "ws_obs", full_times_utc=full_times)
    t_wrf, z_wrf, ws_wrf = _pivot_hovmoller(df, "ws_wrf", full_times_utc=full_times)
    if sensitivity:
        t_ref, z_ref, ws_ref = _pivot_hovmoller(df, "ws_cfd_ref", full_times_utc=full_times)
        t_sen, z_sen, ws_sen = _pivot_hovmoller(df, "ws_cfd_sen", full_times_utc=full_times)
    else:
        t_cfd, z_cfd, ws_cfd = _pivot_hovmoller(df, "ws_cfd", full_times_utc=None)

    if sensitivity:
        all_vals = np.concatenate([ws_obs.ravel(), ws_wrf.ravel(), ws_ref.ravel(), ws_sen.ravel()])
    else:
        all_vals = np.concatenate([ws_obs.ravel(), ws_wrf.ravel(), ws_cfd.ravel()])
    finite = np.isfinite(all_vals)
    if not np.any(finite):
        raise ValueError("All ws values are NaN/inf; cannot plot.")

    vmin = float(np.nanmin(all_vals[finite])) if args.vmin is None else float(args.vmin)
    vmax = float(np.nanmax(all_vals[finite])) if args.vmax is None else float(args.vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        raise ValueError(f"Invalid color range: vmin={vmin}, vmax={vmax}")

    nrows = 4 if sensitivity else 3
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 10.5 if sensitivity else 9), sharex=True, constrained_layout=True)

    if sensitivity:
        panels = [
            ("LiDAR Obs", t_obs, z_obs, ws_obs),
            ("Mesoscale (WRF)", t_wrf, z_wrf, ws_wrf),
            ("CFD Control", t_ref, z_ref, ws_ref),
            ("CFD Sensitivity (Sp=-0.01)", t_sen, z_sen, ws_sen),
        ]
    else:
        panels = [
            ("LiDAR Obs", t_obs, z_obs, ws_obs),
            ("Mesoscale (WRF)", t_wrf, z_wrf, ws_wrf),
            ("CFD (OpenFOAM)", t_cfd, z_cfd, ws_cfd),
        ]

    # add sunrise/sunset guides across all panels (UTC)
    if sensitivity:
        tmin_utc = full_times.min()
        tmax_utc = full_times.max()
    else:
        tmin_utc = pd.to_datetime(np.concatenate([t_obs, t_wrf, t_cfd]), utc=True).min()
        tmax_utc = pd.to_datetime(np.concatenate([t_obs, t_wrf, t_cfd]), utc=True).max()
    sunset_h, sunrise_h = _add_sunrise_sunset_guides(axes=list(axes), tmin_utc=tmin_utc, tmax_utc=tmax_utc)

    mappable = None
    for ax, (title, tt, zz, field) in zip(axes, panels):
        # Convert datetime64 -> matplotlib date numbers for stable pcolormesh behavior
        tt_num = mdates.date2num(pd.to_datetime(tt, utc=True).to_pydatetime())
        m = ax.pcolormesh(tt_num, zz, field, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
        mappable = m
        ax.set_ylabel("Height (m)")
        ax.set_title(title)
        ax.set_ylim(0, ymax)
        ax.grid(False)

    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))

    cbar = fig.colorbar(mappable, ax=axes, orientation="vertical", shrink=0.95, pad=0.02)
    cbar.set_label("Wind speed (m/s)")

    mode_tag = "sensitivity" if sensitivity else "standard"
    fig.suptitle(f"Hovmöller: wind speed (shared scale) | obtid={obtid} | mode={mode_tag}", y=1.02)
    guide_handles = [h for h in (sunset_h, sunrise_h) if h is not None]
    if guide_handles:
        fig.legend(
            handles=guide_handles,
            loc="upper right",
            bbox_to_anchor=(0.985, 1.02),
            fontsize=9,
            frameon=True,
            ncol=len(guide_handles),
        )

    out_path = args.out if args.out is not None else _default_out_path(obtid=obtid, ymax=ymax, out_dir=out_dir, sensitivity=sensitivity)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path.resolve()}")
    print(f"Color range: vmin={vmin:.3g}, vmax={vmax:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

