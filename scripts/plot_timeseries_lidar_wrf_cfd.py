#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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


def _plot_one(
    df_ob_h: pd.DataFrame,
    *,
    obtid: str,
    h_req: float,
    tz_mode: str,
    rolling_3h: bool,
    out_path: Path,
    dpi: int,
) -> None:
    required = ["datetime", "ws_obs", "ws_wrf", "ws_cfd"]
    missing = [c for c in required if c not in df_ob_h.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df_ob_h.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime")
    df = df.set_index("datetime")

    if rolling_3h:
        for c in ["ws_obs", "ws_wrf", "ws_cfd"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").rolling("3h", center=True, min_periods=1).mean()
    else:
        for c in ["ws_obs", "ws_wrf", "ws_cfd"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if tz_mode == "lst":
        tzinfo = timezone(timedelta(hours=8))
        df = df.tz_convert(tzinfo)
        x_label = "Time (UTC+8)"
    else:
        tzinfo = timezone.utc
        x_label = "Time (UTC)"

    fig, ax = plt.subplots(figsize=(12.5, 4.8), constrained_layout=True)

    # Color/style mapping per results/颜色映射.md:
    # - LiDAR: black 'o-' (line + markers)
    # - WRF: orange dashed
    # - OpenFOAM: cyan solid
    ax.plot(
        df.index,
        df["ws_obs"],
        linestyle="-",
        marker="o",
        markersize=3.2,
        linewidth=1.4,
        color="black",
        alpha=0.9,
        label="LiDAR Obs",
    )
    ax.plot(df.index, df["ws_wrf"], linestyle="--", linewidth=1.8, color="#ff7f0e", alpha=0.95, label="WRF")
    ax.plot(df.index, df["ws_cfd"], linestyle="-", linewidth=2.0, color="#17becf", alpha=0.95, label="OpenFOAM")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Wind speed (m/s)")
    ax.set_title(f"Wind speed time series | obtid={obtid} | height={_fmt_height_tag(h_req)}")

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=tzinfo))
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.legend(loc="best", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot single-station single-height time series (LiDAR vs WRF vs OpenFOAM)."
    )
    ap.add_argument(
        "--obtid",
        nargs="+",
        required=True,
        help="Station id(s), e.g. --obtid GAW103",
    )
    ap.add_argument(
        "--height",
        nargs="+",
        required=True,
        type=float,
        help="Requested height(s) in meters, e.g. --height 100 200 300",
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
        default="utc",
        help="Timezone for x-axis: utc (default) or lst (UTC+8 local standard time).",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("data") / "260409" / "processed" / "merged_lidar_simulation_final.csv",
        help="Input merged CSV path (default: data/260409/processed/merged_lidar_simulation_final.csv).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "timeseries_lidar_wrf_cfd" / "260409",
        help="Output directory (default: results/timeseries_lidar_wrf_cfd/260409).",
    )
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    csv_path: Path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns or "Height" not in df.columns:
        raise ValueError("CSV must contain columns: datetime, Height")
    if "obtid" not in df.columns:
        raise ValueError("CSV must contain column: obtid")

    # Treat input time as UTC (repo convention).
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    _ensure_ws_cfd(df)

    needed = {"ws_obs", "ws_wrf", "ws_cfd", "Height", "obtid", "datetime"}
    missing = sorted([c for c in needed if c not in df.columns])
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    out_dir: Path = args.out_dir
    tz_mode = str(args.tz).lower()
    rolling_3h = bool(args.rolling_3h)
    roll_tag = "roll3h" if rolling_3h else "raw"

    for obtid in [str(x) for x in args.obtid]:
        df_ob = df[df["obtid"].astype(str) == obtid].copy()
        if len(df_ob) == 0:
            raise ValueError(f"No rows for obtid={obtid}")

        for h_req in [float(x) for x in args.height]:
            h_used = _pick_nearest_height(df_ob, h_req)
            df_ob_h = df_ob[np.isclose(pd.to_numeric(df_ob["Height"], errors="coerce"), h_used)].copy()
            if len(df_ob_h) == 0:
                raise ValueError(f"No rows for obtid={obtid} at nearest Height={h_used}")

            out_name = (
                f"ts_ws_obs_wrf_cfd_"
                f"obtid-{obtid}_"
                f"h{_fmt_height_tag(h_req)}_"
                f"{roll_tag}_"
                f"tz-{tz_mode}.png"
            )
            out_path = out_dir / out_name

            _plot_one(
                df_ob_h,
                obtid=obtid,
                h_req=h_req,
                tz_mode=tz_mode,
                rolling_3h=rolling_3h,
                out_path=out_path,
                dpi=int(args.dpi),
            )
            print(f"Saved: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

