#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单时刻 WS 垂直廓线：composite | GAW103 | GAW104 | GAW111（1×4，不画 WD）。

LiDAR / WRF / simpleFoam 按 tag 取合并表；bbsf 取对应算例
``postProcessing/CFD_lidar_simulation_<tag>_bbsf.csv``
（由仓库根 ``util/extract_cfd_lidar_xarray.py`` 抽取）。

Usage
-----
    python experiments/nonisothermal/scripts/plot_ws_composite_stations_simpleFoam_bbsf.py 20250901_1000
    python experiments/nonisothermal/scripts/plot_ws_composite_stations_simpleFoam_bbsf.py 20250908_0000
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent / "util"))
import case_paths as cp  # noqa: E402

SITES = ("GAW103", "GAW104", "GAW111")
HEIGHT_BINS = np.arange(0, 2150, 50)
WS_MAX_OBS = 30.0
WS_MAX_CFD = 20.0

COLOR_OBS = "#1a1a2e"
COLOR_WRF = "#e07b39"
COLOR_SIMPLE = "#2196a5"
COLOR_BBSF = "#c2185b"

LABEL_OBS = "LiDAR (obs)"
LABEL_WRF = "WRF"
LABEL_SIMPLE = "W-OF (simpleFoam)"
LABEL_BBSF = "W-OF (bbsf)"


def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
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
    })


def _format_time_label_for_display(t_raw: str, tz: str) -> str:
    dt = pd.Timestamp(t_raw)
    if tz.lower() == "lst":
        dt = dt + pd.Timedelta(hours=8)
        suffix = "UTC+8"
    else:
        suffix = "UTC"
    return f"{dt.strftime('%Y-%m-%d %H:%M')} ({suffix})"


def load_merged(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["ws_cfd"] = np.sqrt(df["u_cfd"] ** 2 + df["v_cfd"] ** 2)
    df = df[df["obtid"].isin(SITES)].copy()
    obs_ok = (df["ws_obs"] <= WS_MAX_OBS) | df["ws_obs"].isna()
    cfd_ok = df["ws_cfd"] <= WS_MAX_CFD
    df["qc_ok"] = obs_ok & cfd_ok
    return df


def load_bbsf(path: Path, *, extract_hint: str | None = None) -> pd.DataFrame:
    if not path.is_file():
        hint = extract_hint or "<case>"
        raise FileNotFoundError(
            f"bbsf LiDAR CSV not found: {path}\n"
            "Run: python util/extract_cfd_lidar_xarray.py "
            f"{hint}\n"
            f"then move the CSV to {path}"
        )
    b = pd.read_csv(path, parse_dates=["datetime"])
    ws_col = "WS_cfd" if "WS_cfd" in b.columns else "ws_cfd"
    z_col = "z_probe" if "z_probe" in b.columns else "Height"
    b = b.rename(columns={ws_col: "ws_cfd_bbsf", z_col: "Height"})
    if "ws_cfd_bbsf" not in b.columns:
        b["ws_cfd_bbsf"] = np.sqrt(b["U_cfd"] ** 2 + b["V_cfd"] ** 2)
    b = b[b["obtid"].isin(SITES)].copy()
    return b[["datetime", "obtid", "Height", "ws_cfd_bbsf"]]


def attach_bbsf(df: pd.DataFrame, bbsf: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ws_cfd_bbsf"] = np.nan
    for obtid, g in df.groupby("obtid"):
        src = bbsf[bbsf["obtid"] == obtid].sort_values("Height")
        if src.empty:
            continue
        f = interp1d(
            src["Height"].to_numpy(),
            src["ws_cfd_bbsf"].to_numpy(),
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        out.loc[g.index, "ws_cfd_bbsf"] = f(g["Height"].to_numpy())
    bbsf_ok = out["ws_cfd_bbsf"].isna() | (out["ws_cfd_bbsf"] <= WS_MAX_CFD)
    out["qc_ok"] = out["qc_ok"] & bbsf_ok
    return out


def _bin_mean(sub: pd.DataFrame, value_col: str, height_col: str = "Height") -> pd.DataFrame:
    work = sub.dropna(subset=[value_col, height_col]).copy()
    if work.empty:
        return pd.DataFrame(columns=["mean_h", "mean_ws"])
    work["H_bin"] = pd.cut(work[height_col], bins=HEIGHT_BINS)
    agg = work.groupby("H_bin", observed=True).agg(
        mean_ws=(value_col, "mean"),
        mean_h=(height_col, "mean"),
    ).dropna(subset=["mean_h"]).reset_index()
    return agg


def aggregate_composite(sub: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "obs": _bin_mean(sub, "ws_obs"),
        "wrf": _bin_mean(sub, "ws_wrf"),
        "simple": _bin_mean(sub, "ws_cfd"),
        "bbsf": _bin_mean(sub, "ws_cfd_bbsf"),
    }


def aggregate_station(sub: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return aggregate_composite(sub)


def _draw_ws_profiles(ax, profiles: dict[str, pd.DataFrame], zmax: float, title: str) -> None:
    obs = profiles["obs"]
    if not obs.empty:
        ax.plot(
            obs["mean_ws"], obs["mean_h"],
            color=COLOR_OBS, lw=2.0, ls="-", marker="o", ms=4, alpha=0.9, zorder=5,
        )
    wrf = profiles["wrf"]
    if not wrf.empty:
        ax.plot(wrf["mean_ws"], wrf["mean_h"], color=COLOR_WRF, lw=2.0, ls="--")
    simple = profiles["simple"]
    if not simple.empty:
        ax.plot(simple["mean_ws"], simple["mean_h"], color=COLOR_SIMPLE, lw=2.0, ls="-")
    bbsf = profiles["bbsf"]
    if not bbsf.empty:
        ax.plot(bbsf["mean_ws"], bbsf["mean_h"], color=COLOR_BBSF, lw=2.2, ls="-.")

    for h_line in (300, 1000):
        ax.axhline(h_line, color="0.6", lw=0.8, ls=":", zorder=0)
    ax.set_ylim(0, zmax)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.set_xlabel(r"Wind Speed (m s$^{-1}$)", fontsize=11)
    ax.set_title(title, fontweight="bold", loc="left")


def plot_four_panels(
    df: pd.DataFrame,
    datetime_utc: str,
    out_dir: Path,
    *,
    tz: str = "lst",
    zmax: float = 1000.0,
) -> Path:
    tz = tz.lower()
    if tz not in {"utc", "lst"}:
        raise ValueError("tz must be one of: utc, lst")
    dt = pd.Timestamp(datetime_utc)
    sub_all = df[(df["datetime"] == dt) & df["qc_ok"]].copy()
    if sub_all.empty:
        raise ValueError(f"no QC-passed rows at {datetime_utc}")

    fig, axes = plt.subplots(
        1, 4, figsize=(16.5, 6.0), sharey=True, constrained_layout=True,
    )
    titles = tuple(
        f"({letter}) {name}"
        for letter, name in zip("abcd", ("Composite",) + SITES)
    )
    subsets = [sub_all] + [sub_all[sub_all["obtid"] == s] for s in SITES]
    aggregators = [aggregate_composite] + [aggregate_station] * 3

    for i, (ax, title, sub, agg_fn) in enumerate(zip(axes, titles, subsets, aggregators)):
        profiles = agg_fn(sub)
        _draw_ws_profiles(ax, profiles, zmax, title)
        if i == 0:
            ax.set_ylabel("Height (m)", fontsize=11)
        else:
            ax.tick_params(axis="y", labelleft=False)

    legend_handles = [
        Line2D([0], [0], color=COLOR_OBS, lw=2, ls="-", marker="o", ms=5, label=LABEL_OBS),
        Line2D([0], [0], color=COLOR_WRF, lw=2, ls="--", label=LABEL_WRF),
        Line2D([0], [0], color=COLOR_SIMPLE, lw=2, ls="-", label=LABEL_SIMPLE),
        Line2D([0], [0], color=COLOR_BBSF, lw=2.2, ls="-.", label=LABEL_BBSF),
    ]
    axes[0].legend(handles=legend_handles, fontsize=9, loc="upper right", framealpha=0.9)

    tl_disp = _format_time_label_for_display(datetime_utc, tz=tz)
    fig.suptitle(f"WS Profiles — {tl_disp}", fontsize=14, fontweight="bold")

    out_dir.mkdir(parents=True, exist_ok=True)
    dt_tag = dt.strftime("%Y%m%d_%H%M")
    save_path = out_dir / f"ws_composite_stations_{dt_tag}_z{int(zmax)}m_tz-{tz}.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot 4-panel WS profiles: composite + GAW103/104/111 (simpleFoam vs bbsf).",
    )
    p.add_argument(
        "tag",
        help="time tag: 20250901_1000 or 20250908_0000 (optional _bbsf suffix)",
    )
    p.add_argument("--datetime", default=None, help="UTC hour; default follows tag")
    p.add_argument("--tz", choices=["utc", "lst"], default="lst")
    p.add_argument("--zmax", type=float, default=1000.0)
    p.add_argument("--csv", type=Path, default=None, help="merged LiDAR/WRF/simpleFoam CSV")
    p.add_argument("--bbsf-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = cp.normalize_tag(args.tag)
    merged = args.csv or cp.merged_csv(tag)
    bbsf_csv = args.bbsf_csv or cp.bbsf_lidar_csv(tag)
    out_dir = args.out_dir or cp.figs_dir()
    datetime_utc = (args.datetime or cp.datetime_utc(tag)).strip()
    configure_matplotlib_style()
    df = attach_bbsf(
        load_merged(merged),
        load_bbsf(bbsf_csv, extract_hint=cp.rel_case_dir(tag)),
    )
    save_path = plot_four_panels(
        df,
        datetime_utc,
        out_dir,
        tz=args.tz,
        zmax=args.zmax,
    )
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
