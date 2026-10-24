#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""URANS (and optional LES) vs LiDAR horizontal wind speed at ~500 m, 0–1800 s.

Both panels use DBS_095 second-level LiDAR. --with-les (default on) overlays
LES on the left (0901) panel only.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
URANS_ROOT = _SCRIPTS.parent
REPO_ROOT = URANS_ROOT.parent.parent
sys.path.insert(0, str(URANS_ROOT / "util"))
import case_paths as cp  # noqa: E402

LIDAR_0901 = (
    REPO_ROOT / "data" / "260409" / "raw" / "lidar"
    / "lidar_dbs095_20250901_0900-1100UTC.csv"
)
LIDAR_0908 = (
    REPO_ROOT / "data" / "260409" / "raw" / "lidar"
    / "lidar_dbs095_20250908_0000-0200UTC.csv"
)
LES_CASE = REPO_ROOT / "LES" / "experiments" / "20250901_1000_long_DFI"

HEIGHTS = (60, 80, 100, 120, 150, 200, 232, 300, 400, 500, 600)
STATIONS = (
    ("GAW103", 0),
    ("GAW104", 11),
    ("GAW111", 22),
)
COLORS = ("#2196a5", "#c2185b", "#5c6bc0")
Z_PLOT = 500
Z_LIDAR_TARGET = 500.0
SWITCH_S = (600.0, 1200.0)
RAMP_S = 30.0
VEC_RE = re.compile(r"\(([-+\deE.]+) ([-+\deE.]+) ([-+\deE.]+)\)")

CASES = (
    {
        "tag": "20250901_1000",
        "title": "(a) 2025-09-01 10:00 UTC",
        "t0": "2025-09-01 10:00:00",
        "t1": "2025-09-01 10:30:00",
        "ylim": (0.0, 7.0),
        "lidar_kind": "dbs095",
    },
    {
        "tag": "20250908_0000",
        "title": "(b) 2025-09-08 00:00 UTC",
        "t0": "2025-09-08 00:00:00",
        "t1": "2025-09-08 00:30:00",
        "ylim": (0.0, 30.0),
        "lidar_kind": "dbs095",
    },
)


def _probe_ws_from_files(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    rows: dict[float, np.ndarray] = {}
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            t = float(line.split()[0])
            vecs = VEC_RE.findall(line)
            if len(vecs) < 33:
                continue
            u = np.asarray(vecs, dtype=float)
            rows[t] = np.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2)
    if not rows:
        raise FileNotFoundError("no probe U rows")
    t = np.fromiter(sorted(rows), dtype=float)
    ws = np.stack([rows[x] for x in t])
    return t, ws


def load_urans_ws(tag: str) -> tuple[np.ndarray, np.ndarray]:
    case = URANS_ROOT / f"{tag}_pimplefoam" / "postProcessing" / "probesLidar"
    paths = [case / sub / "U" for sub in ("0", "840")]
    return _probe_ws_from_files(paths)


def load_les_ws(case: Path) -> tuple[np.ndarray, np.ndarray]:
    base = case / "postProcessing" / "probesLidar"
    paths = sorted(
        base.glob("*/U"),
        key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else 10**9,
    )
    t, ws = _probe_ws_from_files(paths)
    sel = t <= 1800.0
    return t[sel], ws[sel]


def _nearest_height(heights: np.ndarray, target: float) -> float:
    heights = np.asarray(heights, dtype=float)
    return float(heights[int(np.argmin(np.abs(heights - target)))])


def load_lidar_window(csv_path: Path, t0: str, t1: str) -> pd.DataFrame:
    t0p, t1p = pd.Timestamp(t0), pd.Timestamp(t1)
    usecols = ["obtid", "datetime", "Height", "WindSpd"]
    stations = {n for n, _ in STATIONS}
    parts = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=200_000):
        chunk["datetime"] = pd.to_datetime(chunk["datetime"])
        m = (
            chunk["obtid"].isin(stations)
            & (chunk["datetime"] >= t0p)
            & (chunk["datetime"] <= t1p)
        )
        if m.any():
            parts.append(chunk.loc[m])
    if not parts:
        raise RuntimeError(f"no LiDAR rows in {csv_path} for {t0}–{t1}")
    raw = pd.concat(parts, ignore_index=True)
    z_use = _nearest_height(raw["Height"].dropna().unique(), Z_LIDAR_TARGET)
    sub = raw[np.isclose(raw["Height"], z_use)].copy()
    sub["t_s"] = (sub["datetime"] - t0p).dt.total_seconds()
    sub = sub[(sub["t_s"] >= 0) & (sub["t_s"] <= 1800)]
    print(f"LiDAR {csv_path.name}: z={z_use:g} m  n={len(sub)}")
    return sub[["obtid", "t_s", "Height", "WindSpd"]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot URANS / LiDAR / optional LES wind-speed time series",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--with-les", dest="with_les", action="store_true",
        help="overlay LES on the 0901 panel (default)",
    )
    g.add_argument(
        "--no-les", dest="with_les", action="store_false",
        help="do not overlay LES",
    )
    p.set_defaults(with_les=True)
    p.add_argument("--les-case", type=Path, default=LES_CASE)
    p.add_argument(
        "--output", type=Path, default=None,
        help="output PNG (default: results/urans/urans_lidar_ws_z500_0-1800s.png)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
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
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })
    z_idx = HEIGHTS.index(Z_PLOT)
    lidar_0901 = load_lidar_window(LIDAR_0901, CASES[0]["t0"], CASES[0]["t1"])
    lidar_0908 = load_lidar_window(LIDAR_0908, CASES[1]["t0"], CASES[1]["t1"])
    z_l1 = float(lidar_0901["Height"].iloc[0])
    z_l8 = float(lidar_0908["Height"].iloc[0])

    les_t = les_ws = None
    if args.with_les:
        les_case = args.les_case if args.les_case.is_absolute() else REPO_ROOT / args.les_case
        les_t, les_ws = load_les_ws(les_case)
        print(f"LES {les_case.name}: n={len(les_t)} t=[{les_t[0]:.1f},{les_t[-1]:.1f}]")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    lidars = (lidar_0901, lidar_0908)

    for ax, spec, lid in zip(axes, CASES, lidars):
        t, ws = load_urans_ws(spec["tag"])
        for (name, off), color in zip(STATIONS, COLORS):
            ax.plot(
                t, ws[:, off + z_idx], color=color, lw=1.6, ls="-",
                label=f"{name} URANS",
            )
        if spec["tag"] == "20250901_1000" and les_t is not None:
            for (name, off), color in zip(STATIONS, COLORS):
                ax.plot(
                    les_t, les_ws[:, off + z_idx], color=color, lw=1.15, ls=":",
                    label=f"{name} LES",
                )
                print(
                    f"LES {name} z={Z_PLOT}: p50={np.median(les_ws[:, off + z_idx]):.2f} "
                    f"max={les_ws[:, off + z_idx].max():.2f}"
                )
        for (name, off), color in zip(STATIONS, COLORS):
            s = lid[lid["obtid"] == name].sort_values("t_s")
            if not len(s):
                continue
            # Full native time base: no resampling; dashed line only (no markevery).
            ax.plot(
                s["t_s"], s["WindSpd"], color=color, lw=1.2, ls="--",
                label=f"{name} LiDAR",
            )
            print(
                f"{spec['tag']} {name}: URANS t1800={ws[-1, off + z_idx]:.2f}  "
                f"LiDAR p50={s['WindSpd'].median():.2f} max={s['WindSpd'].max():.2f}"
            )
        for ts in SWITCH_S:
            ax.axvspan(ts - RAMP_S, ts, color="0.85", zorder=0)
            ax.axvline(ts, color="0.6", lw=0.8, ls="--", zorder=1)
        ax.set_xlim(0, 1800)
        ax.set_ylim(*spec["ylim"])
        ax.set_xlabel("Time (s)")
        ax.set_title(spec["title"], loc="left")
        ax.set_xticks([0, 600, 1200, 1800])
        ax.set_xticklabels(["0\n(HH:00)", "600\n(HH:10)", "1200\n(HH:20)", "1800\n(HH:30)"])

    ncol_left = 3 if args.with_les else 3
    axes[0].legend(fontsize=6.5, loc="lower right", ncol=ncol_left)
    axes[1].legend(fontsize=7, loc="upper right", ncol=3)
    axes[0].set_ylabel(r"Horizontal wind speed (m s$^{-1}$)")
    lidar_h = (
        f"{z_l1:.0f} m" if abs(z_l1 - z_l8) < 0.5 else f"{z_l1:.0f}/{z_l8:.0f} m"
    )
    fig.suptitle(
        f"Wind speed at z ≈ {Z_PLOT} m  "
        f"(URANS {Z_PLOT} m, LiDAR {lidar_h}"
        + (", LES 500 m" if args.with_les else "")
        + ")",
        fontsize=13, fontweight="bold",
    )
    out = args.output or (cp.figs_dir() / f"urans_lidar_ws_z{Z_PLOT}_0-1800s.png")
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
