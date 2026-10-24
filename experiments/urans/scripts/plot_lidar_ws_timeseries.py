#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""URANS (and optional LES) vs LiDAR horizontal wind speed, 0–1800 s.

Layout is 3 stations × 2 cases. Both columns use DBS_095 second-level LiDAR.
--with-les (default on) overlays LES on the 0901 column only. Probe / LiDAR
levels use the nearest available height to --height; requests below the
lowest probe (60 m) are rejected.
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
SWITCH_S = (600.0, 1200.0)
VEC_RE = re.compile(r"\(([-+\deE.]+) ([-+\deE.]+) ([-+\deE.]+)\)")

COLUMN_TITLES = (
    "2025-09-01 18:00 UTC+8",
    "2025-09-08 8:00 UTC+8",
)

CASES = (
    {
        "tag": "20250901_1000",
        "t0": "2025-09-01 10:00:00",
        "t1": "2025-09-01 10:30:00",
        "lidar_kind": "dbs095",
    },
    {
        "tag": "20250908_0000",
        "t0": "2025-09-08 00:00:00",
        "t1": "2025-09-08 00:30:00",
        "lidar_kind": "dbs095",
    },
)


def _panel_letter(row: int, col: int) -> str:
    return chr(ord("a") + row * 2 + col)


def _height_label_m(z: float) -> str:
    return f"{int(z)}m" if z == int(z) else f"{z:g}m"


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


def _height_index(z_req: float) -> tuple[int, float]:
    heights = np.asarray(HEIGHTS, dtype=float)
    z_use = _nearest_height(heights, z_req)
    return int(np.argmin(np.abs(heights - z_use))), z_use


def _ylim_auto(values: list[np.ndarray], floor: float = 0.0) -> tuple[float, float]:
    vmax = 0.0
    for arr in values:
        if arr is None or len(arr) == 0:
            continue
        a = np.asarray(arr, dtype=float)
        if np.any(np.isfinite(a)):
            vmax = max(vmax, float(np.nanmax(a)))
    if vmax <= 0:
        return floor, 1.0
    pad = max(0.5, 0.08 * vmax)
    ymax = float(np.ceil((vmax + pad) * 2) / 2)
    return floor, ymax


def load_lidar_window(
    csv_path: Path, t0: str, t1: str, *, z_target: float
) -> pd.DataFrame:
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
    z_use = _nearest_height(raw["Height"].dropna().unique(), z_target)
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
        "--height",
        type=float,
        default=500.0,
        help="requested height (m); URANS/LES use nearest probe level",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="output PNG (default: results/urans/urans_lidar_ws_z<height>_0-1800s.png)",
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
    z_req = float(args.height)
    z_lo = float(min(HEIGHTS))
    if z_req < z_lo:
        raise SystemExit(
            f"requested z={z_req:g} m is below the lowest probe ({z_lo:g} m)"
        )
    z_idx, z_urans = _height_index(z_req)
    lidar_0901 = load_lidar_window(
        LIDAR_0901, CASES[0]["t0"], CASES[0]["t1"], z_target=z_req,
    )
    lidar_0908 = load_lidar_window(
        LIDAR_0908, CASES[1]["t0"], CASES[1]["t1"], z_target=z_req,
    )
    z_l1 = float(lidar_0901["Height"].iloc[0])
    z_l8 = float(lidar_0908["Height"].iloc[0])

    les_t = les_ws = None
    if args.with_les:
        les_case = args.les_case if args.les_case.is_absolute() else REPO_ROOT / args.les_case
        les_t, les_ws = load_les_ws(les_case)
        print(f"LES {les_case.name}: n={len(les_t)} t=[{les_t[0]:.1f},{les_t[-1]:.1f}]")

    n_sta = len(STATIONS)
    fig, axes = plt.subplots(
        n_sta,
        2,
        figsize=(11.2, 8.6),
        sharex=True,
        sharey="col",
        constrained_layout=True,
        gridspec_kw={"wspace": 0.08},
    )
    lidars = (lidar_0901, lidar_0908)

    for col, (spec, lid) in enumerate(zip(CASES, lidars)):
        t, ws = load_urans_ws(spec["tag"])
        ymax_parts: list[np.ndarray] = []
        for row, ((name, off), color) in enumerate(zip(STATIONS, COLORS)):
            ax = axes[row, col]
            ax.plot(
                t, ws[:, off + z_idx], color=color, lw=1.6, ls="-",
                label="URANS", zorder=4,
            )
            ymax_parts.append(ws[:, off + z_idx])
            if spec["tag"] == "20250901_1000" and les_t is not None:
                ax.plot(
                    les_t, les_ws[:, off + z_idx], color=color, lw=1.15, ls=":",
                    label="LES", zorder=3,
                )
                ymax_parts.append(les_ws[:, off + z_idx])
                print(
                    f"LES {name} z={z_urans:g}: p50={np.median(les_ws[:, off + z_idx]):.2f} "
                    f"max={les_ws[:, off + z_idx].max():.2f}"
                )
            s = lid[lid["obtid"] == name].sort_values("t_s")
            if len(s):
                ymax_parts.append(s["WindSpd"].to_numpy())
                ax.plot(
                    s["t_s"], s["WindSpd"], color=color, lw=1.2, ls="--",
                    label="LiDAR", zorder=2,
                )
                print(
                    f"{spec['tag']} {name}: URANS t1800={ws[-1, off + z_idx]:.2f}  "
                    f"LiDAR p50={s['WindSpd'].median():.2f} max={s['WindSpd'].max():.2f}"
                )
            for ts in SWITCH_S:
                ax.axvline(ts, color="0.6", lw=0.8, ls="--", zorder=1)
            ax.set_xlim(0, 1800)
            letter = _panel_letter(row, col)
            z_lab = _height_label_m(z_urans)
            ax.text(
                0.02, 0.96,
                f"({letter}) {name}, {z_lab}",
                transform=ax.transAxes,
                va="top", ha="left", fontsize=10, fontweight="bold",
            )
            if col == 0:
                # In-axes legend keeps the inter-column gap narrow (no bbox outside).
                leg_loc = "lower right" if row == n_sta - 1 else "upper right"
                ax.legend(fontsize=7, loc=leg_loc, frameon=True)
            if row == n_sta - 1:
                ax.set_xlabel("Time (s)")
                ax.set_xticks([0, 600, 1200, 1800])
                ax.set_xticklabels(
                    ["0\n(HH:00)", "600\n(HH:10)", "1200\n(HH:20)", "1800\n(HH:30)"]
                )
            if col == 0 and row == 1:
                ax.set_ylabel(r"Horizontal wind speed (m s$^{-1}$)")
        _, ymax = _ylim_auto(ymax_parts)
        axes[0, col].set_ylim(0.0, ymax)

    fig.canvas.draw()
    for col, col_title in enumerate(COLUMN_TITLES):
        pos = axes[0, col].get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y1 + 0.018,
            col_title,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    z_tag = int(z_urans) if z_urans == int(z_urans) else z_urans
    print(f"LiDAR heights: 0901={z_l1:g} m, 0908={z_l8:g} m")
    out = args.output or (cp.figs_dir() / f"urans_lidar_ws_z{z_tag}_0-1800s.png")
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
