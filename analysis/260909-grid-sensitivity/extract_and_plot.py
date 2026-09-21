#!/usr/bin/env python3
"""Extract and overlay LiDAR-site wind-speed profiles for the 3/5.4/8M grid study.

Usage (after M1/M3 simpleFoam finishes):
  python analysis/260909-grid-sensitivity/extract_and_plot.py
  python analysis/260909-grid-sensitivity/extract_and_plot.py --datetime "2025-09-08 00:00:00"
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

REPO = Path(__file__).resolve().parents[2]
SITES = {
    "GAW103": {"x": 975.0, "y": -320.0},
    "GAW104": {"x": 450.0, "y": 350.0},
    "GAW111": {"x": 75.0, "y": 30.0},
}
HOUR_CFGS = {
    "2025-09-01 10:00:00": {
        "tag": "20250901_1000",
        "fig_tag": "20250901_1800BJT",
        "title": "Grid sensitivity, 2025-09-01 18:00 (UTC+8)",
        "merged": REPO / "data" / "260409" / "processed" / "merged_lidar_simulation_final.csv",
        "cases": {
            "M1_3.0M": REPO / "experiments" / "sensitivity_experiments" / "20250901_1000_two_boundaries_as_outlet-grid_3.0M",
            "M2_5.4M": REPO / "steady_experiments_finer_ABL" / "20250901_1000_two_boundaries_as_outlet",
            "M3_8.0M": REPO / "experiments" / "sensitivity_experiments" / "20250901_1000_two_boundaries_as_outlet-grid_8.0M",
        },
    },
    "2025-09-08 00:00:00": {
        "tag": "20250908_0000",
        "fig_tag": "20250908_0800BJT",
        "title": "Grid sensitivity, 2025-09-08 08:00 (UTC+8)",
        "merged": REPO / "data" / "260707" / "processed" / "merged_lidar_simulation_final.csv",
        "cases": {
            "M1_3.0M": REPO / "experiments" / "sensitivity_experiments" / "20250908_0000_two_boundaries_as_outlet-grid_3.0M",
            "M2_5.4M": REPO / "steady_experiments_finer_ABL" / "20250908_0000_two_boundaries_as_outlet",
            "M3_8.0M": REPO / "experiments" / "sensitivity_experiments" / "20250908_0000_two_boundaries_as_outlet-grid_8.0M",
        },
    },
}
JSON_PATH = REPO / "util" / "lidar_station_info.json"
OUT_CSV = REPO / "data" / "260909" / "raw" / "cfd"
OUT_FIG = REPO / "results" / "grid_sensitivity"
VEC_RE = re.compile(
    r"internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*;",
    re.DOTALL,
)
VEC_TRIPLE = re.compile(r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)")
COLOR = {
    "obs": "#1a1a2e",
    "wrf": "#e07b39",
    "M1_3.0M": "#8d99ae",
    "M2_5.4M": "#2196a5",
    "M3_8.0M": "#1d3557",
}


def latest_time(case: Path) -> str:
    times = []
    for p in case.iterdir():
        if p.is_dir():
            try:
                times.append(int(p.name))
            except ValueError:
                pass
    if not times:
        raise FileNotFoundError(f"no time directories in {case}")
    return str(max(times))


def read_vector_field(path: Path) -> np.ndarray:
    text = path.read_text(errors="replace")
    m = VEC_RE.search(text)
    if not m:
        raise ValueError(f"cannot parse vector field: {path}")
    triples = VEC_TRIPLE.findall(m.group(2))
    return np.array([[float(a), float(b), float(c)] for a, b, c in triples])


def find_cell_centres(case: Path) -> np.ndarray:
    for cand in (case / "0" / "C", case / "constant" / "cellCentres"):
        if cand.exists():
            return read_vector_field(cand)
    raise FileNotFoundError(f"cell centres missing in {case}")


def extract_case(case: Path, label: str, station_levels: dict, datetime: str) -> pd.DataFrame:
    last = latest_time(case)
    coords = find_cell_centres(case)
    U = read_vector_field(case / last / "U")
    if len(U) != len(coords):
        raise ValueError(f"{label}: U n={len(U)} vs centres n={len(coords)}")
    recs = []
    cx, cy, cz = coords[:, 0], coords[:, 1], coords[:, 2]
    for obtid, xy in SITES.items():
        dist_h = np.sqrt((cx - xy["x"]) ** 2 + (cy - xy["y"]) ** 2)
        n_min, r = 30, 50.0
        mask = dist_h <= r
        while mask.sum() < n_min and r <= 2000.0:
            r *= 1.5
            mask = dist_h <= r
        if mask.sum() < 3:
            mask = np.zeros(len(cx), dtype=bool)
            mask[np.argsort(dist_h)[:n_min]] = True
        z_sel = cz[mask]
        u_sel, v_sel = U[mask, 0], U[mask, 1]
        w = 1.0 / (dist_h[mask] + 1e-6)
        edges = np.arange(cz.min() - 10.0, cz.max() + 30.0, 20.0)
        centres = 0.5 * (edges[:-1] + edges[1:])
        num_u = np.zeros(len(centres))
        den_u = np.zeros(len(centres))
        num_v = np.zeros(len(centres))
        den_v = np.zeros(len(centres))
        for i, _zb in enumerate(centres):
            in_bin = (z_sel >= edges[i]) & (z_sel < edges[i + 1])
            if not np.any(in_bin):
                continue
            wb = w[in_bin]
            num_u[i] = (wb * u_sel[in_bin]).sum()
            den_u[i] = wb.sum()
            num_v[i] = (wb * v_sel[in_bin]).sum()
            den_v[i] = wb.sum()
        valid = (den_u > 0) & (den_v > 0)
        z_prof = centres[valid]
        u_prof = num_u[valid] / den_u[valid]
        v_prof = num_v[valid] / den_v[valid]
        _, ui = np.unique(z_prof, return_index=True)
        z_prof, u_prof, v_prof = z_prof[ui], u_prof[ui], v_prof[ui]
        f_u = interp1d(z_prof, u_prof, kind="linear", bounds_error=False, fill_value=(u_prof[0], u_prof[-1]))
        f_v = interp1d(z_prof, v_prof, kind="linear", bounds_error=False, fill_value=(v_prof[0], v_prof[-1]))
        heights = np.array(station_levels[obtid], dtype=float)
        heights = heights[heights < 2000.0]
        u_out, v_out = f_u(heights), f_v(heights)
        ws = np.sqrt(u_out**2 + v_out**2)
        for z, uu, vv, ww in zip(heights, u_out, v_out, ws):
            recs.append(
                {
                    "datetime": datetime,
                    "grid": label,
                    "obtid": obtid,
                    "z_probe": float(z),
                    "U_cfd": float(uu),
                    "V_cfd": float(vv),
                    "WS_cfd": float(ww),
                    "cfd_time": last,
                }
            )
    return pd.DataFrame(recs)


def layer_stats(df: pd.DataFrame) -> pd.DataFrame:
    low = df[(df["z_probe"] >= 52.0) & (df["z_probe"] <= 300.0)].copy()
    rows = []
    for obtid in SITES:
        a = low[low["obtid"] == obtid]
        m2 = a[a["grid"] == "M2_5.4M"].set_index("z_probe")["WS_cfd"]
        for other in ("M1_3.0M", "M3_8.0M"):
            b = a[a["grid"] == other].set_index("z_probe")["WS_cfd"]
            idx = m2.index.intersection(b.index)
            if len(idx) < 3:
                continue
            d = (m2.loc[idx] - b.loc[idx]).to_numpy()
            rows.append(
                {
                    "obtid": obtid,
                    "pair": f"M2_vs_{other}",
                    "n": int(len(idx)),
                    "rmse": float(np.sqrt(np.mean(d**2))),
                    "max_abs": float(np.max(np.abs(d))),
                    "mean_abs": float(np.mean(np.abs(d))),
                }
            )
    return pd.DataFrame(rows)


def plot_overlay(cfd: pd.DataFrame, merged: pd.DataFrame, out: Path, title: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 5.4), sharey=True)
    zmax = 1000.0
    for ax, obtid in zip(axes, SITES):
        sub_m = merged[(merged["obtid"] == obtid) & (merged["Height"] <= zmax)]
        ax.plot(sub_m["ws_obs"], sub_m["Height"], color=COLOR["obs"], lw=1.8, marker="o", ms=2.5, label="LiDAR")
        ax.plot(sub_m["ws_wrf"], sub_m["Height"], color=COLOR["wrf"], lw=1.6, ls="--", label="WRF")
        legend = {"M1_3.0M": "M1 3.78M", "M2_5.4M": "M2 5.47M", "M3_8.0M": "M3 8.71M"}
        for grid, ls in (("M1_3.0M", ":"), ("M2_5.4M", "-"), ("M3_8.0M", "-")):
            g = cfd[(cfd["grid"] == grid) & (cfd["obtid"] == obtid) & (cfd["z_probe"] <= zmax)]
            ax.plot(g["WS_cfd"], g["z_probe"], color=COLOR[grid], lw=1.8 if grid != "M1_3.0M" else 1.4, ls=ls, label=legend[grid])
        ax.set_title(obtid)
        ax.set_xlabel(r"$U$ (m s$^{-1}$)")
        ax.set_xlim(left=0)
        ax.axhline(300.0, color="0.5", lw=0.7, ls="--")
        ax.grid(True, alpha=0.25, ls="--")
    axes[0].set_ylabel("Height (m AGL)")
    axes[0].set_ylim(0, zmax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.08)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument(
        "--datetime",
        default="2025-09-01 10:00:00",
        choices=sorted(HOUR_CFGS),
        help="UTC snapshot to extract (default: 2025-09-01 10:00:00)",
    )
    args = ap.parse_args()
    cfg = HOUR_CFGS[args.datetime]
    cases = cfg["cases"]
    merged_path = cfg["merged"]
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, encoding="utf-8") as f:
        station_info = json.load(f)
    levels = {k: v["levels"] for k, v in station_info.items() if k in SITES}

    csv_path = OUT_CSV / f"grid_sensitivity_profiles_{cfg['tag']}.csv"
    if args.skip_extract and csv_path.exists():
        cfd = pd.read_csv(csv_path)
    else:
        frames = []
        for label, case in cases.items():
            if label == "M2_5.4M":
                print(f"M2 from merged CSV (production extraction)")
                continue
            if not (case / "constant" / "polyMesh" / "points").exists():
                raise SystemExit(f"mesh/results missing: {case}")
            print(f"extract {label} {case}")
            frames.append(extract_case(case, label, levels, args.datetime))
        merged_all = pd.read_csv(merged_path, parse_dates=["datetime"])
        m2 = merged_all[merged_all["datetime"] == pd.Timestamp(args.datetime)].copy()
        m2 = m2[m2["obtid"].isin(SITES)]
        if m2.empty:
            raise SystemExit(f"no M2 rows for {args.datetime} in {merged_path}")
        m2_rows = pd.DataFrame(
            {
                "datetime": args.datetime,
                "grid": "M2_5.4M",
                "obtid": m2["obtid"],
                "z_probe": m2["Height"],
                "U_cfd": m2["u_cfd"],
                "V_cfd": m2["v_cfd"],
                "WS_cfd": m2["ws_cfd"],
                "cfd_time": "merged",
            }
        )
        frames.append(m2_rows)
        cfd = pd.concat(frames, ignore_index=True)
        cfd.to_csv(csv_path, index=False)
        print(f"wrote {csv_path} {cfd.shape}")

    stats = layer_stats(cfd)
    stats_path = OUT_CSV / f"grid_sensitivity_layer_rmse_52_300m_{cfg['tag']}.csv"
    stats.to_csv(stats_path, index=False)
    print(stats.to_string(index=False))

    merged = pd.read_csv(merged_path, parse_dates=["datetime"])
    merged = merged[merged["datetime"] == pd.Timestamp(args.datetime)].copy()
    fig_path = OUT_FIG / f"grid_sensitivity_ws_profiles_GAW103_104_111_{cfg['fig_tag']}.png"
    plot_overlay(cfd, merged, fig_path, cfg["title"])
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    os.chdir(REPO)
    main()
