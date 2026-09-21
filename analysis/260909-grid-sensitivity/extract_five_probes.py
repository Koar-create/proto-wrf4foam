#!/usr/bin/env python3
"""Step 2: extract M1/M2/M3 profiles at the frozen five-probe transect.

Does not remesh or rerun CFD. Reads probes_5pt.json.

  python analysis/260909-grid-sensitivity/extract_five_probes.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from extract_and_plot import find_cell_centres, latest_time, read_vector_field

REPO = Path(__file__).resolve().parents[2]
CFG_PATH = Path(__file__).with_name("probes_5pt.json")
# Do not reuse 260409 LiDAR / WRF / OpenFOAM colours (#1a1a2e, #e07b39, #2196a5).
COLOR_MESH = {
    "M1_3.0M": "#6b6b6b",
    "M2_5.4M": "#5e3c99",
    "M3_8.0M": "#1b7837",
}
MESH_STYLE = {
    "M1_3.0M": dict(ls=":", lw=1.6),
    "M2_5.4M": dict(ls="-", lw=2.2),
    "M3_8.0M": dict(ls="-", lw=2.0),
}
MESH_LABEL = {
    "M1_3.0M": "M1 3.78M",
    "M2_5.4M": "M2 5.47M",
    "M3_8.0M": "M3 8.71M",
}


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


def _add_panel_label(ax, idx: int) -> None:
    ax.text(
        0.04, 0.97, f"({chr(ord('a') + idx)})",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=12, fontweight="bold", zorder=10,
    )


def load_cfg() -> dict:
    with CFG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def z_levels(cfg: dict) -> np.ndarray:
    spec = cfg["z_levels_m"]
    stop = spec["stop"]
    step = spec["step"]
    return np.arange(spec["start"], stop + 0.5 * step, step)


def extract_case(case: Path, label: str, probes: list[dict], heights: np.ndarray, datetime: str) -> pd.DataFrame:
    last = latest_time(case)
    coords = find_cell_centres(case)
    U = read_vector_field(case / last / "U")
    if len(U) != len(coords):
        raise ValueError(f"{label}: U n={len(U)} vs centres n={len(coords)}")
    recs = []
    cx, cy, cz = coords[:, 0], coords[:, 1], coords[:, 2]
    for probe in probes:
        pid, xyx, xyy = probe["id"], probe["x"], probe["y"]
        dist_h = np.sqrt((cx - xyx) ** 2 + (cy - xyy) ** 2)
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
        u_out, v_out = f_u(heights), f_v(heights)
        ws = np.sqrt(u_out**2 + v_out**2)
        for z, uu, vv, ww in zip(heights, u_out, v_out, ws):
            recs.append(
                {
                    "datetime": datetime,
                    "grid": label,
                    "probe": pid,
                    "x": xyx,
                    "y": xyy,
                    "z_probe": float(z),
                    "U_cfd": float(uu),
                    "V_cfd": float(vv),
                    "WS_cfd": float(ww),
                    "cfd_time": last,
                    "radius_m": float(r),
                }
            )
    return pd.DataFrame(recs)


def layer_stats(df: pd.DataFrame, probes: list[dict], zlo: float, zhi: float) -> pd.DataFrame:
    low = df[(df["z_probe"] >= zlo) & (df["z_probe"] <= zhi)].copy()
    rows = []
    for probe in probes:
        pid = probe["id"]
        a = low[low["probe"] == pid]
        m2 = a[a["grid"] == "M2_5.4M"].set_index("z_probe")["WS_cfd"]
        for other in ("M1_3.0M", "M3_8.0M"):
            b = a[a["grid"] == other].set_index("z_probe")["WS_cfd"]
            idx = m2.index.intersection(b.index)
            if len(idx) < 3:
                continue
            d = (m2.loc[idx] - b.loc[idx]).to_numpy()
            rows.append(
                {
                    "probe": pid,
                    "x": probe["x"],
                    "y": probe["y"],
                    "pair": f"M2_vs_{other}",
                    "n": int(len(idx)),
                    "rmse": float(np.sqrt(np.mean(d**2))),
                    "max_abs": float(np.max(np.abs(d))),
                    "mean_abs": float(np.mean(np.abs(d))),
                }
            )
    return pd.DataFrame(rows)


def plot_profiles(cfd: pd.DataFrame, probes: list[dict], out: Path, title: str) -> Path:
    configure_matplotlib_style()
    n = len(probes)
    fig, axes = plt.subplots(1, n, figsize=(12.8, 5.0), sharey=True)
    fig.subplots_adjust(wspace=0.18, top=0.82, bottom=0.14)
    zmax = 1000.0
    for i, (ax, probe) in enumerate(zip(axes, probes)):
        pid = probe["id"]
        for grid in ("M1_3.0M", "M3_8.0M", "M2_5.4M"):
            g = cfd[(cfd["grid"] == grid) & (cfd["probe"] == pid) & (cfd["z_probe"] <= zmax)]
            ax.plot(
                g["WS_cfd"],
                g["z_probe"],
                color=COLOR_MESH[grid],
                label=MESH_LABEL[grid],
                **MESH_STYLE[grid],
            )
        _add_panel_label(ax, i)
        ax.set_title(f"{pid}  ({probe['x']:.0f}, {probe['y']:.0f} m)")
        ax.set_xlabel(r"Wind Speed (m s$^{-1}$)")
        ax.set_xlim(left=0)
        ax.axhline(300.0, color="0.6", lw=0.8, ls=":", zorder=0)
        ax.axhline(1000.0, color="0.6", lw=0.8, ls=":", zorder=0)
    axes[0].set_ylabel("Height (m a.g.l.)")
    axes[0].set_ylim(0, zmax)
    handles = [
        Line2D([0], [0], color=COLOR_MESH[g], label=MESH_LABEL[g], **MESH_STYLE[g])
        for g in ("M1_3.0M", "M2_5.4M", "M3_8.0M")
    ]
    fig.legend(
        handles=handles,
        loc="upper center", ncol=3, frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(title, fontweight="bold", y=1.08)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg()
    os.chdir(REPO)
    probes = cfg["probes"]
    csv_path = REPO / cfg["outputs_step2"]["csv"]
    if args.skip_extract and csv_path.exists():
        cfd = pd.read_csv(csv_path)
        print(f"loaded {csv_path} {cfd.shape}")
    else:
        heights = z_levels(cfg)
        frames = []
        for label, rel in cfg["cases"].items():
            case = REPO / rel
            if not (case / "constant" / "polyMesh" / "points").exists():
                raise SystemExit(f"mesh missing: {case}")
            print(f"extract {label} {case}")
            frames.append(extract_case(case, label, probes, heights, cfg["datetime_utc"]))
        cfd = pd.concat(frames, ignore_index=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        cfd.to_csv(csv_path, index=False)
        print(f"wrote {csv_path} {cfd.shape}")
    zlo, zhi = cfg["metric_layer_m"]
    stats = layer_stats(cfd, probes, zlo, zhi)
    rmse_path = REPO / cfg["outputs_step2"]["rmse"]
    rmse_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(rmse_path, index=False)
    print(stats.to_string(index=False))
    fig_path = REPO / cfg["outputs_step2"]["fig"]
    plot_profiles(cfd, probes, fig_path, f"Grid sensitivity, five probes, {cfg['datetime_bjt']}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
