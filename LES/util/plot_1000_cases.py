#!/usr/bin/env python3
"""10:00 UTC mean-inlet vs DFSEM 100 s products. Writes only inside each case's figs_1000/.

Figures per case: z100 snapshots (t=25,100), UMean vs RANS, Q=1e-3 projections,
probe time series. Text: limiter stats, UMean-vs-RANS at LiDAR stations.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/20250901_0000_DFSEM"))
from plot_dfsem_z100 import LIDAR, read_vtp_points_U  # noqa: E402
from analyze_probes_les import STATIONS, read_probes_U  # noqa: E402

REPO = ROOT.parent
RANS_DIR = REPO / "steady_experiments_finer_ABL/20250901_1000_two_boundaries_as_outlet/postProcessing"
CASES = {
    "mean": ROOT / "experiments/20250901_1000",
    "dfsem": ROOT / "experiments/20250901_1000_DFSEM",
}
LABEL = {"mean": "mean-inlet LES", "dfsem": "DFSEM LES"}
AVG_START = {"mean": 5, "dfsem": 0}


def load_rans(h: int):
    d = np.loadtxt(RANS_DIR / f"{h}m.csv", delimiter=",", skiprows=1, usecols=(0, 1, 9, 10))
    u, v, x, y = d.T
    return x.reshape(1000, 1000), y.reshape(1000, 1000), np.hypot(u, v).reshape(1000, 1000)


def panel(ax, X, Y, Z, title, norm):
    im = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis", norm=norm)
    ax.set_aspect("equal")
    ax.set_xlim(-2500, 2500)
    ax.set_ylim(-2500, 2500)
    ax.set_xlabel("x (m)")
    ax.set_title(title, fontsize=10)
    for n, xy in LIDAR.items():
        ax.plot(*xy, "ko", ms=3)
        ax.annotate(n, xy, fontsize=7, xytext=(4, 4), textcoords="offset points")
    return im


def vtp_ws(case, kind, t, field):
    pts, vec = read_vtp_points_U(case / "postProcessing" / kind / str(t) / "z100.vtp", field=field)
    return pts, np.hypot(vec[:, 0], vec[:, 1])


def two_panel(tag, case, out, rX, rY, rW, XX, YY, kind, t, field, title_les, fname):
    pts, ws = vtp_ws(case, kind, t, field)
    g = griddata(pts[:, :2], ws, (XX, YY), method="linear")
    vmax = min(max(float(np.nanpercentile(np.concatenate([ws, rW.ravel()]), 99.5)), 1.2), 8.0)
    norm = Normalize(0, vmax)
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True)
    im = panel(ax[0], rX, rY, rW, "RANS mean (simpleFoam)", norm)
    panel(ax[1], XX, YY, g, title_les, norm)
    ax[0].set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("horizontal wind speed (m/s)")
    fig.suptitle(f"2025-09-01 10:00 UTC  z=100 m  |  {LABEL[tag]}", fontsize=11)
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)
    print(f"wrote {out / fname}  ws p50={np.median(ws):.3f} max={ws.max():.2f}")
    return pts, ws, g


def iso_plot(case, out, t, tag):
    p = case / "postProcessing/isoQ" / str(t) / "qIso1em3.vtp"
    pts, _ = read_vtp_points_U(p, field="Q")
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 5.0), constrained_layout=True)
    h0 = ax[0].hexbin(pts[:, 0], pts[:, 1], gridsize=120, cmap="inferno", mincnt=1,
                      extent=(-5000, 5000, -5000, 5000))
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")
    ax[0].set_title("xy projection")
    for n, xy in LIDAR.items():
        ax[0].plot(*xy, "c+", ms=7)
    fig.colorbar(h0, ax=ax[0], fraction=0.046, pad=0.04, label="points")
    h1 = ax[1].hexbin(pts[:, 0], pts[:, 2], gridsize=120, cmap="inferno", mincnt=1,
                      extent=(-5000, 5000, 0, 2000))
    ax[1].set_xlabel("x (m)")
    ax[1].set_ylabel("z (m)")
    ax[1].set_title("xz projection")
    fig.colorbar(h1, ax=ax[1], fraction=0.046, pad=0.04, label="points")
    fig.suptitle(f"{LABEL[tag]}  Q=1e-3 isosurface, t={t} s", fontsize=11)
    f = out / f"isoQ_1em3_t{t}.png"
    fig.savefig(f, dpi=140)
    plt.close(fig)
    east = int((pts[:, 0] > 4000).sum())
    south = int((pts[:, 1] < -4000).sum())
    city = int(((np.abs(pts[:, 0]) < 2500) & (np.abs(pts[:, 1]) < 2500)).sum())
    return f"isoQ t={t}: n={len(pts)} city(|x|,|y|<2500)={city} east(x>4000)={east} south(y<-4000)={south}"


def probe_series(case):
    base = case / "postProcessing/probesLidar"
    ts, Us = [], []
    for d in sorted(base.glob("*/U"), key=lambda p: int(p.parent.name)):
        t, U = read_probes_U(d)
        ts.append(t)
        Us.append(U)
    t = np.concatenate(ts)
    U = np.concatenate(Us)
    o = np.argsort(t, kind="stable")
    t, U = t[o], U[o]
    keep = np.concatenate([[True], np.diff(t) > 1e-9])
    return t[keep], U[keep]


def limiter_stats(case):
    log = case / "logs/log.pimpleFoam.100s"
    txt = log.read_text(errors="replace")
    res = []
    for name in ("limitU", "limitNut"):
        n = [int(m.group(1)) for m in re.finditer(rf"{name} Limited (\d+) \(", txt)]
        if n:
            res.append(f"{name}: n_steps={len(n)} last={n[-1]} max={max(n)} median={int(np.median(n))}")
        else:
            res.append(f"{name}: no lines")
    return res


def main():
    rX, rY, rW = load_rans(100)
    XX, YY = np.meshgrid(rX[0], rY[:, 0])
    rans = {h: load_rans(h) for h in (60, 100, 120)}
    report = []
    umean_g = {}
    for tag, case in CASES.items():
        out = case / "figs_1000"
        out.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
        report.append(f"## {case.name}")
        for t in (25, 100):
            two_panel(tag, case, out, rX, rY, rW, XX, YY, "surfacesU", t, "U",
                      f"{LABEL[tag]} snapshot t={t} s", f"xy_ws_z100_rans_vs_les_t{t}.png")
        pts, ws, g = two_panel(
            tag, case, out, rX, rY, rW, XX, YY, "surfacesUMean", 100, "UMean",
            f"{LABEL[tag]} UMean (avg {AVG_START[tag]}-100 s)", "xy_ws_z100_rans_vs_umean_t100.png")
        umean_g[tag] = g
        inner = (np.abs(pts[:, 0]) < 2500) & (np.abs(pts[:, 1]) < 2500)
        report.append(f"UMean z100 inner-box median={np.median(ws[inner]):.3f}  RANS median={np.nanmedian(rW):.3f}")
        for t in (25, 100):
            report.append(iso_plot(case, out, t, tag))
        report += limiter_stats(case)
        # probes
        t, U = probe_series(case)
        fig, ax = plt.subplots(3, 1, figsize=(8.2, 7.2), sharex=True)
        for a, (n, (i0, _, _)) in zip(ax, STATIONS.items()):
            u = U[:, i0 + 2]
            a.plot(t, np.hypot(u[:, 0], u[:, 1]), lw=0.7)
            a.set_ylabel(f"{n}\nWs z=100 (m/s)")
            a.grid(alpha=0.3)
        ax[-1].set_xlabel("t (s)")
        fig.suptitle(f"{LABEL[tag]} probes z=100 m, 10:00 UTC")
        fig.tight_layout()
        fig.savefig(out / "probes_ws_z100.png", dpi=140)
        plt.close(fig)
        # LES UMean at stations vs RANS (z=100 grid)
        sel = (t >= max(AVG_START[tag], t[0]))
        report.append("station z=100  LES UMean(vtp) | probe mean ws | RANS 100m")
        for n, xy in LIDAR.items():
            ix = int(np.argmin(np.abs(rX[0] - xy[0])))
            iy = int(np.argmin(np.abs(rY[:, 0] - xy[1])))
            i0 = STATIONS[n][0]
            pm = np.hypot(U[sel][:, i0 + 2, 0], U[sel][:, i0 + 2, 1]).mean()
            report.append(f"  {n}: {umean_g[tag][iy, ix]:.3f} | {pm:.3f} | {rW[iy, ix]:.3f}")
        report.append("probe-mean ws vs RANS by height (60/100/120 m)")
        for n, xy in LIDAR.items():
            ix = int(np.argmin(np.abs(rX[0] - xy[0])))
            iy = int(np.argmin(np.abs(rY[:, 0] - xy[1])))
            i0 = STATIONS[n][0]
            row = []
            for h, k in ((60, 0), (100, 2), (120, 3)):
                pm = np.hypot(U[sel][:, i0 + k, 0], U[sel][:, i0 + k, 1]).mean()
                row.append(f"{h}m LES={pm:.3f} RANS={rans[h][2][iy, ix]:.3f}")
            report.append(f"  {n}: " + "  ".join(row))
        report.append("")
    # side by side umean
    if len(umean_g) == 2:
        d = umean_g["dfsem"] - umean_g["mean"]
        report.append(f"DFSEM-minus-mean UMean z100: median={np.nanmedian(d):.4f} p95|d|={np.nanpercentile(np.abs(d), 95):.4f}")
    text = "\n".join(report) + "\n"
    (ROOT / "docs/verification/plots_1000_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
