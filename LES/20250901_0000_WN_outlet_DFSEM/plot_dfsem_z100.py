#!/usr/bin/env python3
"""DFSEM z=100 vs RANS and vs mean-inlet LES. Output stays in the DFSEM case."""
from __future__ import annotations

import argparse
import base64
import os
import struct
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

CASE = Path("/hpc2hdd/home/zyang248/WRF-OpenFOAM-Coupling/LES/20250901_0000_WN_outlet_DFSEM")
MEAN = Path("/hpc2hdd/home/zyang248/WRF-OpenFOAM-Coupling/LES/20250901_0000_WN_outlet")
RANS_CSV = Path(
    "/hpc2hdd/home/zyang248/WRF-OpenFOAM-Coupling/"
    "steady_experiments_finer_ABL/20250901_0000_two_boundaries_as_outlet/"
    "postProcessing/100m.csv"
)
OUTDIR = CASE / "figs_dfsem_100s"
HOTSPOT = (1585.56, -1397.21)
LIDAR = {"GAW103": (975, -320), "GAW104": (450, 350), "GAW111": (75, 30)}
os.environ.setdefault("MPLCONFIGDIR", str(CASE / ".mplconfig"))


def vtk_xml_binary_array(el: ET.Element, n: int, ncomp: int, dtype: str) -> np.ndarray:
    raw = base64.b64decode("".join((el.text or "").split()))
    header_nbytes = struct.unpack("<Q", raw[:8])[0]
    payload = raw[8 : 8 + header_nbytes]
    dt = np.dtype("<f4") if dtype == "Float32" else np.dtype("<i4")
    arr = np.frombuffer(payload, dtype=dt)
    return arr.reshape(n, ncomp) if ncomp > 1 else arr


def read_vtp_points_U(path: Path, field: str = "U"):
    tree = ET.parse(path)
    piece = tree.find(".//Piece")
    n = int(piece.attrib["NumberOfPoints"])
    pts_el = piece.find("./Points/DataArray")
    u_el = None
    for da in piece.iter("DataArray"):
        if da.attrib.get("Name") == field:
            u_el = da
            break
    if u_el is None:
        raise ValueError(f"No {field} array in {path}")
    pts = vtk_xml_binary_array(pts_el, n, 3, "Float32")
    ncomp = int(u_el.attrib.get("NumberOfComponents", "1"))
    u = vtk_xml_binary_array(u_el, n, ncomp, "Float32")
    return pts, u


def load_rans_100m():
    data = np.loadtxt(RANS_CSV, delimiter=",", skiprows=1, usecols=(0, 1, 9, 10))
    u, v, x, y = data.T
    nx = ny = 1000
    XX = x.reshape(ny, nx)
    YY = y.reshape(ny, nx)
    WS = np.hypot(u, v).reshape(ny, nx)
    return XX, YY, WS


def mark_stations(ax):
    ax.plot(*HOTSPOT, "r+", ms=8, mew=1.2)
    for name, xy in LIDAR.items():
        ax.plot(*xy, "ko", ms=3)
        ax.annotate(name, xy, fontsize=7, xytext=(4, 4), textcoords="offset points")


def panel(ax, xg, yg, zz, title, norm):
    im = ax.pcolormesh(xg, yg, zz, shading="auto", cmap="viridis", norm=norm)
    ax.set_aspect("equal")
    ax.set_xlim(-2500, 2500)
    ax.set_ylim(-2500, 2500)
    ax.set_xlabel("x (m)")
    ax.set_title(title, fontsize=10)
    mark_stations(ax)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", default="100")
    ap.add_argument(
        "--mode",
        choices=("snapshot", "umean", "three"),
        default="snapshot",
    )
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True, mode=0o700)

    rXX, rYY, rWS = load_rans_100m()
    xi = rXX[0]
    yi = rYY[:, 0]
    XX, YY = np.meshgrid(xi, yi)

    if args.mode == "snapshot":
        vtk_path = CASE / "postProcessing" / "surfacesU" / args.time / "z100.vtp"
        pts, vec = read_vtp_points_U(vtk_path, field="U")
        les_ws = np.hypot(vec[:, 0], vec[:, 1])
        les_g = griddata(pts[:, :2], les_ws, (XX, YY), method="linear")
        vmax = float(np.nanpercentile(np.concatenate([les_ws, rWS.ravel()]), 99.5))
        vmax = min(max(vmax, 1.2), 8.0)
        norm = Normalize(vmin=0.0, vmax=vmax)
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True)
        im = panel(axes[0], rXX, rYY, rWS, "RANS mean  (simpleFoam, pseudo-t=5000)", norm)
        panel(
            axes[1],
            XX,
            YY,
            les_g,
            f"DFSEM snapshot (WALE, t={args.time} s)",
            norm,
        )
        axes[0].set_ylabel("y (m)")
        cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
        cb.set_label("horizontal wind speed (m s$^{-1}$)")
        fig.suptitle(
            f"2025-09-01 00:00 UTC   z = 100 m   |   DFSEM instantaneous vs RANS (t={args.time} s)",
            fontsize=11,
        )
        out = OUTDIR / f"xy_ws_z100_rans_vs_dfsem_t{args.time}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")
        print(
            f"DFSEM n={len(les_ws)}  ws min/p50/max "
            f"{les_ws.min():.2f}/{np.median(les_ws):.2f}/{les_ws.max():.2f}"
        )
        return

    dfsem_vtp = CASE / "postProcessing" / "surfacesUMean" / args.time / "z100.vtp"
    pts_d, vec_d = read_vtp_points_U(dfsem_vtp, field="UMean")
    d_ws = np.hypot(vec_d[:, 0], vec_d[:, 1])
    d_g = griddata(pts_d[:, :2], d_ws, (XX, YY), method="linear")

    if args.mode == "umean":
        vmax = float(np.nanpercentile(np.concatenate([d_ws, rWS.ravel()]), 99.5))
        vmax = min(max(vmax, 1.2), 8.0)
        norm = Normalize(vmin=0.0, vmax=vmax)
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True)
        im = panel(axes[0], rXX, rYY, rWS, "RANS mean  (simpleFoam, pseudo-t=5000)", norm)
        panel(
            axes[1],
            XX,
            YY,
            d_g,
            f"DFSEM UMean (avg 0–{args.time} s)",
            norm,
        )
        axes[0].set_ylabel("y (m)")
        cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
        cb.set_label("horizontal wind speed (m s$^{-1}$)")
        fig.suptitle(
            f"2025-09-01 00:00 UTC   z = 100 m   |   DFSEM time mean vs RANS (t={args.time} s)",
            fontsize=11,
        )
        out = OUTDIR / f"xy_ws_z100_rans_vs_dfsem_umean_t{args.time}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")
        print(
            f"DFSEM UMean n={len(d_ws)}  ws min/p50/max "
            f"{d_ws.min():.2f}/{np.median(d_ws):.2f}/{d_ws.max():.2f}"
        )
        return

    mean_vtp = MEAN / "postProcessing" / "surfacesUMean" / "300" / "z100.vtp"
    pts_m, vec_m = read_vtp_points_U(mean_vtp, field="UMean")
    m_ws = np.hypot(vec_m[:, 0], vec_m[:, 1])
    m_g = griddata(pts_m[:, :2], m_ws, (XX, YY), method="linear")
    vmax = float(
        np.nanpercentile(np.concatenate([d_ws, m_ws, rWS.ravel()]), 99.5)
    )
    vmax = min(max(vmax, 1.2), 8.0)
    norm = Normalize(vmin=0.0, vmax=vmax)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0), constrained_layout=True)
    im = panel(axes[0], rXX, rYY, rWS, "RANS mean", norm)
    panel(axes[1], XX, YY, m_g, "mean-inlet LES UMean\n(avg 100–300 s)", norm)
    panel(axes[2], XX, YY, d_g, f"DFSEM UMean\n(avg 0–{args.time} s)", norm)
    axes[0].set_ylabel("y (m)")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("horizontal wind speed (m s$^{-1}$)")
    fig.suptitle(
        "2025-09-01 00:00 UTC   z = 100 m   |   RANS vs mean-inlet UMean vs DFSEM UMean",
        fontsize=11,
    )
    out = OUTDIR / f"xy_ws_z100_rans_meaninlet_dfsem_umean_t{args.time}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    print(
        f"DFSEM UMean p50={np.median(d_ws):.3f}  "
        f"mean-inlet UMean p50={np.median(m_ws):.3f}  "
        f"RANS p50={np.nanmedian(rWS):.3f}"
    )


if __name__ == "__main__":
    main()
