#!/usr/bin/env python3
"""LES instantaneous vs RANS mean at z=100 m. Output stays in this LES case."""
from __future__ import annotations

import argparse
import base64
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

CASE = Path(__file__).resolve().parent
REPO = CASE.parents[2]
RANS_CSV = (
    REPO
    / "steady_experiments_finer_ABL"
    / "20250901_0000_two_boundaries_as_outlet"
    / "postProcessing"
    / "100m.csv"
)
OUTDIR = CASE / "figs_les_vs_rans"
HOTSPOT = (1585.56, -1397.21)
CASE_UTC = datetime(2025, 9, 1, 0, 0, tzinfo=timezone.utc)
CASE_BJT_STAMP = CASE_UTC.astimezone(timezone(timedelta(hours=8))).strftime(
    "%Y-%m-%d %H:%M BJT"
)


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
    u = vtk_xml_binary_array(u_el, n, 3, "Float32")
    return pts, u


def load_rans_100m():
    data = np.loadtxt(RANS_CSV, delimiter=",", skiprows=1, usecols=(0, 1, 9, 10))
    u, v, x, y = data.T
    nx = ny = 1000
    # x varies fastest (seen in file preview)
    XX = x.reshape(ny, nx)
    YY = y.reshape(ny, nx)
    WS = np.hypot(u, v).reshape(ny, nx)
    return XX, YY, WS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", default="100")
    ap.add_argument("--umean", action="store_true")
    ap.add_argument(
        "--mark-limitu",
        action="store_true",
        help="mark the limitU hotspot cell (off by default)",
    )
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.umean:
        vtk_path = CASE / "postProcessing" / "surfacesUMean" / args.time / "z100.vtp"
        field = "UMean"
        les_title = f"b) LES (WALE, avg from 100 s to {args.time} s)"
        out = CASE / "figs_les_300s" / f"xy_ws_z100_rans_vs_umean_t{args.time}.png"
    else:
        vtk_path = CASE / "postProcessing" / "surfaces" / args.time / "z100.vtp"
        field = "U"
        les_title = f"b) LES (WALE, t={args.time} s)"
        out = OUTDIR / f"xy_ws_z100_rans_vs_les_t{args.time}.png"
    sup = f"{CASE_BJT_STAMP} | z = 100 m"

    out.parent.mkdir(parents=True, exist_ok=True)
    pts, vec = read_vtp_points_U(vtk_path, field=field)
    les_ws = np.hypot(vec[:, 0], vec[:, 1])

    rXX, rYY, rWS = load_rans_100m()
    xi = rXX[0]
    yi = rYY[:, 0]
    XX, YY = np.meshgrid(xi, yi)
    les_g = griddata(pts[:, :2], les_ws, (XX, YY), method="linear")

    vmax = float(np.nanpercentile(np.concatenate([les_ws, rWS.ravel()]), 99.5))
    vmax = min(max(vmax, 1.2), 8.0)
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True)
    titles = (
        "a) RANS",
        les_title,
    )
    for ax, title, zz, xg, yg in zip(
        axes, titles, (rWS, les_g), (rXX, XX), (rYY, YY)
    ):
        im = ax.pcolormesh(xg, yg, zz, shading="auto", cmap="viridis", norm=norm)
        ax.set_aspect("equal")
        ax.set_xlim(-2500, 2500)
        ax.set_ylim(-2500, 2500)
        ax.set_xlabel("x (m)")
        ax.set_title(title, loc='left', fontsize=10)
        if args.mark_limitu:
            ax.plot(*HOTSPOT, "r+", ms=8, mew=1.2)
    axes[0].set_ylabel("y (m)")
    if args.mark_limitu:
        axes[1].plot(*HOTSPOT, "r+", ms=8, mew=1.2, label="limitU cell (z=232 m)")
        axes[1].legend(loc="upper right", fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("horizontal wind speed (m s$^{-1}$)")
    fig.suptitle(sup, fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    print(
        f"LES n={len(les_ws)}  ws min/p50/max "
        f"{les_ws.min():.2f}/{np.median(les_ws):.2f}/{les_ws.max():.2f}"
    )
    print(
        f"RANS n={rWS.size}  ws min/p50/max "
        f"{rWS.min():.2f}/{np.nanmedian(rWS):.2f}/{rWS.max():.2f}"
    )


if __name__ == "__main__":
    main()
