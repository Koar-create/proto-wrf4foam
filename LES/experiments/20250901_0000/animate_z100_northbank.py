#!/usr/bin/env python3
"""North-bank 1 km animation of z=100 m horizontal wind speed.

Window: x in [-200, 800], y in [-50, 950] (contains GAW104 and GAW111).
Interpolation ~5 m. Instantaneous Ws plus Ws minus the sequence time-mean,
so local wake jitter is visible even when the mean field looks still.
"""
from __future__ import annotations

import argparse
import base64
import os
import struct
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from scipy.interpolate import griddata

REPO = Path("/hpc2hdd/home/zyang248/WRF-OpenFOAM-Coupling")
CASE = REPO / "LES" / "20250901_0000_WN_outlet"
SURF = CASE / "postProcessing" / "surfacesU"
STL = CASE / "constant" / "triSurface" / "buildings.stl"
OUT = REPO / "results" / "les" / "z100_northbank_1km"

XMIN, XMAX = -200.0, 800.0
YMIN, YMAX = -50.0, 950.0
DX = 5.0
Z_SLICE = 100.0
SITES = {"GAW104": (450.0, 350.0), "GAW111": (75.0, 30.0)}
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))


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


def list_times(surf: Path) -> list[int]:
    return sorted(int(p.name) for p in surf.iterdir() if p.name.isdigit())


def load_ws_grid(times: list[int], xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    pad = 80.0
    stack = np.full((len(times),) + xx.shape, np.nan, dtype=np.float32)
    for i, t in enumerate(times):
        path = SURF / str(t) / "z100.vtp"
        pts, vec = read_vtp_points_U(path)
        keep = (
            (pts[:, 0] >= XMIN - pad)
            & (pts[:, 0] <= XMAX + pad)
            & (pts[:, 1] >= YMIN - pad)
            & (pts[:, 1] <= YMAX + pad)
        )
        xy = pts[keep, :2]
        ws = np.hypot(vec[keep, 0], vec[keep, 1])
        stack[i] = griddata(xy, ws, (xx, yy), method="linear")
        print(
            f"t={t:3d}  n={int(keep.sum())}  "
            f"p50={np.nanmedian(stack[i]):.3f}  "
            f"p99={np.nanpercentile(stack[i], 99):.3f}",
            flush=True,
        )
    return stack


def stl_xy_at_z(stl_path: Path, z0: float) -> list[np.ndarray]:
    """xy triangles whose z-range contains z0 (buildings piercing the slice)."""
    if not stl_path.is_file():
        return []
    polys = []
    with stl_path.open("rb") as f:
        header = f.read(80)
        if header.lstrip().lower().startswith(b"solid"):
            print(f"[warn] ASCII STL not supported: {stl_path}")
            return []
        n_tri = struct.unpack("<I", f.read(4))[0]
        for _ in range(n_tri):
            data = f.read(50)
            if len(data) < 48:
                break
            tri = np.array(
                [
                    struct.unpack("<3f", data[12:24]),
                    struct.unpack("<3f", data[24:36]),
                    struct.unpack("<3f", data[36:48]),
                ],
                dtype=np.float32,
            )
            zmin = float(tri[:, 2].min())
            zmax = float(tri[:, 2].max())
            if zmin > z0 or zmax < z0:
                continue
            xy = tri[:, :2]
            if (
                xy[:, 0].max() < XMIN
                or xy[:, 0].min() > XMAX
                or xy[:, 1].max() < YMIN
                or xy[:, 1].min() > YMAX
            ):
                continue
            polys.append(xy)
    print(f"[info] STL ∩ z={z0:.0f} m in window: {len(polys)} triangles")
    return polys


def mark_sites(ax):
    for name, (x, y) in SITES.items():
        ax.plot(x, y, "ko", ms=5, zorder=6)
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            zorder=6,
        )


def add_buildings(ax, polys):
    if not polys:
        return
    ax.add_collection(
        PolyCollection(
            polys,
            facecolor="#3a3a3a",
            edgecolor="#1a1a1a",
            linewidths=0.15,
            alpha=0.85,
            zorder=4,
        )
    )


def draw_panel(ax, xx, yy, zz, title, norm, cmap, polys, cbar_label=None):
    im = ax.pcolormesh(xx, yy, zz, shading="auto", cmap=cmap, norm=norm, zorder=1)
    building = ~np.isfinite(zz)
    if np.any(building):
        ax.pcolormesh(
            xx,
            yy,
            np.ma.masked_where(~building, np.ones_like(zz)),
            shading="auto",
            cmap=ListedColormap(["#2f2f2f"]),
            vmin=0,
            vmax=1,
            zorder=3,
        )
    add_buildings(ax, polys)
    mark_sites(ax)
    ax.set_aspect("equal")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel("x (m)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xticks([-200, 0, 200, 400, 600, 800])
    ax.set_yticks([0, 200, 400, 600, 800])
    ax.tick_params(direction="out", length=3)
    ax.grid(True, ls=":", lw=0.35, alpha=0.4, zorder=0)
    return im


def site_series(stack, xx, yy):
    x1d = xx[0]
    y1d = yy[:, 0]
    out = {}
    for name, (x, y) in SITES.items():
        ix = int(np.argmin(np.abs(x1d - x)))
        iy = int(np.argmin(np.abs(y1d - y)))
        out[name] = stack[:, iy, ix]
    return out


def write_summary(path: Path, times, stack, mean, rms, series):
    lines = [
        "mean-inlet LES  z=100 m  north-bank 1 km",
        f"window x=[{XMIN:.0f},{XMAX:.0f}]  y=[{YMIN:.0f},{YMAX:.0f}]  dx={DX:.0f} m",
        f"times {times[0]}:{times[1]-times[0]}:{times[-1]}  n={len(times)}",
        f"Ws time-mean  p50={np.nanmedian(mean):.3f}  p99={np.nanpercentile(mean, 99):.3f}",
        f"Ws' RMS       p50={np.nanmedian(rms):.4f}  p99={np.nanpercentile(rms, 99):.4f}  max={np.nanmax(rms):.4f}",
        "",
        "station  x  y  Ws_mean  Ws_rms  Ws_min  Ws_max",
    ]
    for name, (x, y) in SITES.items():
        s = series[name]
        lines.append(
            f"{name}  {x:.0f}  {y:.0f}  {np.nanmean(s):.4f}  "
            f"{np.nanstd(s):.4f}  {np.nanmin(s):.4f}  {np.nanmax(s):.4f}"
        )
    path.write_text("\n".join(lines) + "\n")
    print(path.read_text())


def ffmpeg_gif_mp4(frame_dir: Path, stem: Path, n: int, fps: int):
    pattern = str(frame_dir / "frame_%03d.png")
    gif = stem.with_suffix(".gif")
    mp4 = stem.with_suffix(".mp4")
    palette = stem.parent / f"{stem.name}_palette.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-vf",
            "palettegen=stats_mode=full",
            str(palette),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-i",
            str(palette),
            "-lavfi",
            "paletteuse=dither=bayer:bayer_scale=5",
            "-loop",
            "0",
            str(gif),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    palette.unlink(missing_ok=True)
    print(f"wrote {gif}  {mp4}  ({n} frames, {fps} fps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ".mplconfig").mkdir(parents=True, exist_ok=True)
    frames = OUT / "frames"
    frames.mkdir(exist_ok=True)

    times = list_times(SURF)
    if not times:
        sys.exit(f"no surfacesU times in {SURF}")
    xi = np.arange(XMIN, XMAX + 0.5 * DX, DX)
    yi = np.arange(YMIN, YMAX + 0.5 * DX, DX)
    xx, yy = np.meshgrid(xi, yi)
    print(f"grid {xi.size} x {yi.size}  dx={DX} m  ntimes={len(times)}")

    stack = load_ws_grid(times, xx, yy)
    mean = np.nanmean(stack, axis=0)
    prime = stack - mean
    rms = np.nanstd(stack, axis=0)
    series = site_series(stack, xx, yy)
    write_summary(OUT / "summary.txt", times, stack, mean, rms, series)

    ws_vmax = float(np.nanpercentile(stack, 99.5))
    ws_vmax = min(max(ws_vmax, 0.8), 4.0)
    ws_norm = Normalize(vmin=0.0, vmax=ws_vmax)
    pr_lim = float(np.nanpercentile(np.abs(prime), 99))
    pr_lim = max(pr_lim, 0.02)
    pr_norm = TwoSlopeNorm(vmin=-pr_lim, vcenter=0.0, vmax=pr_lim)
    polys = stl_xy_at_z(STL, Z_SLICE)

    for i, t in enumerate(times):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.35), constrained_layout=True)
        im0 = draw_panel(
            axes[0],
            xx,
            yy,
            stack[i],
            f"a) instantaneous $W_s$   t = {t} s",
            ws_norm,
            "viridis",
            polys,
        )
        im1 = draw_panel(
            axes[1],
            xx,
            yy,
            prime[i],
            r"b) $W_s - \langle W_s\rangle$   (local jitter)",
            pr_norm,
            "RdBu_r",
            polys,
        )
        axes[0].set_ylabel("y (m)")
        cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
        cb0.set_label(r"$W_s$ (m s$^{-1}$)")
        cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
        cb1.set_label(r"$W_s'$ (m s$^{-1}$)")
        fig.suptitle(
            "Mean-inlet LES  |  z = 100 m  |  north bank 1 km  |  08:00 BJT",
            fontsize=11,
        )
        fig.savefig(frames / f"frame_{i:03d}.png", dpi=130)
        plt.close(fig)
        print(f"frame {i:03d}  t={t}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.35), constrained_layout=True)
    rms_vmax = float(np.nanpercentile(rms, 99.5))
    rms_vmax = max(rms_vmax, 0.02)
    im0 = draw_panel(
        axes[0], xx, yy, mean, r"a) time-mean $W_s$  (105–300 s)", ws_norm, "viridis", polys
    )
    im1 = draw_panel(
        axes[1],
        xx,
        yy,
        rms,
        r"b) RMS of $W_s'$  (wake jitter map)",
        Normalize(vmin=0.0, vmax=rms_vmax),
        "magma",
        polys,
    )
    axes[0].set_ylabel("y (m)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03).set_label(r"$W_s$ (m s$^{-1}$)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03).set_label(r"RMS (m s$^{-1}$)")
    fig.suptitle(
        "Mean-inlet LES  |  z = 100 m  |  north bank 1 km  |  08:00 BJT",
        fontsize=11,
    )
    static = OUT / "ws_mean_and_rms.png"
    fig.savefig(static, dpi=160)
    plt.close(fig)
    print(f"wrote {static}")

    ffmpeg_gif_mp4(frames, OUT / "ws_jitter", len(times), args.fps)


if __name__ == "__main__":
    main()
