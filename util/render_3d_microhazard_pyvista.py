#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage A: locate a near-surface wind-speed hotspot from postProcessing/100m.csv
         (P99 within urban core + density clustering).

Stage B: render a zoomed 3D scene (buildings + horizontal slice + streamlines;
         optional |U| isosurfaces) with PyVista. Canyon-focused defaults follow
         docs/ops/改进RBM-3D可视化-2.md / 改进RBM-3D可视化-3.md: Z capped for
         analysis (--analysis-z-max-m), clim from p_low–p_high percentiles,
         slice cell→point smoothing, colored tubes by |U|. By default PyVista failure exits
         with an error (roadshow-safe: no silent 2D regression).

Default case: steady_experiments_finer_ABL/20250903_1400_two_boundaries_as_outlet

Usage:
  python util/render_3d_microhazard_pyvista.py --case-dir <path/to/case>
  python util/render_3d_microhazard_pyvista.py --window-size 3840,2160 --iso-percentiles 95,98
  python util/render_3d_microhazard_pyvista.py --no-require-3d   # allow matplotlib on failure
  python util/render_3d_microhazard_pyvista.py --fallback-only   # debug: 2D panels only
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hotspot locate + 3D micro-hazard render (PyVista; optional matplotlib only when allowed)."
    )
    p.add_argument(
        "--case-dir",
        type=Path,
        default=_repo_root()
        / "steady_experiments_finer_ABL"
        / "20250903_1400_two_boundaries_as_outlet",
        help="OpenFOAM case directory (contains myExpxx.foam, postProcessing/, constant/).",
    )
    p.add_argument(
        "--buildings-stl",
        type=Path,
        default=_repo_root() / "constant" / "triSurface" / "buildings.stl",
        help="Path to buildings STL.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_repo_root()
        / "results"
        / "microhazard"
        / "20250903_1400"
        / "figure1_streamlines_buildings.png",
        help="Output PNG.",
    )
    p.add_argument("--urban-core-m", type=float, default=1500.0, help="(|x|,|y|) < core for hotspot search.")
    p.add_argument("--p-percentile", type=float, default=99.0, help="Percentile threshold on |U_h|.")
    p.add_argument("--dbscan-eps-m", type=float, default=80.0, help="DBSCAN eps (meters) if sklearn is available.")
    p.add_argument("--dbscan-min-samples", type=int, default=5)
    p.add_argument(
        "--half-width-m",
        type=float,
        default=200.0,
        help="Horizontal half-width (m) of the clip box around the hotspot; larger = more city in frame "
        "(also scales camera baseline). Example: 350–600 for wide context.",
    )
    p.add_argument(
        "--zmax-m",
        type=float,
        default=300.0,
        help="Requested vertical clip top (m); effective top is min(zmax, --analysis-z-max-m).",
    )
    p.add_argument(
        "--analysis-z-max-m",
        type=float,
        default=250.0,
        help="Cap Z for CFD clip / clim / isosurfaces / seeds so highs are canyon-layer not "
        "free troposphere (see docs/ops/改进RBM-3D可视化-3.md).",
    )
    p.add_argument("--slice-z-m", type=float, default=50.0, help="Horizontal slice height for coloring.")
    p.add_argument(
        "--window-size",
        type=str,
        default="1920,1280",
        help="PyVista off-screen window as W,H pixels. For slides use e.g. 3840,2160 or higher.",
    )
    p.add_argument(
        "--require-3d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set (default), PyVista import/read/render errors abort with no matplotlib fallback. "
        "Use --no-require-3d to allow 2D fallback when PyVista fails.",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="turbo",
        help="Matplotlib colormap name for slice, streamlines, and isosurfaces (e.g. turbo, inferno, hot).",
    )
    p.add_argument(
        "--clim-low-percentile",
        type=float,
        default=5.0,
        help="Lower color limit = this percentile of |U| in the analysis clip (canyon layer).",
    )
    p.add_argument(
        "--clim-percentile",
        type=float,
        default=98.0,
        help="Upper color limit = this percentile of |U| in the analysis clip (same volume as lower).",
    )
    p.add_argument(
        "--iso-percentiles",
        type=str,
        default="",
        help="Optional comma-separated percentiles for |U| isosurfaces in the clipped volume (e.g. 95,98). Empty disables.",
    )
    p.add_argument(
        "--iso-opacity",
        type=float,
        default=0.24,
        help="Opacity for each |U| isosurface (see docs/ops/改进RBM-3D可视化-2.md).",
    )
    p.add_argument(
        "--seed-box-m",
        type=float,
        default=50.0,
        help="Half-span (m) in x and y for the 3D streamline seed grid centered on the hotspot.",
    )
    p.add_argument(
        "--seed-z-max-m",
        type=float,
        default=0.0,
        help="Top height (m) of seed grid; 0 = auto min(150, ~0.58×effective Z top). Bottom ~10 m.",
    )
    p.add_argument("--seed-nx", type=int, default=5, help="Seed grid count in x.")
    p.add_argument("--seed-ny", type=int, default=5, help="Seed grid count in y.")
    p.add_argument("--seed-nz", type=int, default=6, help="Seed grid count in z.")
    p.add_argument("--no-edl", action="store_true", help="Disable Eye Dome Lighting (depth cue).")
    p.add_argument(
        "--building-wind",
        action="store_true",
        help="Sample |U| onto building surfaces (default: solid light-gray buildings, no edges).",
    )
    p.add_argument(
        "--tube-radius-m",
        type=float,
        default=0.9,
        help="Streamline tube radius (m); ~0.8–1.0 reads as 3D ribbons in ~400 m views (doc-3).",
    )
    p.add_argument(
        "--no-streamlines",
        action="store_true",
        help="Skip streamline tubes (e.g. isosurface-only figure).",
    )
    p.add_argument(
        "--camera-distance-factor",
        type=float,
        default=1.0,
        help="Multiply camera offset from (x0,y0) for a wider / pulled-back view without changing the "
        "CFD clip. Try 1.3–1.8. Clip box unchanged; only the viewpoint moves back.",
    )
    p.add_argument(
        "--fallback-only",
        action="store_true",
        help="Skip PyVista; matplotlib 2-panel debug only. Do not use for roadshow Figure 1.",
    )
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _parse_iso_percentiles(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _read_100m_windspeed(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y, |U_h| as float64 arrays (full file)."""
    xs, ys, ws = [], [], []
    req = ["Coords:0", "Coords:1", "U:0", "U:1"]
    for chunk in pd.read_csv(csv_path, chunksize=200_000):
        miss = [c for c in req if c not in chunk.columns]
        if miss:
            raise KeyError(f"Missing columns {miss} in {csv_path}")
        u0 = chunk["U:0"].astype(np.float64).to_numpy()
        u1 = chunk["U:1"].astype(np.float64).to_numpy()
        xs.append(chunk["Coords:0"].astype(np.float64).to_numpy())
        ys.append(chunk["Coords:1"].astype(np.float64).to_numpy())
        ws.append(np.sqrt(u0 * u0 + u1 * u1))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    return x, y, w


def _cluster_hotspot_xy(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    urban_core_m: float,
    p_pct: float,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> Tuple[float, float, float, int]:
    """Return (x0, y0, p_threshold, n_candidates)."""
    core = (np.abs(x) < urban_core_m) & (np.abs(y) < urban_core_m)
    if not np.any(core):
        raise RuntimeError("Urban core mask is empty; check coordinates / urban_core_m.")

    thr = float(np.percentile(w[core], p_pct))
    cand = core & (w >= thr)
    n_cand = int(np.sum(cand))
    if n_cand < 10:
        # fallback: use global max within core
        idx = int(np.argmax(np.where(core, w, -np.inf)))
        return float(x[idx]), float(y[idx]), thr, int(np.sum(core))

    xc, yc, wc = x[cand], y[cand], w[cand]

    try:
        from sklearn.cluster import DBSCAN  # type: ignore

        xy = np.column_stack([xc, yc])
        lab = DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min_samples)).fit_predict(xy)
        best_label = None
        best_count = -1
        for lbl in sorted(set(lab)):
            if lbl == -1:
                continue
            c = int(np.sum(lab == lbl))
            if c > best_count:
                best_count = c
                best_label = lbl
        if best_label is None:
            raise RuntimeError("DBSCAN produced no clusters; falling back to grid mode.")
        m = lab == best_label
        x0 = float(np.mean(xc[m]))
        y0 = float(np.mean(yc[m]))
        return x0, y0, thr, n_cand
    except Exception:
        # Grid mode (80 m bins), pick densest cell centroid
        eps = float(dbscan_eps)
        ix = np.floor((xc - np.min(xc)) / eps).astype(int)
        iy = np.floor((yc - np.min(yc)) / eps).astype(int)
        keys = ix * (iy.max() + 5) + iy
        uniq, inv = np.unique(keys, return_inverse=True)
        counts = np.bincount(inv)
        best_j = int(np.argmax(counts))
        sel = inv == best_j
        x0 = float(np.mean(xc[sel]))
        y0 = float(np.mean(yc[sel]))
        return x0, y0, thr, n_cand


def _add_u_mag(mesh) -> None:
    keys = list(mesh.array_names)
    if "U" in keys:
        U = np.asarray(mesh["U"])
        if U.ndim == 2 and U.shape[1] >= 3:
            mesh["U_mag"] = np.linalg.norm(U[:, :3], axis=1)
        elif U.ndim == 2 and U.shape[1] == 2:
            mesh["U_mag"] = np.linalg.norm(U, axis=1)
        else:
            mesh["U_mag"] = np.abs(U).ravel()
    elif "UMean" in keys:
        U = np.asarray(mesh["UMean"])
        mesh["U_mag"] = np.linalg.norm(U[:, :3], axis=1) if U.ndim == 2 else np.abs(U).ravel()


def _pick_internal_block(mb):
    import pyvista as pv

    if isinstance(mb, pv.PolyData):
        return mb
    # MultiBlock
    for i in range(mb.n_blocks):
        b = mb.get_block(i)
        if b is None:
            continue
        name = (mb.get_block_name(i) or "").lower()
        if "internal" in name:
            return b
    # fallback: first non-empty PolyData/UnstructuredGrid
    for i in range(mb.n_blocks):
        b = mb.get_block(i)
        if b is not None and b.n_cells > 0:
            return b
    raise RuntimeError("Could not find internal mesh block in OpenFOAM reader output.")


def _hotspot_seed_cloud(
    x0: float,
    y0: float,
    *,
    half_box: float,
    zmin: float,
    ztop: float,
    nx: int,
    ny: int,
    nz: int,
):
    """3D seed grid around the hotspot (see docs/ops/改进RBM-3D可视化.md)."""
    import pyvista as pv

    ztop = float(max(zmin + 5.0, ztop))
    xs = np.linspace(x0 - half_box, x0 + half_box, max(2, int(nx)))
    ys = np.linspace(y0 - half_box, y0 + half_box, max(2, int(ny)))
    zs = np.linspace(float(zmin), float(ztop), max(2, int(nz)))
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    return pv.PolyData(pts)


def _slice_cell_to_point_if_needed(sl, pv):
    """Reduce blocky cell-wise coloring on slices (docs/ops/改进RBM-3D可视化-3.md)."""
    if sl is None or sl.n_cells == 0:
        return sl
    if sl.cell_data.keys() and ("U" in sl.cell_data or "U_mag" in sl.cell_data):
        return sl.cell_data_to_point_data(pass_cell_data=False)
    return sl


def _render_pyvista(
    *,
    case_dir: Path,
    buildings_stl: Path,
    out_png: Path,
    x0: float,
    y0: float,
    half_w: float,
    zmax: float,
    slice_z: float,
    window_size: Tuple[int, int],
    cmap: str,
    clim_low_percentile: float,
    clim_percentile: float,
    analysis_z_max_m: float,
    iso_percentiles: List[float],
    iso_opacity: float,
    seed_box_m: float,
    seed_z_max_m: float,
    seed_nx: int,
    seed_ny: int,
    seed_nz: int,
    use_edl: bool,
    building_wind_sampling: bool,
    tube_radius_m: float,
    show_streamlines: bool,
    camera_distance_factor: float,
) -> None:
    import pyvista as pv

    foam = case_dir / "myExpxx.foam"
    if not foam.is_file():
        raise FileNotFoundError(f"Missing OpenFOAM case marker: {foam}")

    reader = pv.POpenFOAMReader(str(foam))
    # Prefer last time value (latest convergence field)
    tvals = list(reader.time_values)
    if not tvals:
        raise RuntimeError("OpenFOAM reader returned no time values.")
    reader.set_active_time_value(float(tvals[-1]))
    mb = reader.read()
    internal = _pick_internal_block(mb)
    internal = internal.copy(deep=True)

    # Ensure point vectors for streamlines / slicing
    if "U" in internal.array_names and internal.n_cells > 0 and "U" in internal.cell_data:
        internal = internal.cell_data_to_point_data(pass_cell_data=False)

    _add_u_mag(internal)

    xmin, xmax = x0 - half_w, x0 + half_w
    ymin, ymax = y0 - half_w, y0 + half_w
    z_top = min(float(zmax), float(analysis_z_max_m))
    z_top = max(30.0, z_top)
    zmin, zmax_clip = 0.0, z_top
    bounds = (xmin, xmax, ymin, ymax, zmin, zmax_clip)

    clipped = internal.clip_box(bounds=bounds, invert=False)
    if clipped.n_cells == 0:
        raise RuntimeError("clip_box removed all cells; check hotspot / bounds.")

    bmesh = pv.read(str(buildings_stl))
    bclip = bmesh.clip_box(bounds=bounds, invert=False)

    z_seed_top = float(seed_z_max_m) if seed_z_max_m > 0.0 else min(150.0, zmax_clip * 0.58)
    z_seed_top = min(z_seed_top, zmax_clip - 2.0)
    z_seed_bot = min(10.0, max(5.0, z_seed_top * 0.12))
    if show_streamlines:
        seed_pts = _hotspot_seed_cloud(
            x0,
            y0,
            half_box=float(seed_box_m),
            zmin=z_seed_bot,
            ztop=z_seed_top,
            nx=int(seed_nx),
            ny=int(seed_ny),
            nz=int(seed_nz),
        )

        _sl_sig = inspect.signature(clipped.streamlines_from_source)
        _stream_kw: dict = {
            "vectors": "U",
            "integration_direction": "both",
            "initial_step_length": 0.2,
            "max_step_length": 1.0,
            "max_steps": 4000,
            "terminal_speed": 0.015,
            "interpolator_type": "point",
        }
        if "max_length" in _sl_sig.parameters:
            _stream_kw["max_length"] = 150.0
        elif "max_time" in _sl_sig.parameters:
            _stream_kw["max_time"] = 150.0

        streams = clipped.streamlines_from_source(seed_pts, **_stream_kw)
        if streams.n_points == 0:
            seeds_line = pv.Line(
                (xmin, y0, 8.0),
                (xmin, y0, min(120.0, zmax_clip - 5.0)),
                resolution=28,
            )
            streams = clipped.streamlines_from_source(seeds_line, **_stream_kw)

        if streams.n_points > 0 and "U" in streams.array_names:
            Uv = np.asarray(streams["U"])
            if Uv.ndim == 2 and Uv.shape[1] >= 3:
                streams["U_mag"] = np.linalg.norm(Uv[:, :3], axis=1)
            else:
                streams["U_mag"] = np.abs(Uv).ravel()
    else:
        streams = pv.PolyData()

    # Color limits from canyon-layer clip only (not whole ABL column — doc-3)
    wmag = np.asarray(clipped["U_mag"], dtype=float)
    p_lo = float(np.clip(clim_low_percentile, 0.0, 49.5))
    p_hi = float(np.clip(clim_percentile, p_lo + 1.0, 99.999))
    v_lo = float(np.nanpercentile(wmag, p_lo))
    v_hi = float(np.nanpercentile(wmag, p_hi))
    if not (np.isfinite(v_lo) and np.isfinite(v_hi)) or v_hi <= v_lo:
        v_lo, v_hi = 0.0, float(np.nanpercentile(wmag, 98.0))
    clim = (v_lo, v_hi)

    sl_main = clipped.slice(normal="z", origin=(x0, y0, slice_z))
    sl_main = _slice_cell_to_point_if_needed(sl_main, pv)
    _add_u_mag(sl_main)
    cname = "U_mag" if "U_mag" in sl_main.array_names else None
    if cname is None and "U" in sl_main.array_names:
        U = np.asarray(sl_main["U"])
        sl_main["U_mag"] = np.linalg.norm(U[:, :3], axis=1) if U.ndim == 2 else np.abs(U).ravel()
        cname = "U_mag"

    sl_low = None
    if zmax_clip > 18.0:
        sl_try = clipped.slice(normal="z", origin=(x0, y0, 20.0))
        if sl_try.n_cells > 0:
            sl_try = _slice_cell_to_point_if_needed(sl_try, pv)
            _add_u_mag(sl_try)
            sl_low = sl_try

    w_w, h_h = window_size
    plotter = pv.Plotter(off_screen=True, window_size=(int(w_w), int(h_h)))

    if use_edl:
        try:
            plotter.enable_eye_dome_lighting()
        except Exception:
            pass

    # Optional 3D |U| isosurfaces (draw under slices / buildings)
    for ip in iso_percentiles:
        ip = float(np.clip(ip, 0.0, 100.0))
        iso_val = float(np.nanpercentile(wmag, ip))
        if not np.isfinite(iso_val) or iso_val <= 0.0:
            continue
        try:
            iso_surf = clipped.contour(isosurfaces=[iso_val], scalars="U_mag")
        except Exception:
            continue
        if iso_surf.n_cells == 0:
            continue
        plotter.add_mesh(
            iso_surf,
            scalars="U_mag",
            cmap=cmap,
            clim=clim,
            opacity=float(iso_opacity),
            smooth_shading=True,
            show_scalar_bar=False,
        )

    if sl_low is not None and abs(float(slice_z) - 20.0) > 4.0:
        plotter.add_mesh(
            sl_low,
            scalars="U_mag",
            cmap=cmap,
            clim=clim,
            opacity=0.48,
            smooth_shading=True,
            show_scalar_bar=False,
        )

    bar_on_tubes = bool(show_streamlines and streams.n_points > 0)
    _sl_kw = dict(
        scalars=cname,
        cmap=cmap,
        clim=clim,
        opacity=0.55,
        smooth_shading=True,
        show_scalar_bar=not bar_on_tubes,
    )
    if not bar_on_tubes:
        _sl_kw["scalar_bar_args"] = dict(
            title="Wind speed |U| (m/s)",
            n_labels=6,
            color="black",
            title_font_size=20,
            label_font_size=16,
        )
    plotter.add_mesh(sl_main, **_sl_kw)

    b_draw = bclip
    b_colored = False
    if building_wind_sampling:
        try:
            b_s = bclip.sample(clipped)
            if b_s.n_points > 0:
                _add_u_mag(b_s)
                if "U_mag" in b_s.array_names or "U" in b_s.array_names:
                    if "U_mag" not in b_s.array_names:
                        _add_u_mag(b_s)
                    b_draw = b_s
                    b_colored = "U_mag" in b_draw.array_names
        except Exception:
            b_draw = bclip
            b_colored = False

    if b_colored:
        plotter.add_mesh(
            b_draw,
            scalars="U_mag",
            cmap=cmap,
            clim=clim,
            smooth_shading=True,
            opacity=1.0,
            show_edges=False,
            show_scalar_bar=False,
        )
    else:
        plotter.add_mesh(
            b_draw,
            color="#d3d3d3",
            smooth_shading=True,
            opacity=1.0,
            show_edges=False,
        )

    tube_r = float(max(0.15, min(1.6, tube_radius_m)))
    if show_streamlines and streams.n_points > 0:
        if "U_mag" not in streams.array_names and "U" in streams.array_names:
            Uv = np.asarray(streams["U"])
            if Uv.ndim == 2 and Uv.shape[1] >= 3:
                streams["U_mag"] = np.linalg.norm(Uv[:, :3], axis=1)
            else:
                streams["U_mag"] = np.abs(Uv).ravel()
        tubes = streams.tube(radius=tube_r, n_sides=32)
        if "U_mag" in tubes.array_names:
            tubes.set_active_scalars("U_mag")
        plotter.add_mesh(
            tubes,
            scalars="U_mag" if "U_mag" in tubes.array_names else None,
            cmap=cmap,
            clim=clim,
            smooth_shading=True,
            show_scalar_bar=True,
            scalar_bar_args=dict(
                title="Wind speed |U| (m/s)",
                n_labels=6,
                color="black",
                title_font_size=20,
                label_font_size=16,
            ),
        )

    plotter.add_text(
        "2025-09-03 22:00 LST  (14:00 UTC)\nNocturnal ABL — urban Venturi / canyon acceleration",
        position="upper_left",
        font_size=20,
        color="#1a1a1a",
        shadow=False,
    )
    plotter.set_background("#f4f4f5")

    fp_z = float(min(50.0, zmax_clip * 0.2))
    cdf = float(np.clip(camera_distance_factor, 0.65, 3.0))
    cam_dx = max(400.0, 2.0 * float(half_w)) * cdf
    cam_dz = min(250.0, zmax_clip * 0.98) * cdf
    cam_dz = float(min(cam_dz, max(320.0, zmax_clip * 1.35)))
    cam_pos = (x0 - cam_dx, y0 - cam_dx, cam_dz)
    plotter.camera_position = [cam_pos, (x0, y0, fp_z), (0.0, 0.0, 1.0)]

    if not use_edl:
        try:
            plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
        except Exception:
            pass

    plotter.screenshot(str(out_png), transparent_background=False)
    plotter.close()


def _render_fallback_matplotlib(
    *,
    case_dir: Path,
    buildings_stl: Path,
    out_png: Path,
    x0: float,
    y0: float,
    half_w: float,
    zmax: float,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from scipy.interpolate import griddata

    csv_100 = case_dir / "postProcessing" / "100m.csv"
    csv_y = case_dir / "postProcessing" / "y800m.csv"
    if not csv_100.is_file() or not csv_y.is_file():
        raise FileNotFoundError("Fallback requires postProcessing/100m.csv and y800m.csv")

    xmin, xmax = x0 - half_w, x0 + half_w
    ymin, ymax = y0 - half_w, y0 + half_w

    def _load_xy_ws(path: Path):
        xs, ys, ws = [], [], []
        for ch in pd.read_csv(path, chunksize=200_000):
            x = ch["Coords:0"].astype(np.float64).to_numpy()
            y = ch["Coords:1"].astype(np.float64).to_numpy()
            u0 = ch["U:0"].astype(np.float64).to_numpy()
            u1 = ch["U:1"].astype(np.float64).to_numpy()
            m = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
            xs.append(x[m])
            ys.append(y[m])
            ws.append(np.sqrt(u0[m] ** 2 + u1[m] ** 2))
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)

    def _load_xz_ws(path: Path, y0_lo: float, y0_hi: float):
        xs, zs, ws = [], [], []
        for ch in pd.read_csv(path, chunksize=200_000):
            x = ch["Coords:0"].astype(np.float64).to_numpy()
            z = ch["Coords:2"].astype(np.float64).to_numpy()
            y = ch["Coords:1"].astype(np.float64).to_numpy()
            u0 = ch["U:0"].astype(np.float64).to_numpy()
            u2 = ch["U:2"].astype(np.float64).to_numpy()
            m = (x >= xmin) & (x <= xmax) & (z >= 0.0) & (z <= zmax) & (y >= y0_lo) & (y <= y0_hi)
            xs.append(x[m])
            zs.append(z[m])
            ws.append(np.sqrt(u0[m] ** 2 + u2[m] ** 2))
        return np.concatenate(xs), np.concatenate(zs), np.concatenate(ws)

    x1, y1, w1 = _load_xy_ws(csv_100)
    yband = max(25.0, half_w / 10.0)
    x2, z2, w2 = _load_xz_ws(csv_y, y0 - yband, y0 + yband)

    vmax = float(np.nanpercentile(np.concatenate([w1, w2]), 98)) if w1.size and w2.size else 1.0
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-3))

    plt.style.use("seaborn-v0_8-paper")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 6.0), constrained_layout=True)

    hb1 = ax1.hexbin(x1, y1, C=w1, gridsize=90, reduce_C_function=np.mean, cmap="viridis", norm=norm)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title(f"(a) |U| at z≈100 m — zoom ({half_w:.0f} m half-width)")
    cb1 = fig.colorbar(hb1, ax=ax1, fraction=0.046, pad=0.02)
    cb1.set_label("|U_h| (m/s)")

    hb2 = ax2.hexbin(x2, z2, C=w2, gridsize=90, reduce_C_function=np.mean, cmap="viridis", norm=norm)
    # Quiver on coarse grid for xz (single accumulated subsample → one griddata)
    if x2.size > 50:
        gx = np.linspace(xmin, xmax, 22)
        gz = np.linspace(0.0, zmax, 18)
        GX, GZ = np.meshgrid(gx, gz)
        xa, za, u0a, u2a = [], [], [], []
        sub = max(1, len(x2) // 120_000)
        for ch in pd.read_csv(csv_y, chunksize=200_000):
            x = ch["Coords:0"].astype(np.float64).to_numpy()
            z = ch["Coords:2"].astype(np.float64).to_numpy()
            y = ch["Coords:1"].astype(np.float64).to_numpy()
            u0 = ch["U:0"].astype(np.float64).to_numpy()
            u2 = ch["U:2"].astype(np.float64).to_numpy()
            m = (x >= xmin) & (x <= xmax) & (z >= 0.0) & (z <= zmax) & (y >= y0 - yband) & (y <= y0 + yband)
            xa.append(x[m][::sub])
            za.append(z[m][::sub])
            u0a.append(u0[m][::sub])
            u2a.append(u2[m][::sub])
        x = np.concatenate(xa)
        z = np.concatenate(za)
        u0 = np.concatenate(u0a)
        u2 = np.concatenate(u2a)
        U0 = griddata((x, z), u0, (GX, GZ), method="linear")
        U2 = griddata((x, z), u2, (GX, GZ), method="linear")
        qv = ax2.quiver(
            GX,
            GZ,
            U0,
            U2,
            scale=40.0,
            width=0.0025,
            color="k",
            alpha=0.55,
        )
        ax2.quiverkey(qv, 0.9, 0.95, 2.5, "2.5 m/s", labelpos="E", coordinates="axes")

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_title(f"(b) |U| on X–Z slab — y≈{y0:.0f} m ±{yband:.0f} m")
    cb2 = fig.colorbar(hb2, ax=ax2, fraction=0.046, pad=0.02)
    cb2.set_label("|U| (m/s)")

    # Optional: building footprint from STL is heavy; annotate hotspot instead
    ax1.scatter([x0], [y0], s=120, facecolors="none", edgecolors="r", linewidths=2.0, label="Hotspot center")
    ax1.legend(loc="upper right")

    fig.suptitle(
        "OpenFOAM micro-hazard snapshot (matplotlib fallback)\n"
        "2025-09-03 22:00 LST (14:00 UTC) — nocturnal ABL",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(out_png, dpi=int(dpi))
    plt.close(fig)
    _ = buildings_stl  # STL not drawn in lightweight fallback


def main() -> None:
    args = _parse_args()
    root = _repo_root()
    case_dir = args.case_dir if args.case_dir.is_absolute() else (root / args.case_dir)
    buildings = args.buildings_stl if args.buildings_stl.is_absolute() else (root / args.buildings_stl)
    out_png = args.out if args.out.is_absolute() else (root / args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    csv_100 = case_dir / "postProcessing" / "100m.csv"
    if not csv_100.is_file():
        print(f"ERROR: missing {csv_100}", file=sys.stderr)
        sys.exit(1)

    print("=" * 72)
    print("Stage A: hotspot from 100m.csv")
    print("=" * 72)
    x, y, w = _read_100m_windspeed(csv_100)
    x0, y0, thr, n_cand = _cluster_hotspot_xy(
        x,
        y,
        w,
        urban_core_m=float(args.urban_core_m),
        p_pct=float(args.p_percentile),
        dbscan_eps=float(args.dbscan_eps_m),
        dbscan_min_samples=int(args.dbscan_min_samples),
    )
    print(f"  Urban core: |x|,|y| < {args.urban_core_m:.0f} m")
    print(f"  P{args.p_percentile:.1f} threshold (core): {thr:.4f} m/s")
    print(f"  Candidates >= threshold: {n_cand}")
    print(f"  Hotspot center (x0,y0): ({x0:.2f}, {y0:.2f}) m")
    z_eff = min(float(args.zmax_m), float(args.analysis_z_max_m))
    print(f"  Zoom box: x∈[{x0-args.half_width_m:.1f},{x0+args.half_width_m:.1f}], "
          f"y∈[{y0-args.half_width_m:.1f},{y0+args.half_width_m:.1f}], "
          f"z∈[0,{z_eff:.1f}] (min of zmax-m and analysis-z-max-m)")

    print("=" * 72)
    print("Stage B: render")
    print("=" * 72)

    w_w, h_h = [int(s.strip()) for s in str(args.window_size).split(",") if s.strip()]
    used = "unknown"
    if not args.fallback_only:
        try:
            import pyvista as pv  # noqa: F401

            iso_pct = _parse_iso_percentiles(str(args.iso_percentiles))
            _render_pyvista(
                case_dir=case_dir,
                buildings_stl=buildings,
                out_png=out_png,
                x0=x0,
                y0=y0,
                half_w=float(args.half_width_m),
                zmax=float(args.zmax_m),
                slice_z=float(args.slice_z_m),
                window_size=(w_w, h_h),
                cmap=str(args.cmap),
                clim_low_percentile=float(args.clim_low_percentile),
                clim_percentile=float(args.clim_percentile),
                analysis_z_max_m=float(args.analysis_z_max_m),
                iso_percentiles=iso_pct,
                iso_opacity=float(args.iso_opacity),
                seed_box_m=float(args.seed_box_m),
                seed_z_max_m=float(args.seed_z_max_m),
                seed_nx=int(args.seed_nx),
                seed_ny=int(args.seed_ny),
                seed_nz=int(args.seed_nz),
                use_edl=not bool(args.no_edl),
                building_wind_sampling=bool(args.building_wind),
                tube_radius_m=float(args.tube_radius_m),
                show_streamlines=not bool(args.no_streamlines),
                camera_distance_factor=float(args.camera_distance_factor),
            )
            used = "pyvista"
        except Exception as e:
            if bool(args.require_3d):
                print(
                    f"ERROR: PyVista 3D render failed ({type(e).__name__}: {e}). "
                    "Roadshow mode (--require-3d): not falling back to 2D. "
                    "Install PyVista/VTK, verify myExpxx.foam and time fields, or use "
                    "foamToVTK + ParaView; see AUTO-CHECKPOINT.md.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"[!] PyVista render failed ({type(e).__name__}: {e}). Using matplotlib fallback.")
            _render_fallback_matplotlib(
                case_dir=case_dir,
                buildings_stl=buildings,
                out_png=out_png,
                x0=x0,
                y0=y0,
                half_w=float(args.half_width_m),
                zmax=float(args.zmax_m),
                dpi=int(args.dpi),
            )
            used = "matplotlib_fallback"
    else:
        _render_fallback_matplotlib(
            case_dir=case_dir,
            buildings_stl=buildings,
            out_png=out_png,
            x0=x0,
            y0=y0,
            half_w=float(args.half_width_m),
            zmax=float(args.zmax_m),
            dpi=int(args.dpi),
        )
        used = "matplotlib_fallback"

    print(f"OK ({used}): {out_png}")


if __name__ == "__main__":
    main()
