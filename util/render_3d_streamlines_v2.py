#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_3d_streamlines_v2.py
===========================
3D wind-streamline visualisation targeting the "Orbital Stack / commercial CFD"
aesthetic ([`docs/image2.png`](docs/image2.png)):

  • NO horizontal slice shading at the bottom
  • Dark-studio default: background + charcoal-ish buildings; ``--light-theme`` for pale look
  • Hotspot-centred box seeds (default 7×7×6), ``integration_direction=both``, thin tubes (turbo); smaller integration step for smooth polylines
  • Wide horizontal colourbar, title/legend contrast from background luminance
  • Optional: a thin semi-transparent horizontal reference plane at seed height

Usage
-----
  python render_3d_streamlines_v2.py --case-dir <path/to/openfoam/case>
  python render_3d_streamlines_v2.py --window-size 3840,2160 --cmap turbo
  python render_3d_streamlines_v2.py --half-width-m 350 --camera-distance-factor 1.4

All existing Stage-A hotspot-detection logic is preserved unchanged.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# helpers: repo root / arg parsing
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3D CFD streamline render (v2) — commercial-style, no bottom shading."
    )
    p.add_argument(
        "--case-dir",
        type=Path,
        default=_repo_root()
        / "steady_experiments_finer_ABL"
        / "20250903_1400_two_boundaries_as_outlet",
    )
    p.add_argument(
        "--buildings-stl",
        type=Path,
        default=_repo_root() / "constant" / "triSurface" / "buildings.stl",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_repo_root()
        / "results"
        / "microhazard"
        / "20250903_1400"
        / "figure1_streamlines_v2.png",
    )

    # --- domain / hotspot ---
    p.add_argument("--urban-core-m", type=float, default=1500.0)
    p.add_argument("--p-percentile", type=float, default=99.0)
    p.add_argument("--dbscan-eps-m", type=float, default=80.0)
    p.add_argument("--dbscan-min-samples", type=int, default=5)
    p.add_argument(
        "--half-width-m",
        type=float,
        default=235.0,
        help="Horizontal half-width of the clip box around the hotspot (m). Slightly tighter ≈ less empty margin.",
    )
    p.add_argument("--zmax-m", type=float, default=300.0)
    p.add_argument("--analysis-z-max-m", type=float, default=250.0,
                   help="Cap Z for clip / clim / streamline seeds.")

    # --- streamline seeds ---
    # Seed strategy: a wide upstream LINE swept across multiple heights so that
    # streamlines fan out naturally across the whole scene (like image2).
    p.add_argument(
        "--seed-style",
        choices=["line", "plane", "box"],
        default="box",
        help="'line' = west-edge vertical line; 'plane' = YZ plane upstream of hotspot (not xmin); "
        "'box' = 3D grid centred on hotspot (default, best for canyon flow).",
    )
    p.add_argument("--seed-nx", type=int, default=7,
                   help="Plane: samples along Y; Box: samples in X.")
    p.add_argument("--seed-ny", type=int, default=7,
                   help="Plane: unused (Y uses seed-nx); Box: samples in Y.")
    p.add_argument("--seed-nz", type=int, default=6,
                   help="Number of seed heights (Z).")
    p.add_argument("--seed-z-min-m", type=float, default=10.0,
                   help="Lowest seed height (m).")
    p.add_argument("--seed-z-max-m", type=float, default=0.0,
                   help="Highest seed height (m); 0 = auto (0.55 × clip top, ≤150 m).")

    # --- streamline integration ---
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--max-length-m", type=float, default=800.0,
                   help="Maximum streamline arc-length (m). Larger → longer, more dramatic lines.")
    p.add_argument("--initial-step", type=float, default=0.06,
                   help="Smaller → smoother polylines (≈0.04–0.10 typical); trades runtime.")
    p.add_argument("--terminal-speed", type=float, default=0.01)

    # --- tube appearance ---
    p.add_argument(
        "--tube-radius-m",
        type=float,
        default=0.38,
        help="Tube radius (m); thin (~0.35–0.55) matches reference 'filament' look.",
    )
    p.add_argument("--tube-sides", type=int, default=32)
    p.add_argument(
        "--tube-opacity",
        type=float,
        default=0.88,
        help="Slight transparency helps layered streamlines read like image2.",
    )

    # --- colour / colourbar ---
    p.add_argument("--cmap", type=str, default="turbo")
    p.add_argument("--clim-low-percentile", type=float, default=2.0)
    p.add_argument("--clim-percentile", type=float, default=98.0)

    # --- buildings ---
    p.add_argument(
        "--building-color",
        type=str,
        default="#5c5c62",
        help="Building albedo (hex). Default mid-dark grey for dark-studio reference.",
    )
    p.add_argument("--building-opacity", type=float, default=1.0)
    p.add_argument("--building-specular", type=float, default=0.25,
                   help="Specular highlight strength (0–1). Gives subtle 3D sheen.")
    p.add_argument("--building-ambient", type=float, default=0.35)
    p.add_argument("--building-diffuse", type=float, default=0.70)

    # --- optional ground plane ---
    p.add_argument("--ground-plane", action="store_true",
                   help="Add a very thin light-grey ground plane (no wind colouring).")
    p.add_argument("--ground-color", type=str, default="#3a3a40")

    # --- camera ---
    p.add_argument(
        "--camera-distance-factor",
        type=float,
        default=1.0,
        help="Scale camera distance; ~0.85–1.0 tightens framing and cuts empty margin.",
    )
    p.add_argument("--camera-elevation-deg", type=float, default=28.0,
                   help="Camera elevation angle above horizon (degrees). ~25–35 matches image2.")
    p.add_argument("--camera-azimuth-deg", type=float, default=225.0,
                   help="Camera azimuth (degrees, 0=+X axis). 225 ≈ SW view like image2.")

    # --- rendering ---
    p.add_argument("--window-size", type=str, default="1920,1280")
    p.add_argument(
        "--background-color",
        type=str,
        default="#1e1e22",
        help="Scene background (hex). Default dark studio; use #f5f5f5 with --light-theme.",
    )
    p.add_argument(
        "--light-theme",
        action="store_true",
        help="Pale background + dark text + light ground (legacy look).",
    )
    p.add_argument("--no-edl", action="store_true")
    p.add_argument("--title", type=str,
                   default="2025-09-03 22:00 LST (14:00 UTC)\n"
                           "Nocturnal ABL · Venturi / canyon acceleration")
    p.add_argument(
        "--title-font-size",
        type=int,
        default=11,
        help="Base corner title size (1080p ref); colorbar uses its own floor sizes for legibility.",
    )

    p.add_argument("--require-3d", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _parse_window(s: str) -> Tuple[int, int]:
    w, h = [int(x.strip()) for x in s.split(",")]
    return w, h


# ---------------------------------------------------------------------------
# Stage A: hotspot detection (unchanged from original)
# ---------------------------------------------------------------------------

def _read_100m_windspeed(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


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
    core = (np.abs(x) < urban_core_m) & (np.abs(y) < urban_core_m)
    if not np.any(core):
        raise RuntimeError("Urban core mask empty; check coordinates / urban_core_m.")
    thr = float(np.percentile(w[core], p_pct))
    cand = core & (w >= thr)
    n_cand = int(np.sum(cand))
    if n_cand < 10:
        idx = int(np.argmax(np.where(core, w, -np.inf)))
        return float(x[idx]), float(y[idx]), thr, int(np.sum(core))
    xc, yc = x[cand], y[cand]
    try:
        from sklearn.cluster import DBSCAN  # type: ignore
        xy = np.column_stack([xc, yc])
        lab = DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min_samples)).fit_predict(xy)
        best_label, best_count = None, -1
        for lbl in sorted(set(lab)):
            if lbl == -1:
                continue
            c = int(np.sum(lab == lbl))
            if c > best_count:
                best_count, best_label = c, lbl
        if best_label is None:
            raise RuntimeError("no clusters")
        m = lab == best_label
        return float(np.mean(xc[m])), float(np.mean(yc[m])), thr, n_cand
    except Exception:
        eps = float(dbscan_eps)
        ix = np.floor((xc - np.min(xc)) / eps).astype(int)
        iy = np.floor((yc - np.min(yc)) / eps).astype(int)
        keys = ix * (iy.max() + 5) + iy
        uniq, inv = np.unique(keys, return_inverse=True)
        counts = np.bincount(inv)
        sel = inv == int(np.argmax(counts))
        return float(np.mean(xc[sel])), float(np.mean(yc[sel])), thr, n_cand


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

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
    for i in range(mb.n_blocks):
        b = mb.get_block(i)
        if b is None:
            continue
        name = (mb.get_block_name(i) or "").lower()
        if "internal" in name:
            return b
    for i in range(mb.n_blocks):
        b = mb.get_block(i)
        if b is not None and b.n_cells > 0:
            return b
    raise RuntimeError("Could not find internal mesh block in OpenFOAM reader.")


# ---------------------------------------------------------------------------
# Seed generators
# ---------------------------------------------------------------------------

def _make_seeds(
    style: str,
    x0: float, y0: float,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    z_seed_min: float, z_seed_max: float,
    nx: int, ny: int, nz: int,
):
    """Return a pyvista PolyData seed cloud for streamlines."""
    import pyvista as pv

    nz = max(2, nz)

    if style == "line":
        # Single vertical inlet line — gives focused fan of lines
        return pv.Line(
            (xmin + 5.0, y0, z_seed_min),
            (xmin + 5.0, y0, z_seed_max),
            resolution=nz,
        )

    elif style == "plane":
        # YZ plane upstream of hotspot (not xmin+5): avoids all seeds hugging west clip edge.
        nx_y = max(3, int(nx))
        nz_e = max(2, int(nz))
        span_x = float(xmax - xmin)
        span_y = float(ymax - ymin)
        margin = max(8.0, min(20.0, 0.03 * span_x))
        x_seed = float(x0 - 0.42 * span_x)
        x_seed = float(np.clip(x_seed, xmin + margin, x0 - margin * 0.75))
        ys = np.linspace(ymin + margin, ymax - margin, nx_y)
        zs = np.linspace(z_seed_min, z_seed_max, nz_e)
        YY, ZZ = np.meshgrid(ys, zs, indexing="ij")
        XX = np.full_like(YY, x_seed)
        pts = np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()]
        return pv.PolyData(pts)

    else:  # box — dense 3D cloud over hotspot (image2-style canyon sampling)
        nx, ny = max(3, nx), max(3, ny)
        half_xy = min((xmax - xmin) * 0.48, (ymax - ymin) * 0.48)
        xs = np.linspace(x0 - half_xy, x0 + half_xy, nx)
        ys = np.linspace(y0 - half_xy, y0 + half_xy, ny)
        zs = np.linspace(z_seed_min, z_seed_max, max(2, nz))
        XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()]
        return pv.PolyData(pts)


# ---------------------------------------------------------------------------
# Stage B: PyVista render — v2 aesthetic
# ---------------------------------------------------------------------------

def _render_pyvista_v2(
    *,
    case_dir: Path,
    buildings_stl: Path,
    out_png: Path,
    x0: float,
    y0: float,
    half_w: float,
    zmax: float,
    analysis_z_max_m: float,
    # seeds
    seed_style: str,
    seed_nx: int,
    seed_ny: int,
    seed_nz: int,
    seed_z_min_m: float,
    seed_z_max_m: float,
    # integration
    max_steps: int,
    max_length_m: float,
    initial_step: float,
    terminal_speed: float,
    # tubes
    tube_radius_m: float,
    tube_sides: int,
    tube_opacity: float,
    # colour
    cmap: str,
    clim_low_percentile: float,
    clim_percentile: float,
    # buildings
    building_color: str,
    building_opacity: float,
    building_specular: float,
    building_ambient: float,
    building_diffuse: float,
    # ground
    ground_plane: bool,
    ground_color: str,
    # camera
    camera_distance_factor: float,
    camera_elevation_deg: float,
    camera_azimuth_deg: float,
    # rendering
    window_size: Tuple[int, int],
    background_color: str,
    use_edl: bool,
    title: str,
    title_font_size: int,
) -> None:
    import pyvista as pv

    # ------------------------------------------------------------------
    # 1. Read OpenFOAM
    # ------------------------------------------------------------------
    foam = case_dir / "myExpxx.foam"
    if not foam.is_file():
        raise FileNotFoundError(f"Missing OpenFOAM case marker: {foam}")

    reader = pv.POpenFOAMReader(str(foam))
    tvals = list(reader.time_values)
    if not tvals:
        raise RuntimeError("OpenFOAM reader returned no time values.")
    reader.set_active_time_value(float(tvals[-1]))
    mb = reader.read()
    internal = _pick_internal_block(mb).copy(deep=True)

    # Cell → point data for streamlines
    if "U" in internal.array_names and "U" in internal.cell_data:
        internal = internal.cell_data_to_point_data(pass_cell_data=False)
    _add_u_mag(internal)

    # ------------------------------------------------------------------
    # 2. Clip domain
    # ------------------------------------------------------------------
    z_top = float(max(30.0, min(float(zmax), float(analysis_z_max_m))))
    xmin, xmax = x0 - half_w, x0 + half_w
    ymin, ymax = y0 - half_w, y0 + half_w
    bounds = (xmin, xmax, ymin, ymax, 0.0, z_top)

    clipped = internal.clip_box(bounds=bounds, invert=False)
    if clipped.n_cells == 0:
        raise RuntimeError("clip_box removed all cells; check hotspot / bounds.")

    # ------------------------------------------------------------------
    # 3. Buildings
    # ------------------------------------------------------------------
    bmesh = pv.read(str(buildings_stl))
    bclip = bmesh.clip_box(bounds=bounds, invert=False)

    # ------------------------------------------------------------------
    # 4. Colour limits
    # ------------------------------------------------------------------
    wmag = np.asarray(clipped["U_mag"], dtype=float)
    p_lo = float(np.clip(clim_low_percentile, 0.0, 49.5))
    p_hi = float(np.clip(clim_percentile, p_lo + 1.0, 99.999))
    v_lo = float(np.nanpercentile(wmag, p_lo))
    v_hi = float(np.nanpercentile(wmag, p_hi))
    if not (np.isfinite(v_lo) and np.isfinite(v_hi)) or v_hi <= v_lo:
        v_lo, v_hi = 0.0, float(np.nanpercentile(wmag, 98.0))
    clim = (v_lo, v_hi)
    print(f"  Colour limits: {v_lo:.3f} – {v_hi:.3f} m/s")

    # ------------------------------------------------------------------
    # 5. Streamlines
    # ------------------------------------------------------------------
    z_seed_top = (
        float(seed_z_max_m)
        if seed_z_max_m > 0.0
        else min(150.0, z_top * 0.55)
    )
    z_seed_top = float(np.clip(z_seed_top, seed_z_min_m + 5.0, z_top - 2.0))
    z_seed_bot = float(np.clip(seed_z_min_m, 3.0, z_seed_top - 5.0))

    seeds = _make_seeds(
        style=seed_style,
        x0=x0, y0=y0,
        xmin=xmin, xmax=xmax,
        ymin=ymin, ymax=ymax,
        z_seed_min=z_seed_bot,
        z_seed_max=z_seed_top,
        nx=seed_nx, ny=seed_ny, nz=seed_nz,
    )
    print(f"  Seed style: {seed_style}, n_seeds={seeds.n_points}")

    # Build kwargs adaptively (pyvista version differences)
    _sig = inspect.signature(clipped.streamlines_from_source)
    _stream_kw: dict = {
        "vectors": "U",
        "integration_direction": "both",
        "initial_step_length": float(initial_step),
        "max_step_length": float(max(initial_step * 2.2, 0.08)),
        "max_steps": int(max_steps),
        "terminal_speed": float(terminal_speed),
        "interpolator_type": "point",
    }
    if "max_length" in _sig.parameters:
        _stream_kw["max_length"] = float(max_length_m)
    elif "max_time" in _sig.parameters:
        _stream_kw["max_time"] = float(max_length_m)

    print("  Integrating streamlines …")
    streams = clipped.streamlines_from_source(seeds, **_stream_kw)
    print(f"  Streamlines: {streams.n_points} points, {streams.n_cells} lines")

    # Fallback: vertical line seeds if plane produced nothing
    if streams.n_points == 0:
        print("  [!] Plane seeds produced no streamlines — falling back to line seeds.")
        fallback_seeds = pv.Line(
            (xmin + 5.0, y0, z_seed_bot),
            (xmin + 5.0, y0, z_seed_top),
            resolution=int(seed_nz),
        )
        streams = clipped.streamlines_from_source(fallback_seeds, **_stream_kw)

    # Attach U_mag for colouring
    if streams.n_points > 0:
        if "U" in streams.array_names and "U_mag" not in streams.array_names:
            Uv = np.asarray(streams["U"])
            if Uv.ndim == 2 and Uv.shape[1] >= 3:
                streams["U_mag"] = np.linalg.norm(Uv[:, :3], axis=1)
            else:
                streams["U_mag"] = np.abs(Uv).ravel()

    # ------------------------------------------------------------------
    # 6. Plotter
    # ------------------------------------------------------------------
    w_w, h_h = window_size
    plotter = pv.Plotter(off_screen=True, window_size=(int(w_w), int(h_h)))
    plotter.set_background(background_color)

    def _bg_luma(hex_s: str) -> float:
        h = (hex_s or "#000000").lstrip("#")
        if len(h) >= 6:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        return 0.0

    _dark = _bg_luma(background_color) < 0.42
    _ink = "#f0f0f0" if _dark else "#111111"
    # Sub-linear vs 1080p so 4K screenshots don't balloon annotations (title clipping).
    _cb_scale = max(1.0, float(h_h) / 1080.0)
    _text_scale = float(np.clip(_cb_scale ** 0.5, 0.9, 1.28))

    if use_edl:
        try:
            plotter.enable_eye_dome_lighting()
        except Exception:
            pass
    else:
        try:
            plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 7. Ground plane (optional — no wind colour, just pale grey)
    # ------------------------------------------------------------------
    if ground_plane:
        gp = pv.Plane(
            center=(x0, y0, 0.0),
            direction=(0.0, 0.0, 1.0),
            i_size=float(xmax - xmin) * 1.05,
            j_size=float(ymax - ymin) * 1.05,
        )
        plotter.add_mesh(
            gp,
            color=ground_color,
            opacity=0.92 if _dark else 0.85,
            show_edges=False,
            smooth_shading=False,
        )

    # ------------------------------------------------------------------
    # 8. Buildings — solid grey, smooth shading, subtle lighting
    # ------------------------------------------------------------------
    if bclip.n_cells > 0:
        plotter.add_mesh(
            bclip,
            color=building_color,
            opacity=float(building_opacity),
            smooth_shading=True,
            show_edges=False,
            specular=float(building_specular),
            specular_power=20,
            ambient=float(building_ambient),
            diffuse=float(building_diffuse),
        )

    # ------------------------------------------------------------------
    # 9. Streamlines as tubes
    # ------------------------------------------------------------------
    if streams.n_points > 0:
        tube_r = float(max(0.15, min(5.0, tube_radius_m)))
        tubes = streams.tube(radius=tube_r, n_sides=int(tube_sides))
        if "U_mag" in tubes.array_names:
            tubes.set_active_scalars("U_mag")

        # Colourbar: horizontal, bottom centre, matching image2 style
        # PyVista: many builds do not support title_font_family / label_font_family on add_scalar_bar.
        _cb_title_fs = max(16, int(19 * _text_scale))
        _cb_label_fs = max(14, int(16 * _text_scale))
        scalar_bar_args = dict(
            title="Wind speed |U| (m/s)",
            title_font_size=_cb_title_fs,
            label_font_size=_cb_label_fs,
            color=_ink,
            fmt="%.2f",
            n_labels=5,
            vertical=False,
            position_x=0.12,
            position_y=0.07,
            width=0.76,
            height=max(0.06, min(0.095, 0.052 * _cb_scale)),
            shadow=False,
        )

        plotter.add_mesh(
            tubes,
            scalars="U_mag" if "U_mag" in tubes.array_names else None,
            cmap=cmap,
            clim=clim,
            smooth_shading=True,
            ambient=0.35,
            diffuse=0.65,
            specular=0.18,
            opacity=float(np.clip(tube_opacity, 0.2, 1.0)),
            show_scalar_bar=True,
            scalar_bar_args=scalar_bar_args,
        )
    else:
        print("  [!] No streamlines to render.")

    # ------------------------------------------------------------------
    # 10. Camera: isometric-ish SW view matching image2
    # ------------------------------------------------------------------
    scene_center = (x0, y0, float(z_top * 0.25))

    # Convert azimuth/elevation to Cartesian offset
    az_rad = np.deg2rad(float(camera_azimuth_deg))
    el_rad = np.deg2rad(float(camera_elevation_deg))
    cdf = float(np.clip(camera_distance_factor, 0.5, 4.0))
    dist = float(max(half_w, 260.0)) * 2.35 * cdf

    cam_dx = dist * np.cos(el_rad) * np.cos(az_rad)
    cam_dy = dist * np.cos(el_rad) * np.sin(az_rad)
    cam_dz = dist * np.sin(el_rad)

    cam_pos = (x0 + cam_dx, y0 + cam_dy, scene_center[2] + cam_dz)
    focal_pt = (x0, y0, float(z_top * 0.22))

    plotter.camera_position = [cam_pos, focal_pt, (0.0, 0.0, 1.0)]

    # ------------------------------------------------------------------
    # 11. Title text — after camera so overlay matches final framing
    # ------------------------------------------------------------------
    plotter.add_text(
        title,
        position="upper_left",
        font_size=max(9, int(title_font_size * _text_scale)),
        color=_ink,
        shadow=False,
    )

    # ------------------------------------------------------------------
    # 12. Render
    # ------------------------------------------------------------------
    print(f"  Saving to {out_png} …")
    plotter.screenshot(str(out_png), transparent_background=False)
    plotter.close()
    print("  Done.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    if bool(getattr(args, "light_theme", False)):
        args.background_color = "#f5f5f5"
        args.building_color = "#b0b0b0"
        args.ground_color = "#e6e6e6"
    root = _repo_root()

    case_dir = args.case_dir if args.case_dir.is_absolute() else (root / args.case_dir)
    buildings = args.buildings_stl if args.buildings_stl.is_absolute() else (root / args.buildings_stl)
    out_png = args.out if args.out.is_absolute() else (root / args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    csv_100 = case_dir / "postProcessing" / "100m.csv"
    if not csv_100.is_file():
        print(f"ERROR: missing {csv_100}", file=sys.stderr)
        sys.exit(1)

    # ---- Stage A: hotspot ----
    print("=" * 72)
    print("Stage A: hotspot from 100m.csv")
    print("=" * 72)
    x, y, w = _read_100m_windspeed(csv_100)
    x0, y0, thr, n_cand = _cluster_hotspot_xy(
        x, y, w,
        urban_core_m=float(args.urban_core_m),
        p_pct=float(args.p_percentile),
        dbscan_eps=float(args.dbscan_eps_m),
        dbscan_min_samples=int(args.dbscan_min_samples),
    )
    z_eff = min(float(args.zmax_m), float(args.analysis_z_max_m))
    print(f"  P{args.p_percentile:.1f} threshold: {thr:.4f} m/s  |  candidates: {n_cand}")
    print(f"  Hotspot (x0, y0): ({x0:.2f}, {y0:.2f}) m")
    print(f"  Clip box: [{x0-args.half_width_m:.1f}, {x0+args.half_width_m:.1f}] × "
          f"[{y0-args.half_width_m:.1f}, {y0+args.half_width_m:.1f}] × [0, {z_eff:.1f}] m")

    # ---- Stage B: render ----
    print("=" * 72)
    print("Stage B: PyVista v2 render")
    print("=" * 72)

    w_w, h_h = _parse_window(args.window_size)

    try:
        import pyvista as pv  # noqa: F401
        _render_pyvista_v2(
            case_dir=case_dir,
            buildings_stl=buildings,
            out_png=out_png,
            x0=x0,
            y0=y0,
            half_w=float(args.half_width_m),
            zmax=float(args.zmax_m),
            analysis_z_max_m=float(args.analysis_z_max_m),
            seed_style=str(args.seed_style),
            seed_nx=int(args.seed_nx),
            seed_ny=int(args.seed_ny),
            seed_nz=int(args.seed_nz),
            seed_z_min_m=float(args.seed_z_min_m),
            seed_z_max_m=float(args.seed_z_max_m),
            max_steps=int(args.max_steps),
            max_length_m=float(args.max_length_m),
            initial_step=float(args.initial_step),
            terminal_speed=float(args.terminal_speed),
            tube_radius_m=float(args.tube_radius_m),
            tube_sides=int(args.tube_sides),
            tube_opacity=float(args.tube_opacity),
            cmap=str(args.cmap),
            clim_low_percentile=float(args.clim_low_percentile),
            clim_percentile=float(args.clim_percentile),
            building_color=str(args.building_color),
            building_opacity=float(args.building_opacity),
            building_specular=float(args.building_specular),
            building_ambient=float(args.building_ambient),
            building_diffuse=float(args.building_diffuse),
            ground_plane=bool(args.ground_plane),
            ground_color=str(args.ground_color),
            camera_distance_factor=float(args.camera_distance_factor),
            camera_elevation_deg=float(args.camera_elevation_deg),
            camera_azimuth_deg=float(args.camera_azimuth_deg),
            window_size=(w_w, h_h),
            background_color=str(args.background_color),
            use_edl=not bool(args.no_edl),
            title=str(args.title),
            title_font_size=int(args.title_font_size),
        )
        print(f"OK (pyvista): {out_png}")

    except Exception as e:
        if bool(args.require_3d):
            print(
                f"ERROR: PyVista render failed ({type(e).__name__}: {e}). "
                "Use --no-require-3d to allow matplotlib fallback.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[!] PyVista failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()