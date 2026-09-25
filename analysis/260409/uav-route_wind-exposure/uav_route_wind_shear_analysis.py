#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV 低空航线风场与垂直风切变对比：WRF vs WRF-to-OpenFOAM (CFD)

沿两条典型航线（沿江开阔 / 穿楼复杂）提取 120 m（可扩展 30/60 m）水平风速
与三维矢量垂直风切变 |∂V/∂Z|，并结合 constant/triSurface/buildings.stl 判定航线障碍物，输出出版级对比图。
航线总览底图使用 Export_Output.shp 建筑轮廓（按 jzgd 高度着色）。

默认快照：2025-09-03 12:00:00 UTC
垂直风切变：Δz=40 m，中心差分
  |∂V/∂Z| = sqrt(du²+dv²+dw²) / Δz，V=(U,V,W)

用法:
  python analysis/260409/uav_route_wind_shear_analysis.py
  python analysis/260409/uav_route_wind_shear_analysis.py --datetime "2025-09-03 12:00:00"
  python analysis/260409/uav_route_wind_shear_analysis.py --height 30
  python analysis/260409/uav_route_wind_shear_analysis.py --height 60 --skip-overview

高度相关输出（fig1/fig2、沿轨 CSV）文件名带 _z{H} 后缀，避免不同高度互相覆盖。
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
import time
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATETIME = "2025-09-03 12:00:00"
DEFAULT_HEIGHT = 120.0
DEFAULT_SAMPLE_STEP = 20.0
DEFAULT_SHEAR_DZ = 40.0
DEFAULT_STL = Path(__file__).resolve().parents[3] / "constant" / "triSurface" / "buildings.stl"
DEFAULT_SHP = REPO_ROOT / "data" / "Guangzhou_shp_file" / "project_UTM49" / "Export_Output.shp"
# Same origin as scripts/shp_to_lod1_stl.py → OpenFOAM / STL local XY
DEFAULT_ORIGIN_LON = 113.3218197
DEFAULT_ORIGIN_LAT = 23.1133057
DEFAULT_CELL_CENTRES = (
    REPO_ROOT
    / "steady_experiments_finer_ABL"
    / "20250901_0000_two_boundaries_as_outlet"
    / "0"
    / "C"
)

# Route definitions (local relative metres; ~~out-and-back~~ -> one-way outbound only)
ROUTE_SPECS = {
    "Route1_open_river": {
        "label": "Route 1 (open river)",
        "start": (-1500.0, 0.0),
        "end": (1500.0, 0.0),
        "description": "沿江开阔航线",
    },
    "Route2_urban_canyon": {
        "label": "Route 2 (urban canyon)",
        "start": (400.0, -1000.0),
        "end": (400.0, 1000.0),
        "description": "穿楼复杂航线",
    },
}

_VECTOR_FIELD_RE = re.compile(
    r"internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*;",
    re.DOTALL,
)
_UNIFORM_VECTOR_RE = re.compile(r"internalField\s+uniform\s+\(([^)]+)\)")
_VEC_TOKEN_RE = re.compile(
    r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)"
)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def configure_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": ":",
            "grid.color": "black",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


# ---------------------------------------------------------------------------
# 1. Route generation
# ---------------------------------------------------------------------------
def build_route(
    name: str,
    p_start: tuple[float, float],
    p_end: tuple[float, float],
    step: float,
) -> pd.DataFrame:
    """
    old: Generate out-and-back sampling points along a straight segment.
    new: Generate one-way sampling points along a straight segment (outbound only).

    Returns columns: route, leg, seq, x, y, distance_m
    """
    x0, y0 = p_start
    x1, y1 = p_end
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length <= 0:
        raise ValueError(f"Route {name} has zero length")

    n_seg = max(1, int(round(length / step)))
    t = np.linspace(0.0, 1.0, n_seg + 1)
    '''
    x_out = x0 + t * (x1 - x0)
    y_out = y0 + t * (y1 - y0)
    dist_out = t * length

    # Return trip: reverse, drop duplicate turning point
    x_back = x_out[-2::-1]
    y_back = y_out[-2::-1]
    dist_back = length + (length - dist_out[-2::-1])

    x = np.concatenate([x_out, x_back])
    y = np.concatenate([y_out, y_back])
    distance = np.concatenate([dist_out, dist_back])
    n_out = len(x_out)
    leg = np.array(["out"] * n_out + ["back"] * (len(x) - n_out))
    '''
    x = x0 + t * (x1 - x0)
    y = y0 + t * (y1 - y0)
    distance = t * length

    return pd.DataFrame(
        {
            "route": name,
            "leg": "out",
            "seq": np.arange(len(x), dtype=int),
            "x": x,
            "y": y,
            "distance_m": distance,
        }
    )


def build_all_routes(sample_step: float) -> dict[str, pd.DataFrame]:
    routes: dict[str, pd.DataFrame] = {}
    for name, spec in ROUTE_SPECS.items():
        routes[name] = build_route(name, spec["start"], spec["end"], sample_step)
    return routes


# ---------------------------------------------------------------------------
# 2. Binary STL → roof height query
# ---------------------------------------------------------------------------
class BuildingRoofIndex:
    """Approximate vertical-ray roof height from a binary STL mesh."""

    def __init__(self, triangles: np.ndarray, grid_size: float = 25.0):
        """
        Parameters
        ----------
        triangles : (N, 3, 3) float
            Triangle vertices [v0,v1,v2] each with (x,y,z).
        grid_size : float
            Spatial-hash cell size in metres.
        """
        self.triangles = triangles.astype(np.float64)
        self.n = len(self.triangles)
        self.grid_size = float(grid_size)

        # Per-triangle XY bbox and max roof z (vertex max is enough for LoD1 boxes)
        self.xy = self.triangles[:, :, :2]  # (N,3,2)
        self.z_max = self.triangles[:, :, 2].max(axis=1)
        self.xmin = self.xy[:, :, 0].min(axis=1)
        self.xmax = self.xy[:, :, 0].max(axis=1)
        self.ymin = self.xy[:, :, 1].min(axis=1)
        self.ymax = self.xy[:, :, 1].max(axis=1)

        # Spatial hash: cell -> triangle indices
        self._hash: dict[tuple[int, int], list[int]] = {}
        for i in range(self.n):
            ix0 = int(np.floor(self.xmin[i] / self.grid_size))
            ix1 = int(np.floor(self.xmax[i] / self.grid_size))
            iy0 = int(np.floor(self.ymin[i] / self.grid_size))
            iy1 = int(np.floor(self.ymax[i] / self.grid_size))
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    self._hash.setdefault((ix, iy), []).append(i)

    @staticmethod
    def _point_in_triangle_2d(px: float, py: float, tri_xy: np.ndarray) -> bool:
        """Barycentric point-in-triangle test (strict interior + boundary)."""
        (x0, y0), (x1, y1), (x2, y2) = tri_xy
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            return False
        a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        c = 1.0 - a - b
        return (a >= -1e-9) and (b >= -1e-9) and (c >= -1e-9)

    def query_roof_height(self, x: float, y: float) -> float:
        """Return maximum roof z covering (x,y), or 0 if no building."""
        ix = int(np.floor(x / self.grid_size))
        iy = int(np.floor(y / self.grid_size))
        candidates = self._hash.get((ix, iy), [])
        if not candidates:
            return 0.0
        h_max = 0.0
        for i in candidates:
            if not (self.xmin[i] <= x <= self.xmax[i] and self.ymin[i] <= y <= self.ymax[i]):
                continue
            if self._point_in_triangle_2d(x, y, self.xy[i]):
                if self.z_max[i] > h_max:
                    h_max = float(self.z_max[i])
        return h_max

    def query_many(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        out = np.zeros(len(xs), dtype=float)
        for i, (x, y) in enumerate(zip(xs, ys)):
            out[i] = self.query_roof_height(float(x), float(y))
        return out


def load_binary_stl_triangles(stl_path: Path) -> np.ndarray:
    """Parse a binary STL into an (N, 3, 3) array of triangle vertices."""
    with open(stl_path, "rb") as f:
        header = f.read(80)
        n_tri_bytes = f.read(4)
        if len(n_tri_bytes) < 4:
            raise ValueError(f"Invalid STL (too short): {stl_path}")
        n_tri = struct.unpack("<I", n_tri_bytes)[0]
        # Heuristic: if ASCII, header starts with 'solid'
        if header.lstrip().lower().startswith(b"solid") and n_tri == 0:
            raise ValueError(f"ASCII STL not supported: {stl_path}")

        tris = np.empty((n_tri, 3, 3), dtype=np.float32)
        for i in range(n_tri):
            data = f.read(50)
            if len(data) < 50:
                raise ValueError(f"Truncated STL at triangle {i}/{n_tri}")
            vals = struct.unpack("<12fH", data)  # normal(3) + 3 verts(9) + attr
            tris[i, 0] = vals[3:6]
            tris[i, 1] = vals[6:9]
            tris[i, 2] = vals[9:12]
    return tris


def load_building_roof_index(stl_path: Path, grid_size: float = 25.0) -> BuildingRoofIndex:
    print(f"[STL] Loading {stl_path} ...", flush=True)
    t0 = time.time()
    tris = load_binary_stl_triangles(stl_path)
    idx = BuildingRoofIndex(tris, grid_size=grid_size)
    print(
        f"[STL] {idx.n} triangles  "
        f"x=[{tris[:,:,0].min():.0f},{tris[:,:,0].max():.0f}]  "
        f"y=[{tris[:,:,1].min():.0f},{tris[:,:,1].max():.0f}]  "
        f"z=[{tris[:,:,2].min():.0f},{tris[:,:,2].max():.0f}]  "
        f"({time.time()-t0:.1f}s)",
        flush=True,
    )
    return idx


def annotate_obstruction(
    route_df: pd.DataFrame,
    roof_index: BuildingRoofIndex,
    target_heights: tuple[float, ...] = (30.0, 60.0, 120.0),
) -> pd.DataFrame:
    """Add building_height_m and obstructed_<h> columns."""
    df = route_df.copy()
    print(f"[STL] Querying roof height for {len(df)} points ({df['route'].iloc[0]}) ...", flush=True)
    t0 = time.time()
    heights = roof_index.query_many(df["x"].to_numpy(), df["y"].to_numpy())
    df["building_height_m"] = heights
    for h in target_heights:
        col = f"obstructed_{int(h)}"
        df[col] = heights >= h
    print(f"[STL] Done in {time.time()-t0:.1f}s", flush=True)
    return df


# ---------------------------------------------------------------------------
# 3. WRF extraction
# ---------------------------------------------------------------------------
def resolve_wrf_nc(dt: pd.Timestamp, explicit: Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if p.is_file():
            return p
        raise FileNotFoundError(p)

    # Split: <= 2025-09-06 → myExp03; later → myExp05
    split = pd.Timestamp("2025-09-06")
    exp = "W_myExp03" if dt.normalize() <= split.normalize() else "W_myExp05"
    base = REPO_ROOT / exp / "auxhist2"
    name = f"auxhist2_d03_{dt.strftime('%Y-%m-%d_%H:%M:%S')}_1h-rolling_cartesian.nc"
    candidates = [
        base / name,
        base / name.replace(":", "%3A"),
        base / name.replace(":", "%3a"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"WRF NetCDF not found for {dt}. Tried: {candidates}")


def extract_wrf_along_route(
    nc_path: Path,
    route_df: pd.DataFrame,
    heights: tuple[float, ...],
) -> pd.DataFrame:
    """Horizontally interpolate WRF U,V,W→WS at given z levels onto route points."""
    print(f"[WRF] Opening {nc_path.name} ...", flush=True)
    with xr.open_dataset(nc_path, mask_and_scale=False) as ds:
        x_coords = np.asarray(ds["x_rel"].values).squeeze().astype(float)
        y_coords = np.asarray(ds["y_rel"].values).squeeze().astype(float)
        z_coords = np.asarray(ds["z"].values).squeeze().astype(float)
        u_all = np.asarray(ds["U"].values).squeeze().astype(float)  # (z,y,x)
        v_all = np.asarray(ds["V"].values).squeeze().astype(float)
        has_w = "W" in ds
        w_all = np.asarray(ds["W"].values).squeeze().astype(float) if has_w else None

    pts = np.column_stack([route_df["y"].to_numpy(), route_df["x"].to_numpy()])
    out = route_df.copy()

    for h in heights:
        zi = int(np.argmin(np.abs(z_coords - h)))
        z_actual = float(z_coords[zi])
        if abs(z_actual - h) > 1e-6:
            print(f"[WRF] WARN: requested z={h}, nearest grid z={z_actual}")
        # RegularGridInterpolator expects axes in order matching data dims (y, x)
        fu = RegularGridInterpolator(
            (y_coords, x_coords), u_all[zi], method="linear", bounds_error=False, fill_value=np.nan
        )
        fv = RegularGridInterpolator(
            (y_coords, x_coords), v_all[zi], method="linear", bounds_error=False, fill_value=np.nan
        )
        u = fu(pts)
        v = fv(pts)
        tag = int(round(h))
        out[f"U_wrf_{tag}"] = u
        out[f"V_wrf_{tag}"] = v
        if w_all is not None:
            fw = RegularGridInterpolator(
                (y_coords, x_coords), w_all[zi], method="linear", bounds_error=False, fill_value=np.nan
            )
            out[f"W_wrf_{tag}"] = fw(pts)
        out[f"WS_wrf_{tag}"] = np.sqrt(u**2 + v**2)
        print(f"[WRF] z={tag}m  WS mean={np.nanmean(out[f'WS_wrf_{tag}']):.3f} m/s", flush=True)

    return out


# ---------------------------------------------------------------------------
# 4. CFD extraction (OpenFOAM ASCII fields + KDTree IDW)
# ---------------------------------------------------------------------------
def find_last_timestep(case_dir: Path) -> str:
    timesteps: list[int] = []
    for entry in case_dir.iterdir():
        if entry.is_dir():
            try:
                timesteps.append(int(entry.name))
            except ValueError:
                pass
    if not timesteps:
        raise FileNotFoundError(f"No numeric time directories under {case_dir}")
    return str(max(timesteps))


def _parse_vector_field(path: Path) -> np.ndarray:
    """Parse OpenFOAM volVectorField internalField into (N,3) array."""
    print(f"[CFD] Parsing vector field {path} ...", flush=True)
    t0 = time.time()
    content = path.read_text(errors="replace")
    m_vec = _VECTOR_FIELD_RE.search(content)
    if m_vec:
        n_expected = int(m_vec.group(1))
        raw = m_vec.group(2)
        vecs = _VEC_TOKEN_RE.findall(raw)
        arr = np.array([[float(a), float(b), float(c)] for a, b, c in vecs], dtype=np.float64)
        if len(arr) != n_expected:
            print(f"[CFD] WARN: expected {n_expected} vectors, got {len(arr)}")
        print(f"[CFD] Parsed {len(arr)} vectors in {time.time()-t0:.1f}s", flush=True)
        return arr

    m_uni = _UNIFORM_VECTOR_RE.search(content)
    if m_uni:
        return np.array(list(map(float, m_uni.group(1).split())), dtype=np.float64)

    raise ValueError(f"Cannot parse vector internalField: {path}")


def read_cell_centres(case_dir: Path, fallback: Path | None = None) -> np.ndarray:
    for candidate in (
        case_dir / "constant" / "cellCentres",
        case_dir / "0" / "C",
    ):
        if candidate.is_file():
            return _parse_vector_field(candidate)
    if fallback is not None and Path(fallback).is_file():
        print(f"[CFD] Using fallback cell centres: {fallback}", flush=True)
        return _parse_vector_field(Path(fallback))
    raise FileNotFoundError(
        f"Cannot find cell centres for {case_dir}. "
        "Provide --cell-centres or run: postProcess -func writeCellCentres -time 0"
    )


def resolve_cfd_case(dt: pd.Timestamp, explicit: Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if p.is_dir():
            return p
        raise FileNotFoundError(p)
    name = f"{dt.strftime('%Y%m%d_%H00')}_two_boundaries_as_outlet"
    p = REPO_ROOT / "steady_experiments_finer_ABL" / name
    if not p.is_dir():
        raise FileNotFoundError(p)
    return p


def idw_interpolate(
    tree: cKDTree,
    values: np.ndarray,
    query_xyz: np.ndarray,
    k: int = 12,
    power: float = 2.0,
    max_dist: float = 80.0,
) -> np.ndarray:
    """Inverse-distance-weighted interpolation at query points."""
    dists, idxs = tree.query(query_xyz, k=k, workers=-1)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    out = np.full(len(query_xyz), np.nan, dtype=float)
    for i in range(len(query_xyz)):
        d = dists[i]
        ii = idxs[i]
        # Keep neighbours within max_dist; if none, keep nearest
        mask = d <= max_dist
        if not np.any(mask):
            mask = np.zeros_like(d, dtype=bool)
            mask[0] = True
        d = d[mask]
        ii = ii[mask]
        # Exact hit
        if np.any(d < 1e-9):
            out[i] = float(values[ii[np.argmin(d)]])
            continue
        w = 1.0 / (d**power)
        w /= w.sum()
        out[i] = float(np.dot(w, values[ii]))
    return out


def load_cfd_velocity_band(
    case_dir: Path,
    heights: tuple[float, ...],
    cell_centres_fallback: Path | None = None,
    z_band_pad: float = 60.0,
) -> tuple[cKDTree, np.ndarray]:
    """Parse cell centres + U once; return KDTree and U on a vertical band."""
    last_ts = find_last_timestep(case_dir)
    print(f"[CFD] Case={case_dir.name}  last_ts={last_ts}", flush=True)

    cell_coords = read_cell_centres(case_dir, fallback=cell_centres_fallback)
    u_arr = _parse_vector_field(case_dir / last_ts / "U")
    if u_arr.ndim == 1:
        u_arr = np.tile(u_arr, (len(cell_coords), 1))
    if len(u_arr) != len(cell_coords):
        raise ValueError(
            f"Cell-centre count ({len(cell_coords)}) != U count ({len(u_arr)}). "
            "Mesh mismatch — check --cell-centres."
        )

    z_lo = min(heights) - z_band_pad
    z_hi = max(heights) + z_band_pad
    z_mask = (cell_coords[:, 2] >= z_lo) & (cell_coords[:, 2] <= z_hi)
    coords_band = cell_coords[z_mask]
    u_band = u_arr[z_mask]
    print(
        f"[CFD] Band z∈[{z_lo:.0f},{z_hi:.0f}]: {z_mask.sum()}/{len(cell_coords)} cells",
        flush=True,
    )
    return cKDTree(coords_band), u_band


def extract_cfd_along_route(
    route_df: pd.DataFrame,
    heights: tuple[float, ...],
    tree: cKDTree,
    u_band: np.ndarray,
    k_neighbors: int = 12,
) -> pd.DataFrame:
    """IDW-interpolate CFD U,V,W→WS at given heights onto route points."""
    out = route_df.copy()
    xs = route_df["x"].to_numpy()
    ys = route_df["y"].to_numpy()
    has_w = u_band.ndim == 2 and u_band.shape[1] >= 3

    for h in heights:
        query = np.column_stack([xs, ys, np.full(len(xs), h)])
        u = idw_interpolate(tree, u_band[:, 0], query, k=k_neighbors)
        v = idw_interpolate(tree, u_band[:, 1], query, k=k_neighbors)
        tag = int(round(h))
        out[f"U_cfd_{tag}"] = u
        out[f"V_cfd_{tag}"] = v
        if has_w:
            out[f"W_cfd_{tag}"] = idw_interpolate(tree, u_band[:, 2], query, k=k_neighbors)
        out[f"WS_cfd_{tag}"] = np.sqrt(u**2 + v**2)
        print(f"[CFD] z={tag}m  WS mean={np.nanmean(out[f'WS_cfd_{tag}']):.3f} m/s", flush=True)

    return out


# ---------------------------------------------------------------------------
# 5. Shear & merge
# ---------------------------------------------------------------------------
def add_vertical_shear(
    df: pd.DataFrame,
    target_height: float,
    shear_dz: float,
    prefix: str,
) -> pd.DataFrame:
    """Add WS_<prefix> and Vector_Shear_<prefix> using 3-D vector central difference.

    shear_magnitude = sqrt(du_dz^2 + dv_dz^2 + dw_dz^2), where
    d*_dz = (*_hi - *_lo) / shear_dz.  W is included when present.
    """
    out = df.copy()
    z_lo = int(round(target_height - shear_dz / 2.0))
    z_hi = int(round(target_height + shear_dz / 2.0))
    z_mid = int(round(target_height))
    u_lo, u_hi = f"U_{prefix}_{z_lo}", f"U_{prefix}_{z_hi}"
    v_lo, v_hi = f"V_{prefix}_{z_lo}", f"V_{prefix}_{z_hi}"
    w_lo, w_hi = f"W_{prefix}_{z_lo}", f"W_{prefix}_{z_hi}"
    ws_lo = f"WS_{prefix}_{z_lo}"
    ws_hi = f"WS_{prefix}_{z_hi}"
    ws_mid = f"WS_{prefix}_{z_mid}"
    for col in (u_lo, u_hi, v_lo, v_hi):
        if col not in out.columns:
            raise KeyError(f"Missing column {col} for vector shear of {prefix}")
    if ws_lo not in out.columns or ws_hi not in out.columns:
        raise KeyError(f"Missing columns {ws_lo}/{ws_hi} for shear of {prefix}")
    out[f"WS_{prefix}"] = out[ws_mid] if ws_mid in out.columns else 0.5 * (out[ws_lo] + out[ws_hi])
    du_dz = (out[u_hi] - out[u_lo]) / shear_dz
    dv_dz = (out[v_hi] - out[v_lo]) / shear_dz
    if w_lo in out.columns and w_hi in out.columns:
        dw_dz = (out[w_hi] - out[w_lo]) / shear_dz
    else:
        dw_dz = 0.0
    out[f"Vector_Shear_{prefix}"] = np.sqrt(du_dz**2 + dv_dz**2 + dw_dz**2)
    return out


def obstruction_summary(dfs: dict[str, pd.DataFrame], heights: tuple[float, ...] = (30, 60, 120)) -> pd.DataFrame:
    rows = []
    for name, df in dfs.items():
        n = len(df)
        row: dict = {
            "route": name,
            "n_points": n,
            "max_building_height_m": float(df["building_height_m"].max()) if n else 0.0,
        }
        for h in heights:
            col = f"obstructed_{int(h)}"
            n_obs = int(df[col].sum()) if col in df.columns else 0
            row[f"n_obstructed_{int(h)}"] = n_obs
            row[f"frac_obstructed_{int(h)}"] = n_obs / n if n else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------
def _draw_building_profile(
    ax,
    distance_km: np.ndarray,
    building_height_m: np.ndarray,
    *,
    ref_height: float | None = None,
    label: str = "Building height",
):
    """Stamp along-track building heights onto the lower part of ``ax`` via twin y.

    Returns the twin Axes (or None if there is nothing to draw).
    """
    h = np.asarray(building_height_m, dtype=float)
    d = np.asarray(distance_km, dtype=float)
    if h.size == 0:
        return None

    h_plot = np.nan_to_num(h, nan=0.0)
    h_max = float(np.nanmax(h_plot)) if h_plot.size else 0.0
    refs: list[float] = []
    if ref_height is not None and ref_height > 0:
        refs = [float(ref_height)]
    # Always reserve a twin axis so panels share the same bottom context;
    # empty routes just show a flat zero baseline.
    ymax = max(h_max * 1.15, max(refs, default=0.0) * 1.25, 40.0)

    ax_b = ax.twinx()
    ax_b.fill_between(
        d,
        0.0,
        h_plot,
        step="mid",
        color="#8D6E4A",
        alpha=0.40,
        linewidth=0.0,
        zorder=0,
        label=label if h_max > 0 else None,
    )
    if h_max > 0:
        ax_b.plot(d, h_plot, color="#5D4037", lw=0.6, alpha=0.7, zorder=1, solid_capstyle="butt")
    for rh in refs:
        ax_b.axhline(
            rh,
            color="#5D4037",
            ls="--",
            lw=0.9,
            alpha=0.65,
            zorder=1,
            label=f"z = {int(rh)} m",
        )
    ax_b.set_ylim(0.0, ymax)
    ax_b.set_ylabel(r"$H_\mathrm{b}$ (m)", color="#5D4037", fontsize=10)
    ax_b.tick_params(axis="y", colors="#5D4037", labelsize=9)
    ax_b.grid(False)

    # Let the building fill sit visually under the wind/shear curves
    ax.set_zorder(ax_b.get_zorder() + 1)
    ax.patch.set_visible(False)
    return ax_b


def _collect_legend_entries(ax, ax_extra=None) -> tuple[list, list]:
    handles, labels = ax.get_legend_handles_labels()
    if ax_extra is not None:
        h2, l2 = ax_extra.get_legend_handles_labels()
        for hh, ll in zip(h2, l2):
            if ll:
                handles.append(hh)
                labels.append(ll)
    return handles, labels


def _dedupe_legend_entries(handles, labels) -> tuple[list, list]:
    seen: set[str] = set()
    out_h, out_l = [], []
    for h, lab in zip(handles, labels):
        if not lab or lab in seen:
            continue
        seen.add(lab)
        out_h.append(h)
        out_l.append(lab)
    return out_h, out_l


def format_datetime_utc8(dt: pd.Timestamp) -> str:
    """Convert naive/aware UTC datetime to ``YYYY-mm-dd HH:MM (UTC+8)``."""
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    local = ts.tz_convert("Asia/Shanghai")
    return local.strftime("%Y-%m-%d %H:%M (UTC+8)")


def plot_figure1_wind_speed(
    route_dfs: dict[str, pd.DataFrame],
    out_path: Path,
    target_height: float,
    datetime_utc: pd.Timestamp | None = None,
) -> None:
    configure_matplotlib_style()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=False)
    route_order = list(ROUTE_SPECS.keys())

    all_handles: list = []
    all_labels: list = []

    for ax, rname in zip(axes, route_order):
        df = route_dfs[rname]
        spec = ROUTE_SPECS[rname]
        dist_km = df["distance_m"].to_numpy() / 1000.0
        bh = (
            df["building_height_m"].to_numpy()
            if "building_height_m" in df.columns
            else np.zeros(len(df))
        )
        ax_b = _draw_building_profile(
            ax, dist_km, bh, ref_height=target_height, label="Building height"
        )

        ax.plot(
            dist_km,
            df["WS_wrf"],
            color="#4C78A8",
            lw=2.4,
            alpha=0.85,
            label="WRF",
            zorder=2,
        )
        ax.plot(
            dist_km,
            df["WS_cfd"],
            color="#E45756",
            lw=1.1,
            alpha=0.95,
            label="WRF-to-OpenFOAM",
            zorder=3,
        )
        ax.set_ylabel(r"$U$ (m/s)")
        ax.set_title(spec["label"])
        ax.set_xlim(dist_km.min(), dist_km.max())
        ax.set_xlabel("Distance along route (km)")

        h, lab = _collect_legend_entries(ax, ax_b)
        all_handles.extend(h)
        all_labels.extend(lab)

    all_handles, all_labels = _dedupe_legend_entries(all_handles, all_labels)
    wind_h, wind_l, other_h, other_l = [], [], [], []
    for h, lab in zip(all_handles, all_labels):
        if lab.startswith("WRF"):
            wind_h.append(h)
            wind_l.append(lab)
        else:
            other_h.append(h)
            other_l.append(lab)
    fig.legend(
        wind_h + other_h,
        wind_l + other_l,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )

    time_txt = format_datetime_utc8(datetime_utc) if datetime_utc is not None else None
    if time_txt:
        fig.suptitle(
            f"Along-track horizontal wind speed  (z={int(target_height)} m)\n{time_txt}",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
    else:
        fig.suptitle(
            f"Along-track horizontal wind speed  (z={int(target_height)} m)",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Plot] Saved {out_path}", flush=True)


def plot_figure2_shear(
    route_dfs: dict[str, pd.DataFrame],
    out_path: Path,
    target_height: float,
    shear_dz: float,
    datetime_utc: pd.Timestamp | None = None,
) -> None:
    configure_matplotlib_style()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=False)
    route_order = list(ROUTE_SPECS.keys())

    all_handles: list = []
    all_labels: list = []

    for ax, rname in zip(axes, route_order):
        df = route_dfs[rname]
        spec = ROUTE_SPECS[rname]
        dist_km = df["distance_m"].to_numpy() / 1000.0
        bh = (
            df["building_height_m"].to_numpy()
            if "building_height_m" in df.columns
            else np.zeros(len(df))
        )
        ax_b = _draw_building_profile(
            ax, dist_km, bh, ref_height=target_height, label="Building height"
        )

        ax.plot(
            dist_km,
            df["Vector_Shear_wrf"],
            color="#4C78A8",
            lw=2.4,
            alpha=0.85,
            label="WRF",
            zorder=2,
        )
        ax.plot(
            dist_km,
            df["Vector_Shear_cfd"],
            color="#E45756",
            lw=1.1,
            alpha=0.95,
            label="WRF-to-OpenFOAM",
            zorder=3,
        )

        ax.set_ylabel("Vertical Wind Shear Magnitude (1/s)")
        ax.set_title(spec["label"])
        ax.set_xlim(dist_km.min(), dist_km.max())
        ax.axhline(0.0, color="0.4", lw=0.6, zorder=1)
        ax.set_xlabel("Distance along route (km)")

        h, lab = _collect_legend_entries(ax, ax_b)
        all_handles.extend(h)
        all_labels.extend(lab)

    all_handles, all_labels = _dedupe_legend_entries(all_handles, all_labels)
    shear_h, shear_l, other_h, other_l = [], [], [], []
    for h, lab in zip(all_handles, all_labels):
        if lab.startswith("WRF"):
            shear_h.append(h)
            shear_l.append(lab)
        else:
            other_h.append(h)
            other_l.append(lab)
    fig.legend(
        shear_h + other_h,
        shear_l + other_l,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    time_txt = format_datetime_utc8(datetime_utc) if datetime_utc is not None else None
    if time_txt:
        fig.suptitle(
            f"Along-track vertical wind shear (z={int(target_height)} m)\n{time_txt}",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
    else:
        fig.suptitle(
            "Along-track vertical wind shear",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Plot] Saved {out_path}", flush=True)


def origin_utm49n(lon: float, lat: float) -> tuple[float, float]:
    """WGS84 lon/lat (deg) → UTM Zone 49N easting/northing (m)."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    ox, oy = transformer.transform(lon, lat)
    return float(ox), float(oy)


def _rings_from_shape(shp) -> list[np.ndarray]:
    """Extract exterior rings from a pyshp polygon (skip hole rings).

    ESRI shapefiles in this dataset use clockwise exteriors; holes (if any)
    are subsequent parts. Nearly all buildings here are single-ring polygons.
    """
    pts = np.asarray(shp.points, dtype=float)
    if pts.size == 0:
        return []
    parts = list(shp.parts) + [len(pts)]
    rings: list[np.ndarray] = []
    # First ring = exterior; later rings = holes (omit for filled basemap)
    for i in range(len(parts) - 1):
        if i > 0:
            break
        ring = pts[parts[i] : parts[i + 1]]
        if ring.shape[0] >= 2 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if ring.shape[0] < 3:
            continue
        x, y = ring[:, 0], ring[:, 1]
        area = abs(0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))
        if area < 5.0:  # m² — drop slivers
            continue
        rings.append(ring)
    return rings


def load_building_footprints_local(
    shp_path: Path,
    origin_xy: tuple[float, float],
    clip_xy: float,
    height_field: str = "jzgd",
    encoding: str = "gbk",
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load building rings in OpenFOAM/STL local XY, clipped to ±clip_xy.

    Returns
    -------
    rings : list of (N, 2) arrays in local metres
    heights : (len(rings),) building height (m), one value per ring
    """
    import shapefile

    ox, oy = origin_xy
    reader = shapefile.Reader(str(shp_path), encoding=encoding)
    field_names = [f[0] for f in reader.fields[1:]]
    if height_field not in field_names:
        raise ValueError(f"Height field '{height_field}' not in {field_names}")
    hi = field_names.index(height_field)

    rings: list[np.ndarray] = []
    heights: list[float] = []
    pad = 50.0  # keep buildings that only slightly poke into the view

    for shp, rec in zip(reader.shapes(), reader.records()):
        if shp.shapeType not in (5, 15, 25):  # POLYGON / Z / M
            continue
        try:
            h = float(rec[hi])
        except (TypeError, ValueError):
            h = float("nan")
        if not np.isfinite(h):
            continue
        for ring_utm in _rings_from_shape(shp):
            ring = np.column_stack([ring_utm[:, 0] - ox, ring_utm[:, 1] - oy])
            if (
                ring[:, 0].max() < -clip_xy - pad
                or ring[:, 0].min() > clip_xy + pad
                or ring[:, 1].max() < -clip_xy - pad
                or ring[:, 1].min() > clip_xy + pad
            ):
                continue
            rings.append(ring)
            heights.append(h)

    print(
        f"[SHP] {len(rings)} footprint rings within ±{clip_xy:.0f} m "
        f"from {shp_path.name}",
        flush=True,
    )
    return rings, np.asarray(heights, dtype=float)


def plot_route_overview(
    route_dfs: dict[str, pd.DataFrame],
    shp_path: Path,
    out_path: Path,
    clip_xy: float = 1800.0,
    origin_lon: float = DEFAULT_ORIGIN_LON,
    origin_lat: float = DEFAULT_ORIGIN_LAT,
) -> None:
    """Top-down map of routes on SHP building footprints (height-coloured)."""
    configure_matplotlib_style()
    # Lock one face for every text artist (avoids bold/regular fallback mismatch)
    font_family = "Times New Roman"
    ox, oy = origin_utm49n(origin_lon, origin_lat)
    rings, heights = load_building_footprints_local(Path(shp_path), (ox, oy), clip_xy)

    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    fig.patch.set_facecolor("white")
    # Cool open-space ground so warm building fills read clearly
    ax.set_facecolor("#e6ebf0")

    cbar = None
    if len(rings) == 0:
        print("[SHP] WARNING: no footprints in view; drawing routes only", flush=True)
    else:
        # Clip colour scale at P96 so a few towers do not wash out the city
        vmax = float(np.nanpercentile(heights, 96))
        vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = max(float(np.nanmax(heights)), vmin + 1.0)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        # Truncate Oranges so the lightest fill is still darker than the ground
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "urban_ht",
            ["#f0d5a8", "#e09a3e", "#c45c26", "#7a2e12"],
            N=256,
        )
        patches = [MplPolygon(r, closed=True) for r in rings]
        coll = PatchCollection(
            patches,
            cmap=cmap,
            norm=norm,
            array=heights,
            edgecolors="#3d342c",
            linewidths=0.18,
            alpha=0.95,
            zorder=1,
        )
        ax.add_collection(coll)
        cbar = fig.colorbar(coll, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Building height (m)", fontsize=20, fontfamily=font_family)
        cbar.ax.tick_params(labelsize=16)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontfamily(font_family)

    route_colors = {"Route1_open_river": "#0D47A1", "Route2_urban_canyon": "#B71C1C"}
    # Text offsets: Route 1 (E–W) label above; Route 2 (N–S) label to the right
    endpoint_ann = {
        "Route1_open_river": {
            "start": {"xytext": (0, 22), "ha": "center", "va": "bottom"},
            "end": {"xytext": (0, 22), "ha": "center", "va": "bottom"},
        },
        "Route2_urban_canyon": {
            "start": {"xytext": (22, 0), "ha": "left", "va": "center"},
            "end": {"xytext": (22, 0), "ha": "left", "va": "center"},
        },
    }
    for rname, df in route_dfs.items():
        color = route_colors.get(rname, "k")
        ax.plot(
            df["x"],
            df["y"],
            color=color,
            lw=2.6,
            solid_capstyle="round",
            label=ROUTE_SPECS[rname]["label"],
            zorder=3,
        )
        x0, y0 = float(df["x"].iloc[0]), float(df["y"].iloc[0])
        x1, y1 = float(df["x"].iloc[-1]), float(df["y"].iloc[-1])
        for x, y in ((x0, y0), (x1, y1)):
            ax.scatter(
                x,
                y,
                c="k",
                s=40,
                zorder=4,
                edgecolors="white",
                linewidths=0.7,
            )
        ann = endpoint_ann.get(
            rname,
            {
                "start": {"xytext": (0, 22), "ha": "center", "va": "bottom"},
                "end": {"xytext": (0, 22), "ha": "center", "va": "bottom"},
            },
        )
        for label, (x, y), key in (
            ("Start", (x0, y0), "start"),
            ("End", (x1, y1), "end"),
        ):
            sty = ann[key]
            ax.annotate(
                label,
                xy=(x, y),
                xytext=sty["xytext"],
                textcoords="offset points",
                color=color,
                fontsize=16,
                fontfamily=font_family,
                fontweight="bold",
                ha=sty["ha"],
                va=sty["va"],
                zorder=5,
            )

    ax.set_xlim(-clip_xy, clip_xy)
    ax.set_ylim(-clip_xy, clip_xy)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=20, fontfamily=font_family)
    ax.set_ylabel("y (m)", fontsize=20, fontfamily=font_family)
    ax.tick_params(axis="both", labelsize=18)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily(font_family)
    ax.set_title(
        "UAV route overview with building footprints",
        fontsize=18,
        fontfamily=font_family,
        fontweight="bold",
    )
    ax.grid(True, color="0.45", alpha=0.22, linestyle=":", linewidth=0.7)
    ax.legend(
        loc="upper left",
        fontsize=16,
        framealpha=0.92,
        prop={"family": font_family, "size": 16},
    )
    fig.tight_layout()
    # Re-assert family after layout (tick labels can be rebuilt)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily(font_family)
    if cbar is not None:
        cbar.ax.yaxis.label.set_fontfamily(font_family)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontfamily(font_family)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datetime", default=DEFAULT_DATETIME, help="UTC datetime YYYY-MM-DD HH:MM:SS")
    p.add_argument("--height", type=float, default=DEFAULT_HEIGHT, help="Analysis height (m), default 120")
    p.add_argument("--sample-step", type=float, default=DEFAULT_SAMPLE_STEP, help="Route sampling step (m)")
    p.add_argument("--shear-dz", type=float, default=DEFAULT_SHEAR_DZ, help="Shear layer thickness Δz (m)")
    p.add_argument("--wrf-nc", type=Path, default=None, help="Explicit WRF cartesian NetCDF path")
    p.add_argument("--cfd-case-dir", type=Path, default=None, help="Explicit OpenFOAM case directory")
    p.add_argument("--cell-centres", type=Path, default=DEFAULT_CELL_CENTRES, help="Fallback cellCentres / 0/C")
    p.add_argument("--stl-path", type=Path, default=DEFAULT_STL, help="Buildings binary STL (obstruction)")
    p.add_argument(
        "--shp-path",
        type=Path,
        default=DEFAULT_SHP,
        help="Building footprints SHP for route overview basemap",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory under results/")
    p.add_argument("--skip-overview", action="store_true", help="Skip route overview map")
    return p.parse_args()


def default_out_dir(dt: pd.Timestamp) -> Path:
    tag = dt.strftime("%Y%m%d_%H%M")
    return REPO_ROOT / "results" / "uav_route_wind_shear" / tag


def main() -> int:
    args = parse_args()
    dt = pd.Timestamp(args.datetime)
    target_h = float(args.height)
    shear_dz = float(args.shear_dz)
    z_lo = target_h - shear_dz / 2.0
    z_hi = target_h + shear_dz / 2.0
    # Shear needs neighbours of --height
    heights = tuple(sorted({float(z_lo), float(target_h), float(z_hi)}))

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(dt)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("UAV route wind / shear analysis")
    print(f"  datetime     : {dt}")
    print(f"  height       : {target_h} m")
    print(f"  shear Δz     : {shear_dz} m  → layers {(z_lo, target_h, z_hi)}")
    print(f"  extract z    : {heights}")
    print(f"  sample step  : {args.sample_step} m")
    print(f"  out_dir      : {out_dir}")
    print("=" * 64)

    # --- Routes ---
    routes = build_all_routes(args.sample_step)
    for name, df in routes.items():
        print(f"[Route] {name}: {len(df)} pts, total distance={df['distance_m'].iloc[-1]:.0f} m")

    # --- Buildings ---
    roof_index = load_building_roof_index(Path(args.stl_path))
    for name in list(routes.keys()):
        routes[name] = annotate_obstruction(routes[name], roof_index, (30.0, 60.0, 120.0))

    # --- WRF (open once, interpolate both routes) ---
    wrf_nc = resolve_wrf_nc(dt, args.wrf_nc)
    for name in list(routes.keys()):
        routes[name] = extract_wrf_along_route(wrf_nc, routes[name], heights)
        routes[name] = add_vertical_shear(routes[name], target_h, shear_dz, "wrf")

    # --- CFD (parse mesh/U once, reuse KDTree for all routes) ---
    cfd_case = resolve_cfd_case(dt, args.cfd_case_dir)
    tree, u_band = load_cfd_velocity_band(
        cfd_case, heights, cell_centres_fallback=args.cell_centres
    )
    for name in list(routes.keys()):
        routes[name] = extract_cfd_along_route(routes[name], heights, tree, u_band)
        routes[name] = add_vertical_shear(routes[name], target_h, shear_dz, "cfd")

    # Height tag so different --height runs do not overwrite each other
    z_tag = f"z{int(round(target_h))}"

    # --- Save CSVs ---
    csv_map = {
        "Route1_open_river": f"route1_along_track_{z_tag}.csv",
        "Route2_urban_canyon": f"route2_along_track_{z_tag}.csv",
    }
    keep_cols_base = [
        "route",
        "leg",
        "seq",
        "x",
        "y",
        "distance_m",
        "building_height_m",
        "obstructed_30",
        "obstructed_60",
        "obstructed_120",
        "WS_wrf",
        "WS_cfd",
        "Vector_Shear_wrf",
        "Vector_Shear_cfd",
    ]
    # Also keep height-specific columns for transparency
    extra = []
    for h in heights:
        tag = int(round(h))
        for prefix in ("wrf", "cfd"):
            for var in ("U", "V", "W", "WS"):
                col = f"{var}_{prefix}_{tag}"
                extra.append(col)

    for name, fname in csv_map.items():
        df = routes[name]
        cols = [c for c in keep_cols_base + extra if c in df.columns]
        path = out_dir / fname
        df[cols].to_csv(path, index=False)
        print(f"[CSV] {path}  shape={df[cols].shape}")

    summary = obstruction_summary(routes)
    summary_path = out_dir / "building_obstruction_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[CSV] {summary_path}")
    print(summary.to_string(index=False))

    # --- Figures ---
    plot_figure1_wind_speed(
        routes,
        out_dir / f"fig1_wind_speed_along_route_{z_tag}.png",
        target_h,
        datetime_utc=dt,
    )
    plot_figure2_shear(
        routes,
        out_dir / f"fig2_vertical_wind_shear_along_route_{z_tag}.png",
        target_h,
        shear_dz,
        datetime_utc=dt,
    )
    if not args.skip_overview:
        plot_route_overview(routes, Path(args.shp_path), out_dir / "route_overview_map.png")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
