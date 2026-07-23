#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WRF 域面风场 + 边界出入口诊断图。

输入（任选其一，只需给一次时刻/算例标识）：
  - 紧凑时刻：20250903_1200
  - construct 时刻：09-03_12:00
  - 算例目录：20250903_1200_two_boundaries_as_outlet
  - cartesian NC 路径（兼容旧用法）

自动推断：
  - cartesian NC：W_myExp03/auxhist2/auxhist2_d03_<时刻>_1h-rolling_cartesian.nc
  - wrfout_d01：W_myExp03/WRF/run/wrfout_d01_<时刻>
  - boundaryData：<算例>/constant/boundaryData

输出 PNG（默认）:
  results/wrf_openfoam/domain_surface_inlet_diag/
    domain_surface_inlet_diag_<YYYYMMDD_HHMM>.png
  可用 -o/--output 覆盖。

用法:
  python util/plot_wrf_domain_surface_inlet_diag.py 20250903_1200
  python util/plot_wrf_domain_surface_inlet_diag.py 09-03_12:00 -o results/wrf_openfoam/foo.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import xarray as xr
from pyproj import Proj

from auto_detect_inlet_diag import (
    PATCHES,
    DEFAULT_Z_MAX,
    format_boundary_flux_report,
    read_boundary_wind_data,
)

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    import netCDF4 as nc
    import wrf
    HAS_WRF = True
except ImportError:
    HAS_WRF = False

LON0 = 113.32
LAT0 = 23.115
REGION_A_LON = (110.0, 120.0)
REGION_A_LAT = (17.0, 25.0)
REGION_B_XY = (-5000.0, 5000.0)
DEFAULT_CASE_SUFFIX = '_two_boundaries_as_outlet'
DEFAULT_EXPERIMENTS_DIR = 'steady_experiments_finer_ABL'
WRF_AUXHIST_DIR = _REPO_ROOT / 'W_myExp05' / 'auxhist2'
WRF_RUN_DIR = _REPO_ROOT / 'W_myExp05' / 'WRF' / 'run'
DEFAULT_WRF_YEAR = 2025
RESULTS_DIAG_DIR = _REPO_ROOT / 'results' / 'wrf_openfoam' / 'domain_surface_inlet_diag'


@dataclass(frozen=True)
class RunPaths:
    """一次运行所需的输入路径（由时刻/算例标识自动解析）。"""
    label: str
    wrf_iso: str
    cartesian_nc: Path
    wrfout_d01: Path
    boundary_data: Path | None
    case_dir: Path | None


_COASTLINE_CACHE = _SCRIPT_DIR / '.cache' / 'ne_110m_coastline.geojson'
_NE_COASTLINE_URL = (
    'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/'
    'master/geojson/ne_110m_coastline.geojson'
)


def _destagger_np(arr: np.ndarray, axis: int) -> np.ndarray:
    slc_lo = [slice(None)] * arr.ndim
    slc_hi = [slice(None)] * arr.ndim
    slc_lo[axis] = slice(None, -1)
    slc_hi[axis] = slice(1, None)
    return 0.5 * (arr[tuple(slc_lo)] + arr[tuple(slc_hi)])


def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
        'grid.color': 'black',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def _get_array(ds: xr.Dataset, name: str) -> np.ndarray:
    arr = ds[name].values
    if arr.ndim >= 1 and 'Time' in ds[name].dims:
        return arr[0]
    return arr


def _is_cartesian(ds: xr.Dataset) -> bool:
    return 'x_rel' in ds.coords and 'y_rel' in ds.coords and 'z' in ds.coords


def _lowest_wind_level(ds: xr.Dataset) -> tuple[int, str]:
    """cartesian 取第一个 z>0；原生 WRF 取 bottom_top=0。"""
    if _is_cartesian(ds):
        z = np.asarray(ds['z'].values, dtype=float)
        for i, zi in enumerate(z):
            if zi > 0:
                return i, f"z={zi:g} m"
        return 0, f"z={z[0]:g} m"
    return 0, "lowest mass level"


def resolve_input_path(path: str | os.PathLike[str]) -> Path:
    """解析相对路径（相对 cwd）与 repo 内路径。"""
    p = Path(path)
    if p.is_file():
        return p.resolve()
    if not p.is_absolute():
        cand = (Path.cwd() / p).resolve()
        if cand.is_file():
            return cand
    cand = (_REPO_ROOT / p).resolve()
    if cand.is_file():
        return cand
    return p.resolve()


def resolve_existing_dir(path: str | os.PathLike[str]) -> Path | None:
    """解析存在的目录（cwd / repo 相对）。"""
    p = Path(path)
    candidates = [p]
    if not p.is_absolute():
        candidates.append(Path.cwd() / p)
        candidates.append(_REPO_ROOT / p)
    for cand in candidates:
        cand = cand.resolve()
        if cand.is_dir():
            return cand
    return None


def _first_existing_file(candidates: list[Path]) -> Path | None:
    for cand in candidates:
        cand = cand.resolve()
        if cand.is_file():
            return cand
    return None


def _wrf_iso_to_construct_label(wrf_iso: str) -> str:
    """2025-09-03_12:00:00 -> 09-03_12:00"""
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})_(\d{2}):(\d{2}):\d{2}', wrf_iso)
    if not m:
        raise ValueError(f"无法解析 WRF ISO 时刻: {wrf_iso}")
    return f"{m.group(2)}-{m.group(3)}_{m.group(4)}:{m.group(5)}"


def _format_utc_timestamp(wrf_iso: str) -> str:
    """2025-09-03_12:00:00 -> 2025-09-03 12:00 (UTC)"""
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})_(\d{2}):(\d{2}):\d{2}', wrf_iso)
    if not m:
        return f"{wrf_iso} (UTC)"
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)} (UTC)"


def parse_run_timestamp(target: str, *, default_year: int = DEFAULT_WRF_YEAR) -> tuple[str, str]:
    """
    从多种标识解析 (wrf_iso, compact_label)。

    支持：
      - 20250903_1200
      - 2025-09-03_12:00:00
      - 09-03_12:00
      - 算例目录名 20250903_1200_two_boundaries_as_outlet
    """
    text = target.strip().rstrip('/')

    m = re.fullmatch(r'(\d{8})_(\d{4})', text)
    if m:
        y, mo, d, h, mi = m.group(1)[:4], m.group(1)[4:6], m.group(1)[6:8], m.group(2)[:2], m.group(2)[2:4]
        wrf_iso = f"{y}-{mo}-{d}_{h}:{mi}:00"
        return wrf_iso, f"{m.group(1)}_{m.group(2)}"

    m = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', text)
    if m:
        wrf_iso = m.group(1)
        compact = wrf_iso.replace('-', '').replace(':', '')[:13]
        compact = f"{compact[:8]}_{compact[8:]}"
        return wrf_iso, compact

    m = re.fullmatch(r'(\d{2})-(\d{2})_(\d{2}):(\d{2})', text)
    if m:
        mo, d, h, mi = m.groups()
        wrf_iso = f"{default_year}-{mo}-{d}_{h}:{mi}:00"
        compact = f"{default_year}{mo}{d}_{h}{mi}"
        return wrf_iso, compact

    m = re.search(r'(\d{8})_(\d{4})', text)
    if m:
        y, mo, d, h, mi = m.group(1)[:4], m.group(1)[4:6], m.group(1)[6:8], m.group(2)[:2], m.group(2)[2:4]
        wrf_iso = f"{y}-{mo}-{d}_{h}:{mi}:00"
        return wrf_iso, f"{m.group(1)}_{m.group(2)}"

    raise ValueError(
        f"无法从 '{target}' 解析时刻。请用 20250903_1200、09-03_12:00、"
        f"算例目录名或 cartesian NC 路径。"
    )


def cartesian_nc_candidates(wrf_iso: str) -> list[Path]:
    stem = f'auxhist2_d03_{wrf_iso}_1h-rolling_cartesian.nc'
    stem_enc = stem.replace(':', '%3A')
    return [
        WRF_AUXHIST_DIR / stem,
        WRF_AUXHIST_DIR / stem_enc,
        Path.cwd() / '..' / 'W_myExp03' / 'auxhist2' / stem,
        Path.cwd() / '..' / 'W_myExp03' / 'auxhist2' / stem_enc,
    ]


def wrfout_d01_candidates(wrf_iso: str) -> list[Path]:
    name = f'wrfout_d01_{wrf_iso}'
    return [
        WRF_RUN_DIR / name,
        Path.cwd() / '..' / 'W_myExp03' / 'WRF' / 'run' / name,
    ]


def case_dir_candidates(
    compact: str,
    experiments_dir: Path,
    case_suffix: str,
) -> list[Path]:
    case_name = f'{compact}{case_suffix}'
    return [
        Path.cwd() / case_name,
        experiments_dir / case_name,
        _REPO_ROOT / DEFAULT_EXPERIMENTS_DIR / case_name,
    ]


def infer_boundary_data(case_dir: Path | None) -> Path | None:
    if case_dir is None:
        return None
    bd = case_dir / 'constant' / 'boundaryData'
    return bd if bd.is_dir() else None


def resolve_run_paths(
    target: str,
    *,
    experiments_dir: Path,
    case_suffix: str,
    cartesian_nc: str | None = None,
    wrfout_a: str | None = None,
    boundary_data: str | None = None,
) -> RunPaths:
    """从单一 target（或显式覆盖项）解析全部输入路径。"""
    target_path = Path(target)
    case_dir: Path | None = None
    wrf_iso: str | None = None
    compact: str | None = None

    if cartesian_nc:
        nc_path = resolve_input_path(cartesian_nc)
        if not nc_path.is_file():
            raise FileNotFoundError(f"找不到 cartesian NC: {cartesian_nc}")
        m = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', nc_path.name)
        if not m:
            raise ValueError(f"无法从 NC 文件名解析时刻: {nc_path.name}")
        wrf_iso = m.group(1)
        compact = wrf_iso.replace('-', '').replace(':', '')[:13]
        compact = f"{compact[:8]}_{compact[8:]}"
    elif target_path.suffix == '.nc' or str(target).endswith('.nc'):
        nc_path = resolve_input_path(target)
        if not nc_path.is_file():
            raise FileNotFoundError(f"找不到 cartesian NC: {target}")
        m = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', nc_path.name)
        if not m:
            raise ValueError(f"无法从 NC 文件名解析时刻: {nc_path.name}")
        wrf_iso = m.group(1)
        compact = wrf_iso.replace('-', '').replace(':', '')[:13]
        compact = f"{compact[:8]}_{compact[8:]}"
    else:
        resolved_dir = resolve_existing_dir(target)
        if resolved_dir is not None:
            case_dir = resolved_dir
            wrf_iso, compact = parse_run_timestamp(resolved_dir.name)
        else:
            wrf_iso, compact = parse_run_timestamp(target)

        nc_path = _first_existing_file(cartesian_nc_candidates(wrf_iso))
        if nc_path is None:
            raise FileNotFoundError(
                f"找不到 cartesian NC（时刻 {wrf_iso}）。"
                f"已尝试: {cartesian_nc_candidates(wrf_iso)[0]}"
            )

    if wrfout_a:
        wrfout_path = resolve_input_path(wrfout_a)
        if not wrfout_path.is_file():
            raise FileNotFoundError(f"找不到 wrfout 文件: {wrfout_a}")
    else:
        wrfout_path = _first_existing_file(wrfout_d01_candidates(wrf_iso))
        if wrfout_path is None:
            raise FileNotFoundError(
                f"找不到 wrfout_d01（时刻 {wrf_iso}）。"
                f"已尝试: {wrfout_d01_candidates(wrf_iso)[0]}"
            )

    if case_dir is None:
        for cand in case_dir_candidates(compact, experiments_dir, case_suffix):
            if cand.is_dir() and (cand / 'constant').is_dir():
                case_dir = cand.resolve()
                break

    if boundary_data:
        bd_path = resolve_existing_dir(boundary_data)
        if bd_path is None:
            raise FileNotFoundError(f"找不到 boundaryData 目录: {boundary_data}")
    else:
        bd_path = infer_boundary_data(case_dir)

    return RunPaths(
        label=compact,
        wrf_iso=wrf_iso,
        cartesian_nc=nc_path,
        wrfout_d01=wrfout_path,
        boundary_data=bd_path,
        case_dir=case_dir,
    )


def _read_wrf_2d(dataset, name: str, level_idx: int = 0, time_idx: int = 0) -> np.ndarray:
    var = dataset.variables[name]
    slices = []
    for dim in var.dimensions:
        if dim in ('Time', 'time'):
            slices.append(time_idx)
        elif dim == 'bottom_top':
            slices.append(level_idx)
        else:
            slices.append(slice(None))
    return np.asarray(var[tuple(slices)])


def load_wrfout_surface_wind(
    wrfout_path: str | os.PathLike[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """从 wrfout 读取近地面风场（优先 10 m）。"""
    path = resolve_input_path(wrfout_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 wrfout 文件: {wrfout_path}")

    if HAS_WRF:
        dataset = nc.Dataset(str(path))
        try:
            try:
                uv = wrf.getvar(dataset, 'uvmet10', timeidx=0)
                u = wrf.to_np(uv[0, :])
                v = wrf.to_np(uv[1, :])
                lats, lons = wrf.latlon_coords(uv)
                lats = wrf.to_np(lats)
                lons = wrf.to_np(lons)
                level_label = '10 m'
            except Exception:
                if 'U10' in dataset.variables:
                    u = _read_wrf_2d(dataset, 'U10')
                    v = _read_wrf_2d(dataset, 'V10')
                    lats = _read_wrf_2d(dataset, 'XLAT')
                    lons = _read_wrf_2d(dataset, 'XLONG')
                    level_label = '10 m'
                else:
                    u_st = _read_wrf_2d(dataset, 'U')
                    v_st = _read_wrf_2d(dataset, 'V')
                    u = 0.5 * (u_st[:, :-1] + u_st[:, 1:])
                    v = 0.5 * (v_st[:-1, :] + v_st[1:, :])
                    lats = _read_wrf_2d(dataset, 'XLAT')
                    lons = _read_wrf_2d(dataset, 'XLONG')
                    level_label = 'lowest mass level'
        finally:
            dataset.close()
    else:
        ds = xr.open_dataset(path, engine='netcdf4').squeeze()
        try:
            if 'U10' in ds:
                u = _get_array(ds, 'U10')
                v = _get_array(ds, 'V10')
                level_label = '10 m'
            else:
                level_idx, level_label = _lowest_wind_level(ds)
                u = _destagger_np(_get_array(ds, 'U'), axis=2)[level_idx]
                v = _destagger_np(_get_array(ds, 'V'), axis=1)[level_idx]
            lons = np.asarray(_get_array(ds, 'XLONG'), dtype=float)
            lats = np.asarray(_get_array(ds, 'XLAT'), dtype=float)
            if lons.ndim == 3:
                lons = lons[0]
                lats = lats[0]
        finally:
            ds.close()

    ws = np.sqrt(u ** 2 + v ** 2)
    return lons, lats, u, v, ws, level_label


def load_cartesian_surface_wind(
    nc_path: str | os.PathLike[str],
    lon0: float = LON0,
    lat0: float = LAT0,
) -> tuple[xr.Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """读取 cartesian NC 最底层有效风场。"""
    path = resolve_input_path(nc_path)
    ds = xr.open_dataset(path, engine='netcdf4').squeeze()
    if not _is_cartesian(ds):
        raise ValueError(f"子图 B 需要 1h-rolling_cartesian NC，当前文件不符合: {path}")

    level_idx, level_label = _lowest_wind_level(ds)
    u = ds['U'].isel(z=level_idx).values
    v = ds['V'].isel(z=level_idx).values
    x_1d = np.asarray(ds['x_rel'].values, dtype=float)
    y_1d = np.asarray(ds['y_rel'].values, dtype=float)
    x_grid, y_grid = np.meshgrid(x_1d, y_1d)
    proj = Proj(proj='aeqd', lat_0=lat0, lon_0=lon0, datum='WGS84', units='m')
    lon_grid, lat_grid = proj(x_grid, y_grid, inverse=True)
    ws = np.sqrt(u ** 2 + v ** 2)
    return ds, lon_grid, lat_grid, x_grid, y_grid, u, v, ws, level_label


def crop_by_lonlat(
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    ws: np.ndarray,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> tuple:
    """按经纬度框裁剪；请求范围超出数据范围时裁到数据边界。"""
    if lon.ndim != 2:
        raise ValueError("Expected 2D lon/lat arrays.")

    lon_min_req, lon_max_req = lon_range
    lat_min_req, lat_max_req = lat_range

    lon_min = max(lon_min_req, float(np.nanmin(lon)))
    lon_max = min(lon_max_req, float(np.nanmax(lon)))
    lat_min = max(lat_min_req, float(np.nanmin(lat)))
    lat_max = min(lat_max_req, float(np.nanmax(lat)))

    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError(
            f"Requested box {lon_range} x {lat_range} has no overlap with data extent "
            f"({float(np.nanmin(lon)):.4f}-{float(np.nanmax(lon)):.4f}°E, "
            f"{float(np.nanmin(lat)):.4f}-{float(np.nanmax(lat)):.4f}°N)."
        )

    in_box = (
        (lon >= lon_min) & (lon <= lon_max)
        & (lat >= lat_min) & (lat <= lat_max)
    )
    rows = np.any(in_box, axis=1)
    cols = np.any(in_box, axis=0)
    if not np.any(rows) or not np.any(cols):
        raise ValueError("No grid points inside clipped lon/lat box.")

    yi = np.where(rows)[0]
    xi = np.where(cols)[0]
    bounds = (lon_min, lon_max, lat_min, lat_max)
    out = (
        lon[np.ix_(yi, xi)], lat[np.ix_(yi, xi)],
        u[np.ix_(yi, xi)], v[np.ix_(yi, xi)], ws[np.ix_(yi, xi)],
        bounds,
    )
    if x is not None and y is not None:
        out = out + (x[np.ix_(yi, xi)], y[np.ix_(yi, xi)])
    return out


def crop_by_xy(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    ws: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple:
    """按笛卡尔 (x,y) 米坐标框裁剪；请求范围超出数据时裁到数据边界。"""
    if x.ndim != 2:
        raise ValueError("Expected 2D x/y arrays.")

    x_min_req, x_max_req = x_range
    y_min_req, y_max_req = y_range

    x_min = max(x_min_req, float(np.nanmin(x)))
    x_max = min(x_max_req, float(np.nanmax(x)))
    y_min = max(y_min_req, float(np.nanmin(y)))
    y_max = min(y_max_req, float(np.nanmax(y)))

    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            f"Requested box x{x_range} y{y_range} has no overlap with data extent "
            f"(x {float(np.nanmin(x)):.0f}–{float(np.nanmax(x)):.0f} m, "
            f"y {float(np.nanmin(y)):.0f}–{float(np.nanmax(y)):.0f} m)."
        )

    in_box = (
        (x >= x_min) & (x <= x_max)
        & (y >= y_min) & (y <= y_max)
    )
    rows = np.any(in_box, axis=1)
    cols = np.any(in_box, axis=0)
    if not np.any(rows) or not np.any(cols):
        raise ValueError("No grid points inside clipped x/y box.")

    yi = np.where(rows)[0]
    xi = np.where(cols)[0]
    bounds = (x_min, x_max, y_min, y_max)
    return (
        lon[np.ix_(yi, xi)], lat[np.ix_(yi, xi)],
        u[np.ix_(yi, xi)], v[np.ix_(yi, xi)], ws[np.ix_(yi, xi)],
        bounds,
        x[np.ix_(yi, xi)], y[np.ix_(yi, xi)],
    )


def boundary_wind_from_dataset(ds: xr.Dataset) -> dict[str, dict[str, float]]:
    wind_data: dict[str, dict[str, float]] = {}
    level_idx, _ = _lowest_wind_level(ds)
    u_lvl = ds['U'].isel(z=level_idx)
    v_lvl = ds['V'].isel(z=level_idx)
    slices = {
        'west': (u_lvl.isel(x_rel=0), v_lvl.isel(x_rel=0)),
        'east': (u_lvl.isel(x_rel=-1), v_lvl.isel(x_rel=-1)),
        'south': (u_lvl.isel(y_rel=0), v_lvl.isel(y_rel=0)),
        'north': (u_lvl.isel(y_rel=-1), v_lvl.isel(y_rel=-1)),
    }
    for patch, (u_da, v_da) in slices.items():
        wind_data[patch] = {
            'U': float(u_da.mean(skipna=True)),
            'V': float(v_da.mean(skipna=True)),
        }
    return wind_data


def build_diagnostic_text(
    ds: xr.Dataset,
    boundary_data_dir: str | None = None,
    *,
    z_max: float = DEFAULT_Z_MAX,
) -> str:
    warnings: list[str] = []
    compare_data = None
    flux_data = None
    compare_label = f"0–{z_max:g} m 带平均"
    if boundary_data_dir:
        wind_data = read_boundary_wind_data(boundary_data_dir, level="surface")
        compare_data = read_boundary_wind_data(
            boundary_data_dir, level="band", z_max=z_max,
        )
        flux_data = read_boundary_wind_data(boundary_data_dir, level="full")
        for patch in PATCHES:
            if patch not in wind_data:
                warnings.append(
                    f"⚠️  警告: 找不到或无法读取 {patch} 边界的 0/U 数据。"
                )
        source_note = (
            f"（诊断来源: OpenFOAM boundaryData，近地面层 z≈10 m；"
            f"一致性对照: {compare_label}；质量平衡: 全柱）"
        )
    else:
        wind_data = boundary_wind_from_dataset(ds)
        source_note = "（诊断来源: NetCDF 四边最底层平均风）"

    text = format_boundary_flux_report(
        wind_data,
        warnings=warnings,
        compare_wind_data=compare_data,
        compare_label=compare_label,
        flux_wind_data=flux_data,
        flux_label="全柱平均",
    )
    return text.replace(
        "Boundary Flux Detector)",
        f"Boundary Flux Detector)\n   {source_note}",
        1,
    )


def _region_title(label: str, bounds: tuple[float, float, float, float], level_label: str) -> str:
    lon_min, lon_max, lat_min, lat_max = bounds
    return (
        f"Region {label} ({lon_min:.2f}–{lon_max:.2f}°E, "
        f"{lat_min:.2f}–{lat_max:.2f}°N), {level_label}"
    )


def _region_b_title(bounds: tuple[float, float, float, float], level_label: str) -> str:
    x_min, x_max, y_min, y_max = bounds
    return (
        f"Region B (x: {x_min:.0f}–{x_max:.0f} m, "
        f"y: {y_min:.0f}–{y_max:.0f} m), {level_label}"
    )


def _panel_vrange(ws: np.ndarray) -> tuple[float, float]:
    vmax = float(np.nanmax(ws))
    return 0.0, min(max(vmax, 0.5), 12.0)


def _format_meter_ticks(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))


def _format_lonlat_ticks(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    lon_min, lon_max, lat_min, lat_max = bounds
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    lon_dec = 3 if lon_span < 0.2 else 2 if lon_span < 2.0 else 1
    lat_dec = 3 if lat_span < 0.2 else 2 if lat_span < 2.0 else 1
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{lon_dec}f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{lat_dec}f'))


def _ensure_coastline_geojson() -> Path:
    if _COASTLINE_CACHE.is_file():
        return _COASTLINE_CACHE
    _COASTLINE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_NE_COASTLINE_URL, _COASTLINE_CACHE)
    return _COASTLINE_CACHE


def _plot_coastlines_fallback(
    ax: plt.Axes,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> None:
    with _ensure_coastline_geojson().open(encoding='utf-8') as f:
        geojson = json.load(f)
    pad = 0.5
    for feat in geojson.get('features', []):
        geom = feat.get('geometry', {})
        lines: list[list] = []
        if geom.get('type') == 'LineString':
            lines = [geom['coordinates']]
        elif geom.get('type') == 'MultiLineString':
            lines = geom['coordinates']
        for line in lines:
            coords = np.asarray(line, dtype=float)
            if coords.size == 0:
                continue
            lon_c, lat_c = coords[:, 0], coords[:, 1]
            if (
                lon_c.max() < lon_min - pad or lon_c.min() > lon_max + pad
                or lat_c.max() < lat_min - pad or lat_c.min() > lat_max + pad
            ):
                continue
            ax.plot(lon_c, lat_c, color='#1a1a1a', linewidth=0.8, zorder=5)


def _add_coastlines(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    *,
    use_cartopy: bool,
) -> None:
    if use_cartopy:
        ax.coastlines(resolution='10m', linewidth=0.8, color='#1a1a1a', zorder=5)
        return
    lon_min, lon_max, lat_min, lat_max = bounds
    _plot_coastlines_fallback(ax, lon_min, lon_max, lat_min, lat_max)


def plot_wind_panel(
    ax: plt.Axes,
    px: np.ndarray,
    py: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    ws: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    *,
    add_coastlines: bool = False,
    geo_transform=None,
    lonlat_bounds: tuple[float, float, float, float] | None = None,
    use_xy_coords: bool = False,
) -> plt.cm.ScalarMappable:
    use_cartopy = geo_transform is not None

    if use_cartopy and lonlat_bounds is not None:
        lon_min, lon_max, lat_min, lat_max = lonlat_bounds
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    cf = ax.contourf(
        px, py, ws, levels=20, cmap=cmap, vmin=vmin, vmax=vmax, extend='max',
        transform=geo_transform, zorder=1,
    )

    ny, nx = u.shape
    step = max(1, min(ny, nx) // (8 if use_xy_coords else 18))
    px_q = px[::step, ::step]
    py_q = py[::step, ::step]
    u_q = u[::step, ::step]
    v_q = v[::step, ::step]

    if use_xy_coords:
        dx = float(np.nanmean(np.abs(np.diff(px, axis=1))))
        dy = float(np.nanmean(np.abs(np.diff(py, axis=0))))
        cell = max(dx, dy, 1.0)
        arrow_len = cell * 3.0
        mag = np.maximum(np.hypot(u_q, v_q), 1e-6)
        u_q = u_q / mag * arrow_len
        v_q = v_q / mag * arrow_len
        quiver_kw = dict(
            color='white',
            edgecolors='black',
            linewidths=0.45,
            width=0.0045,
            angles='xy',
            scale_units='xy',
            scale=1,
            headwidth=4.5,
            headlength=5.5,
            headaxislength=4.5,
        )
    else:
        ws_max = float(np.nanmax(ws))
        quiver_kw = dict(
            color='white',
            alpha=0.85,
            scale=max(ws_max * 15.0, 1.0),
            width=0.003,
        )

    ax.quiver(
        px_q, py_q, u_q, v_q,
        transform=geo_transform,
        zorder=4,
        alpha=0.9 if use_xy_coords else quiver_kw.pop('alpha', 0.85),
        **quiver_kw,
    )

    if add_coastlines and lonlat_bounds is not None:
        _add_coastlines(ax, lonlat_bounds, use_cartopy=use_cartopy)

    ax.set_title(title)

    if use_xy_coords:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_xlim(float(np.nanmin(px)), float(np.nanmax(px)))
        ax.set_ylim(float(np.nanmin(py)), float(np.nanmax(py)))
        ax.set_aspect('equal')
        _format_meter_ticks(ax)
    elif use_cartopy and lonlat_bounds is not None:
        lon_min, lon_max, lat_min, lat_max = lonlat_bounds
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        xticks = np.linspace(lon_min, lon_max, 5)
        yticks = np.linspace(lat_min, lat_max, 5)
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    else:
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        if lonlat_bounds is not None:
            lon_min, lon_max, lat_min, lat_max = lonlat_bounds
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            _format_lonlat_ticks(ax, lonlat_bounds)

    return cf


def plot_region_b_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    ws: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    x_range: tuple[float, float] = REGION_B_XY,
    y_range: tuple[float, float] = REGION_B_XY,
) -> plt.cm.ScalarMappable:
    """子图 B：笛卡尔 (x,y) 米坐标，pcolormesh + quiver。"""
    pcm = ax.pcolormesh(
        x, y, ws, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax, zorder=1,
    )

    ny, nx = ws.shape
    step = max(1, min(ny, nx) // 10)
    ax.quiver(
        x[::step, ::step], y[::step, ::step],
        u[::step, ::step], v[::step, ::step],
        color='black',
        pivot='mid',
        scale=25,
        scale_units='width',
        width=0.0035,
        headwidth=4,
        headlength=5,
        headaxislength=4,
        zorder=5,
    )

    ax.set_title(title)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_aspect('equal', adjustable='box')
    _format_meter_ticks(ax)
    return pcm


def _display_friendly_report(text: str) -> str:
    replacements = {
        '🌬️ ': '',
        '📊 ': '',
        '🎯 ': '',
        '🛠️ ': '',
        '🟢 ': '[IN]  ',
        '🔴 ': '[OUT] ',
        '⚠️  ': '[!] ',
        '⚠️ ': '[!] ',
        '⚖️  ': '',
        '⚖️ ': '',
        '❗ ': '[!!] ',
        '✅ ': '[OK] ',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def plot_diagnostic_text(ax: plt.Axes, diag_text: str) -> None:
    ax.axis('off')
    ax.text(
        0.02, 0.98, _display_friendly_report(diag_text),
        transform=ax.transAxes,
        va='top', ha='left',
        fontsize=9.0,
        family='WenQuanYi Micro Hei',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f8f8', edgecolor='#cccccc'),
    )


def default_output_path(paths: RunPaths) -> Path:
    """
    results/wrf_openfoam/domain_surface_inlet_diag/
      domain_surface_inlet_diag_<YYYYMMDD_HHMM>.png
    """
    return RESULTS_DIAG_DIR / f"domain_surface_inlet_diag_{paths.label}.png"


def _print_resolved_paths(paths: RunPaths) -> None:
    print(f"时刻: {paths.label} ({paths.wrf_iso})")
    print(f"  cartesian NC: {paths.cartesian_nc}")
    print(f"  wrfout_d01:   {paths.wrfout_d01}")
    if paths.boundary_data:
        print(f"  boundaryData: {paths.boundary_data}")
    else:
        print("  boundaryData: （未找到，左下诊断将使用 NetCDF 四边风）")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='WRF 域面风场 (A: wrfout_d01 / B: cartesian) + 边界出入口诊断图。',
        epilog=(
            '示例: python util/plot_wrf_domain_surface_inlet_diag.py 20250903_1200\n'
            '      python util/plot_wrf_domain_surface_inlet_diag.py 09-03_12:00\n'
            '      python util/plot_wrf_domain_surface_inlet_diag.py path/to/cartesian.nc\n'
            '默认输出: results/wrf_openfoam/domain_surface_inlet_diag/'
            'domain_surface_inlet_diag_<YYYYMMDD_HHMM>.png'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'target',
        help='时刻/算例标识或 cartesian NC 路径（如 20250903_1200、算例目录名、.nc 文件）',
    )
    parser.add_argument(
        '--cartesian-nc', default=None,
        help='显式指定 cartesian NC 路径（覆盖 target 推断）',
    )
    parser.add_argument(
        '--wrfout-a', default=None,
        help='子图 A 用 wrfout_d01 路径（默认由时刻推断）',
    )
    parser.add_argument(
        '--boundary-data', default=None,
        help='OpenFOAM constant/boundaryData 目录（默认由算例目录推断）',
    )
    parser.add_argument(
        '--experiments-dir', default=None,
        help=f'OpenFOAM 算例根目录（默认: {DEFAULT_EXPERIMENTS_DIR}）',
    )
    parser.add_argument(
        '--case-suffix', default=DEFAULT_CASE_SUFFIX,
        help=f'算例目录名后缀（默认: {DEFAULT_CASE_SUFFIX}）',
    )
    parser.add_argument('--lon0', type=float, default=LON0)
    parser.add_argument('--lat0', type=float, default=LAT0)
    parser.add_argument('--region-a-lon', type=float, nargs=2, default=REGION_A_LON,
                        metavar=('LON_MIN', 'LON_MAX'))
    parser.add_argument('--region-a-lat', type=float, nargs=2, default=REGION_A_LAT,
                        metavar=('LAT_MIN', 'LAT_MAX'))
    parser.add_argument('--region-b-x', type=float, nargs=2, default=REGION_B_XY,
                        metavar=('X_MIN', 'X_MAX'),
                        help='子图 B x 米坐标范围（默认 -5000 5000）')
    parser.add_argument('--region-b-y', type=float, nargs=2, default=REGION_B_XY,
                        metavar=('Y_MIN', 'Y_MAX'),
                        help='子图 B y 米坐标范围（默认 -5000 5000）')
    parser.add_argument(
        '-o', '--output', default=None,
        help=(
            '输出 PNG 路径（默认: results/wrf_openfoam/domain_surface_inlet_diag/'
            'domain_surface_inlet_diag_<YYYYMMDD_HHMM>.png）'
        ),
    )
    parser.add_argument(
        '--z-max', type=float, default=DEFAULT_Z_MAX,
        help=f'对照/质量平衡高度上限 (m)，默认 {DEFAULT_Z_MAX:g}',
    )
    args = parser.parse_args()

    experiments_dir = (
        resolve_input_path(args.experiments_dir)
        if args.experiments_dir
        else (_REPO_ROOT / DEFAULT_EXPERIMENTS_DIR)
    )

    try:
        paths = resolve_run_paths(
            args.target,
            experiments_dir=experiments_dir,
            case_suffix=args.case_suffix,
            cartesian_nc=args.cartesian_nc,
            wrfout_a=args.wrfout_a,
            boundary_data=args.boundary_data,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)

    _print_resolved_paths(paths)

    configure_matplotlib_style()

    boundary_data_str = str(paths.boundary_data) if paths.boundary_data else None

    lon_a_full, lat_a_full, u_a_full, v_a_full, ws_a_full, level_a = load_wrfout_surface_wind(paths.wrfout_d01)
    ds_b, lon_b_full, lat_b_full, x_b_full, y_b_full, u_b_full, v_b_full, ws_b_full, level_b = load_cartesian_surface_wind(
        paths.cartesian_nc, lon0=args.lon0, lat0=args.lat0,
    )
    try:
        lon_a, lat_a, u_a, v_a, ws_a, bounds_a = crop_by_lonlat(
            lon_a_full, lat_a_full, u_a_full, v_a_full, ws_a_full,
            tuple(args.region_a_lon), tuple(args.region_a_lat),
        )
        lon_b, lat_b, u_b, v_b, ws_b, _, x_b, y_b = crop_by_xy(
            x_b_full, y_b_full, u_b_full, v_b_full, ws_b_full,
            lon_b_full, lat_b_full,
            tuple(args.region_b_x), tuple(args.region_b_y),
        )

        diag_text = build_diagnostic_text(
            ds_b, boundary_data_str, z_max=args.z_max,
        )

        vmin_a, vmax_a = _panel_vrange(ws_a)
        vmin_b, vmax_b = _panel_vrange(ws_b)
        cmap = 'viridis'

        fig = plt.figure(figsize=(14, 11))
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.15, 0.85], hspace=0.32, wspace=0.22)
        fig.suptitle(
            _format_utc_timestamp(paths.wrf_iso),
            fontsize=13,
            fontweight='bold',
            y=0.98,
        )

        geo_transform = ccrs.PlateCarree() if HAS_CARTOPY else None
        if HAS_CARTOPY:
            ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        else:
            ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_diag = fig.add_subplot(gs[1, 0])

        cf_a = plot_wind_panel(
            ax_a, lon_a, lat_a, u_a, v_a, ws_a,
            title=_region_title('A', bounds_a, level_a),
            vmin=vmin_a, vmax=vmax_a, cmap=cmap,
            add_coastlines=True,
            geo_transform=geo_transform,
            lonlat_bounds=bounds_a,
        )
        cf_b = plot_region_b_panel(
            ax_b, x_b, y_b, u_b, v_b, ws_b,
            title=_region_b_title(
                (args.region_b_x[0], args.region_b_x[1],
                 args.region_b_y[0], args.region_b_y[1]),
                level_b,
            ),
            vmin=vmin_b, vmax=vmax_b, cmap=cmap,
            x_range=tuple(args.region_b_x),
            y_range=tuple(args.region_b_y),
        )
        plot_diagnostic_text(ax_diag, diag_text)

        cbar_a = fig.colorbar(
            cf_a, ax=ax_a, orientation='horizontal',
            fraction=0.05, pad=0.08, aspect=30, extend='max',
        )
        cbar_a.set_label('Wind Speed (m/s) — A', fontweight='bold')
        cbar_b = fig.colorbar(
            cf_b, ax=ax_b, orientation='horizontal',
            fraction=0.05, pad=0.08, aspect=30, extend='max',
        )
        cbar_b.set_label('Wind Speed (m/s) — B', fontweight='bold')

        if args.output:
            out_path = Path(args.output)
            if not out_path.is_absolute():
                out_path = (_REPO_ROOT / out_path).resolve()
        else:
            out_path = default_output_path(paths)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Figure saved to: {out_path}")
        if not HAS_CARTOPY:
            print("提示: 未安装 cartopy，海岸线使用 Natural Earth 110m GeoJSON 回退绘制。")
    finally:
        ds_b.close()


if __name__ == '__main__':
    main()
