#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 OpenFOAM steady case 提取 LiDAR 站点 CFD 垂直廓线，写入 260707 数据布局。

基于 util/extract_cfd_lidar_xarray.py，适配 20250906–13 UTC 00/06/12/18 共 32 个实验。

用法:
  # 单个 case
  python analysis/260707/extract_cfd_lidar.py \\
      steady_experiments_finer_ABL/20250909_0600_two_boundaries_as_outlet

  # 批量提取战役内全部可用 case（目录存在且含时间步）
  python analysis/260707/extract_cfd_lidar.py --all

  # 指定 case 根目录与输出目录
  python analysis/260707/extract_cfd_lidar.py --all \\
      --cases-root steady_experiments_finer_ABL \\
      --out-dir data/260707/raw/cfd/control
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d

import campaign_config as cfg

warnings.filterwarnings("ignore")

CASE_SUFFIX = "_two_boundaries_as_outlet"
DEFAULT_CASES_ROOT = cfg.REPO_ROOT / "steady_experiments_finer_ABL"
DEFAULT_OUT_DIR = cfg.DATA_DIR / "raw" / "cfd" / "control"
STATION_JSON = cfg.REPO_ROOT / "util" / "lidar_station_info.json"

LIDAR_SITES = pd.DataFrame(
    {
        "obtid": ["GAW103", "GAW104", "GAW105", "GAW111"],
        "lon": [113.331446, 113.326053, 113.316797, 113.322620],
        "lat": [23.110176, 23.116321, 23.116133, 23.113718],
        "x_rel": [975, 450, -506, 75],
        "y_rel": [-320, 350, 400, 30],
        "altitude_m": [7.2, 11.6, 6, 30.9],
        "altitude_m_cfd": [1.6, 6, 0.4, 25.3],
    }
)

_CASE_DIR_RE = re.compile(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})")
_VECTOR_FIELD_RE = re.compile(
    r"internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*;",
    re.DOTALL,
)
_UNIFORM_VECTOR_RE = re.compile(r"internalField\s+uniform\s+\(([^)]+)\)")
_SCALAR_FIELD_RE = re.compile(
    r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*;",
    re.DOTALL,
)
_UNIFORM_SCALAR_RE = re.compile(r"internalField\s+uniform\s+([-\d.eE+]+)")
_VEC_TOKEN_RE = re.compile(
    r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)"
)


def case_dir_name(dt: pd.Timestamp) -> str:
    return f"{dt.strftime('%Y%m%d_%H00')}{CASE_SUFFIX}"


def output_csv_name(dt: pd.Timestamp) -> str:
    return cfg.cfd_filename(dt)


def parse_case_datetime(case_dir: Path) -> pd.Timestamp:
    m = _CASE_DIR_RE.match(case_dir.name)
    if not m:
        raise ValueError(f"Cannot parse datetime from case directory name: {case_dir.name}")
    yr, mo, dy, hh, mm = m.groups()
    return pd.Timestamp(f"{yr}-{mo}-{dy} {hh}:{mm}:00")


def find_last_timestep(case_dir: Path) -> tuple[str, str]:
    timesteps: list[int] = []
    for entry in case_dir.iterdir():
        if entry.is_dir():
            try:
                timesteps.append(int(entry.name))
            except ValueError:
                pass
    if not timesteps:
        raise FileNotFoundError(f"No numeric time directories under {case_dir}")
    last_ts = str(max(timesteps))
    status = "normal" if last_ts == "5000" else "early_converge"
    return last_ts, status


def _parse_vector_field(path: Path) -> np.ndarray:
    content = path.read_text(errors="replace")
    m_vec = _VECTOR_FIELD_RE.search(content)
    if m_vec:
        raw = m_vec.group(2).strip()
        vecs = _VEC_TOKEN_RE.findall(raw)
        return np.array([[float(a), float(b), float(c)] for a, b, c in vecs])

    m_uni = _UNIFORM_VECTOR_RE.search(content)
    if m_uni:
        return np.array(list(map(float, m_uni.group(1).split())))
    raise ValueError(f"Cannot parse vector internalField: {path}")


def read_cell_centres(case_dir: Path) -> np.ndarray:
    for candidate in (
        case_dir / "constant" / "cellCentres",
        case_dir / "0" / "C",
    ):
        if candidate.is_file():
            return _parse_vector_field(candidate)

    vtk_dir = case_dir / "VTK"
    if vtk_dir.is_dir():
        vtk_files = sorted(glob.glob(str(vtk_dir / "**" / "*.vtk"), recursive=True))
        if vtk_files:
            import vtk
            from vtk.util.numpy_support import vtk_to_numpy

            reader = vtk.vtkUnstructuredGridReader()
            reader.SetFileName(vtk_files[-1])
            reader.Update()
            grid = reader.GetOutput()
            centres_filter = vtk.vtkCellCenters()
            centres_filter.SetInputData(grid)
            centres_filter.Update()
            pts = centres_filter.GetOutput().GetPoints()
            return vtk_to_numpy(pts.GetData())

    raise FileNotFoundError(
        f"Cannot find cell centres for {case_dir}. "
        "Run: postProcess -func writeCellCentres -time 0"
    )


def read_scalar_field(case_dir: Path, last_ts: str, field_name: str, n_cells: int) -> np.ndarray:
    ffile = case_dir / last_ts / field_name
    if not ffile.is_file():
        return np.zeros(n_cells)

    content = ffile.read_text(errors="replace")
    m_scalar = _SCALAR_FIELD_RE.search(content)
    if m_scalar:
        raw = m_scalar.group(2).strip()
        return np.array([float(v) for v in raw.split()])

    m_uni = _UNIFORM_SCALAR_RE.search(content)
    if m_uni:
        return np.full(n_cells, float(m_uni.group(1)))
    return np.zeros(n_cells)


def _extract_site_profile(
    *,
    cell_coords: np.ndarray,
    u_arr: np.ndarray,
    k_arr: np.ndarray,
    eps_arr: np.ndarray,
    site_row: pd.Series,
    probe_heights: np.ndarray,
) -> list[dict[str, float | str]]:
    site_x = float(site_row["x_rel"])
    site_y = float(site_row["y_rel"])
    cx, cy, cz = cell_coords[:, 0], cell_coords[:, 1], cell_coords[:, 2]
    dist_h = np.sqrt((cx - site_x) ** 2 + (cy - site_y) ** 2)

    n_min = 30
    radius = 50.0
    while radius <= 2000.0:
        mask = dist_h <= radius
        if mask.sum() >= n_min:
            break
        radius *= 1.5
    mask = dist_h <= radius
    if mask.sum() < 3:
        idx_sorted = np.argsort(dist_h)[:n_min]
        mask = np.zeros(len(cx), dtype=bool)
        mask[idx_sorted] = True

    z_sel = cz[mask]
    u_sel = u_arr[mask, 0]
    v_sel = u_arr[mask, 1]
    w_sel = u_arr[mask, 2]
    k_sel = k_arr[mask]
    eps_sel = eps_arr[mask]
    dist_sel = dist_h[mask]
    weights = 1.0 / (dist_sel + 1e-6)
    weights /= weights.sum()

    z_min, z_max = cz.min(), cz.max()
    bin_edges = np.arange(z_min - 10, z_max + 30, 20)
    z_bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    num_u = np.zeros(len(z_bin_centres))
    den_u = np.zeros(len(z_bin_centres))
    num_v = np.zeros(len(z_bin_centres))
    den_v = np.zeros(len(z_bin_centres))
    num_w = np.zeros(len(z_bin_centres))
    den_w = np.zeros(len(z_bin_centres))
    num_k = np.zeros(len(z_bin_centres))
    den_k = np.zeros(len(z_bin_centres))
    num_eps = np.zeros(len(z_bin_centres))
    den_eps = np.zeros(len(z_bin_centres))

    for i_bin in range(len(z_bin_centres)):
        in_bin = (z_sel >= bin_edges[i_bin]) & (z_sel < bin_edges[i_bin + 1])
        if in_bin.sum() == 0:
            continue
        w_bin = weights[in_bin]
        num_u[i_bin] = (w_bin * u_sel[in_bin]).sum()
        den_u[i_bin] = w_bin.sum()
        num_v[i_bin] = (w_bin * v_sel[in_bin]).sum()
        den_v[i_bin] = w_bin.sum()
        num_w[i_bin] = (w_bin * w_sel[in_bin]).sum()
        den_w[i_bin] = w_bin.sum()
        num_k[i_bin] = (w_bin * k_sel[in_bin]).sum()
        den_k[i_bin] = w_bin.sum()
        num_eps[i_bin] = (w_bin * eps_sel[in_bin]).sum()
        den_eps[i_bin] = w_bin.sum()

    valid = (den_u > 0) & (den_v > 0) & (den_w > 0) & (den_k > 0) & (den_eps > 0)
    z_prof = z_bin_centres[valid]
    u_prof = num_u[valid] / den_u[valid]
    v_prof = num_v[valid] / den_v[valid]
    w_prof = num_w[valid] / den_w[valid]
    k_prof = num_k[valid] / den_k[valid]
    eps_prof = num_eps[valid] / den_eps[valid]

    if len(z_prof) < 2:
        order = np.argsort(z_sel)
        z_prof = z_sel[order]
        u_prof = u_sel[order]
        v_prof = v_sel[order]
        w_prof = w_sel[order]
        k_prof = k_sel[order]
        eps_prof = eps_sel[order]

    _, ui = np.unique(z_prof, return_index=True)
    z_prof = z_prof[ui]
    u_prof = u_prof[ui]
    v_prof = v_prof[ui]
    w_prof = w_prof[ui]
    k_prof = k_prof[ui]
    eps_prof = eps_prof[ui]

    f_u = interp1d(z_prof, u_prof, kind="linear", bounds_error=False, fill_value=(u_prof[0], u_prof[-1]))
    f_v = interp1d(z_prof, v_prof, kind="linear", bounds_error=False, fill_value=(v_prof[0], v_prof[-1]))
    f_w = interp1d(z_prof, w_prof, kind="linear", bounds_error=False, fill_value=(w_prof[0], w_prof[-1]))
    f_k = interp1d(z_prof, k_prof, kind="linear", bounds_error=False, fill_value=(k_prof[0], k_prof[-1]))
    f_eps = interp1d(
        z_prof, eps_prof, kind="linear", bounds_error=False, fill_value=(eps_prof[0], eps_prof[-1])
    )

    u_out = f_u(probe_heights)
    v_out = f_v(probe_heights)
    w_out = f_w(probe_heights)
    ws_out = np.sqrt(u_out ** 2 + v_out ** 2)
    k_out = f_k(probe_heights)
    eps_out = f_eps(probe_heights)

    records: list[dict[str, float | str]] = []
    for i_idx, z in enumerate(probe_heights):
        records.append(
            {
                "obtid": site_row["obtid"],
                "lon": site_row["lon"],
                "lat": site_row["lat"],
                "x_rel": site_row["x_rel"],
                "y_rel": site_row["y_rel"],
                "altitude_m_cfd": site_row["altitude_m_cfd"],
                "z_probe": float(z),
                "U_cfd": float(u_out[i_idx]),
                "V_cfd": float(v_out[i_idx]),
                "W_cfd": float(w_out[i_idx]),
                "WS_cfd": float(ws_out[i_idx]),
                "k_cfd": float(k_out[i_idx]),
                "eps_cfd": float(eps_out[i_idx]),
            }
        )
    return records


def extract_case(
    case_dir: Path,
    *,
    station_info: dict,
    dt: pd.Timestamp | None = None,
) -> pd.DataFrame:
    case_dir = case_dir.resolve()
    if not case_dir.is_dir():
        raise FileNotFoundError(case_dir)

    dt = dt or parse_case_datetime(case_dir)
    last_ts, status = find_last_timestep(case_dir)
    cell_coords = read_cell_centres(case_dir)
    u_arr = _parse_vector_field(case_dir / last_ts / "U")
    if u_arr.ndim == 1:
        u_arr = np.tile(u_arr, (len(cell_coords), 1))
    k_arr = read_scalar_field(case_dir, last_ts, "k", len(cell_coords))
    eps_arr = read_scalar_field(case_dir, last_ts, "epsilon", len(cell_coords))

    records: list[dict] = []
    dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    exp_name = case_dir.name
    for _, site_row in LIDAR_SITES.iterrows():
        obtid = site_row["obtid"]
        raw_levels = np.array(station_info[obtid]["levels"], dtype=float)
        probe_heights = raw_levels[raw_levels < 2000]
        site_records = _extract_site_profile(
            cell_coords=cell_coords,
            u_arr=u_arr,
            k_arr=k_arr,
            eps_arr=eps_arr,
            site_row=site_row,
            probe_heights=probe_heights,
        )
        for rec in site_records:
            rec.update(
                {
                    "datetime": dt_str,
                    "exp_name": exp_name,
                    "cfd_status": status,
                }
            )
            records.append(rec)

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values(["datetime", "obtid", "z_probe"]).reset_index(drop=True)


def save_case_csv(df: pd.DataFrame, out_path: Path, *, verify_xarray: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if verify_xarray and not df.empty:
        numeric_cols = ["z_probe", "U_cfd", "V_cfd", "W_cfd", "WS_cfd", "k_cfd", "eps_cfd"]
        ds = xr.Dataset.from_dataframe(df[numeric_cols])
        ds.close()
    df.to_csv(out_path, index=False)


def iter_campaign_cases(cases_root: Path) -> list[tuple[pd.Timestamp, Path]]:
    cases: list[tuple[pd.Timestamp, Path]] = []
    for dt in cfg.METRIC_DATETIMES:
        case_dir = cases_root / case_dir_name(dt)
        if case_dir.is_dir():
            cases.append((dt, case_dir))
    return cases


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "case_paths",
        nargs="*",
        help="One or more OpenFOAM case directories.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Extract all campaign cases found under --cases-root.",
    )
    p.add_argument(
        "--cases-root",
        type=Path,
        default=DEFAULT_CASES_ROOT,
        help=f"Root directory containing case folders (default: {DEFAULT_CASES_ROOT}).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for CFD_lidar_simulation_*.csv (default: {DEFAULT_OUT_DIR}).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cases whose output CSV already exists.",
    )
    p.add_argument(
        "--no-xarray-check",
        action="store_true",
        help="Skip xarray.Dataset validation before writing CSV.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.all and not args.case_paths:
        print("Error: provide case path(s) or use --all.", file=sys.stderr)
        return 2

    with STATION_JSON.open(encoding="utf-8") as f:
        station_info = json.load(f)

    if args.all:
        case_jobs = [(dt, case_dir) for dt, case_dir in iter_campaign_cases(args.cases_root)]
        missing = len(cfg.METRIC_DATETIMES) - len(case_jobs)
        print(f"[Config] Campaign slots: {len(cfg.METRIC_DATETIMES)}  found cases: {len(case_jobs)}")
        if missing:
            print(f"[WARN] {missing} campaign case directories missing under {args.cases_root}")
    else:
        case_jobs = []
        for raw in args.case_paths:
            case_dir = Path(raw)
            if not case_dir.is_absolute():
                case_dir = (Path.cwd() / case_dir).resolve()
            case_jobs.append((parse_case_datetime(case_dir), case_dir))

    if not case_jobs:
        print("No cases to extract.", file=sys.stderr)
        return 1

    saved = 0
    skipped = 0
    failed = 0
    for dt, case_dir in case_jobs:
        out_path = args.out_dir / output_csv_name(dt)
        if args.skip_existing and out_path.is_file():
            print(f"  SKIP existing: {out_path.name}")
            skipped += 1
            continue

        print(f"  CFD [lidar] {dt} : {case_dir.name} ...", end=" ", flush=True)
        try:
            df = extract_case(case_dir, station_info=station_info, dt=dt)
            save_case_csv(df, out_path, verify_xarray=not args.no_xarray_check)
            print(f"OK -> {out_path.name}  shape={df.shape}")
            saved += 1
        except Exception as exc:
            print(f"FAILED ({exc})")
            failed += 1

    print(f"\nDone: saved={saved}  skipped={skipped}  failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
