#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 WRF auxhist2 笛卡尔 NetCDF 提取 LiDAR 站点风场廓线，写入 260707 数据布局。

基于 util/extract_wrf_lidar_xarray.py，适配 20250906–13 UTC 00/06/12/18。
数据源按日期拆分：
  - W_myExp03/auxhist2  → 2025-09-06 及更早（战役内仅 9 月 6 日）
  - W_myExp05/auxhist2  → 2025-09-07 及之后

用法:
  python analysis/260707/extract_wrf_lidar.py
  python analysis/260707/extract_wrf_lidar.py --out data/260707/raw/wrf/WRF_lidar_simulation_1h-rolling.csv
"""

from __future__ import annotations

import argparse
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

DEFAULT_EXP03_DIR = cfg.REPO_ROOT / "W_myExp03" / "auxhist2"
DEFAULT_EXP05_DIR = cfg.REPO_ROOT / "W_myExp05" / "auxhist2"
DEFAULT_OUT_CSV = cfg.DATA_DIR / "raw" / "wrf" / "WRF_lidar_simulation_1h-rolling.csv"
STATION_JSON = cfg.REPO_ROOT / "util" / "lidar_station_info.json"
SPLIT_DATE = pd.Timestamp("2025-09-06")

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

WRF_VARS = ("U", "V", "W", "WS", "TKE_PBL")
_FNAME_DT_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}).*?(\d{2}).*?(\d{2})")


def resolve_existing_nc_path(path: Path) -> Path | None:
    """返回存在的 NetCDF 路径；兼容 ':' 与 '%3A' 时间戳文件名。"""
    if path.is_file():
        return path

    name = path.name
    candidates: list[Path] = []
    if ":" in name:
        candidates.append(path.with_name(name.replace(":", "%3A")))
        candidates.append(path.with_name(name.replace(":", "%3a")))
    if "%3A" in name or "%3a" in name:
        candidates.append(path.with_name(name.replace("%3A", ":").replace("%3a", ":")))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def nc_basename(dt: pd.Timestamp) -> str:
    return f"auxhist2_d03_{dt.strftime('%Y-%m-%d_%H:%M:%S')}_1h-rolling_cartesian.nc"


def wrf_auxhist_dir(
    dt: pd.Timestamp,
    *,
    exp03_dir: Path,
    exp05_dir: Path,
    split_date: pd.Timestamp,
) -> Path:
    if dt.normalize() <= split_date.normalize():
        return exp03_dir
    return exp05_dir


def locate_nc_file(
    dt: pd.Timestamp,
    *,
    exp03_dir: Path,
    exp05_dir: Path,
    split_date: pd.Timestamp,
) -> Path | None:
    aux_dir = wrf_auxhist_dir(dt, exp03_dir=exp03_dir, exp05_dir=exp05_dir, split_date=split_date)
    return resolve_existing_nc_path(aux_dir / nc_basename(dt))


def parse_datetime_from_fname(fname: str, fallback: pd.Timestamp) -> pd.Timestamp:
    m = _FNAME_DT_RE.search(fname)
    if not m:
        return fallback
    return pd.Timestamp(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}")


def compute_turbulence(target_z: np.ndarray, tke_pbl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_wrf = np.maximum(tke_pbl, 1e-6)
    cmu = 0.09
    kappa = 0.41
    mixing_length = np.clip(kappa * target_z, 10.0, 100.0)
    eps_wrf = (cmu**0.75) * (k_wrf**1.5) / mixing_length
    return k_wrf, np.maximum(eps_wrf, 1e-8)


def extract_nc_file(nc_path: Path, station_info: dict, dt_hint: pd.Timestamp | None = None) -> pd.DataFrame:
    fname = nc_path.name
    dt = dt_hint or parse_datetime_from_fname(fname, pd.Timestamp("2025-01-01"))

    with xr.open_dataset(nc_path, mask_and_scale=False) as ds:
        x_coords = ds["x_rel"].values.squeeze().astype(float)
        y_coords = ds["y_rel"].values.squeeze().astype(float)
        if x_coords.ndim != 1 or y_coords.ndim != 1:
            raise ValueError(
                f"x_rel/y_rel must be 1-D, got x_rel={x_coords.shape}, y_rel={y_coords.shape}"
            )

        z_arr = ds["z"].values.squeeze().astype(float)
        wrf_vars = {}
        for vn in WRF_VARS:
            arr = ds[vn].values.squeeze().astype(float)
            if arr.ndim != 3:
                raise ValueError(f"{vn} must be 3-D after squeeze, got shape={arr.shape}")
            wrf_vars[vn] = arr

    records: list[dict] = []
    for _, row in LIDAR_SITES.iterrows():
        obtid = row["obtid"]
        site_x = float(row["x_rel"])
        site_y = float(row["y_rel"])
        raw_levels = np.array(station_info[obtid]["levels"], dtype=float)
        target_z = raw_levels[raw_levels < 2000]

        i = int(np.argmin(np.abs(x_coords - site_x)))
        j = int(np.argmin(np.abs(y_coords - site_y)))

        col_data = {vn: wrf_vars[vn][:, j, i] for vn in WRF_VARS}
        h_col = z_arr.copy()
        if h_col.ndim != 1:
            raise ValueError(f"z must be 1-D, got shape={h_col.shape}")

        sort_idx = np.argsort(h_col)
        h_sorted = h_col[sort_idx]
        res: dict[str, np.ndarray] = {}
        for vn in ("U", "V", "W", "WS", "TKE_PBL"):
            v_sorted = np.asarray(col_data[vn], dtype=float)[sort_idx]
            f = interp1d(
                h_sorted,
                v_sorted,
                kind="linear",
                bounds_error=False,
                fill_value=(v_sorted[0], v_sorted[-1]),
            )
            res[vn] = f(target_z)

        k_wrf, eps_wrf = compute_turbulence(target_z, res["TKE_PBL"])
        for i_idx, z in enumerate(target_z):
            records.append(
                {
                    "datetime": dt,
                    "obtid": row["obtid"],
                    "lon": row["lon"],
                    "lat": row["lat"],
                    "x_rel": row["x_rel"],
                    "y_rel": row["y_rel"],
                    "z_probe": float(z),
                    "U_wrf": float(res["U"][i_idx]),
                    "V_wrf": float(res["V"][i_idx]),
                    "W_wrf": float(res["W"][i_idx]),
                    "WS_wrf": float(res["WS"][i_idx]),
                    "k_wrf": float(k_wrf[i_idx]),
                    "eps_wrf": float(eps_wrf[i_idx]),
                }
            )

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values(["datetime", "obtid", "z_probe"]).reset_index(drop=True)


def iter_campaign_nc_files(
    *,
    exp03_dir: Path,
    exp05_dir: Path,
    split_date: pd.Timestamp,
    datetimes: pd.DatetimeIndex | None = None,
) -> list[tuple[pd.Timestamp, Path]]:
    slots = datetimes if datetimes is not None else cfg.METRIC_DATETIMES
    found: list[tuple[pd.Timestamp, Path]] = []
    for dt in slots:
        nc_path = locate_nc_file(dt, exp03_dir=exp03_dir, exp05_dir=exp05_dir, split_date=split_date)
        if nc_path is not None:
            found.append((pd.Timestamp(dt), nc_path))
    return found


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--exp03-dir",
        type=Path,
        default=DEFAULT_EXP03_DIR,
        help=f"WRF auxhist2 for dates <= split-date (default: {DEFAULT_EXP03_DIR}).",
    )
    p.add_argument(
        "--exp05-dir",
        type=Path,
        default=DEFAULT_EXP05_DIR,
        help=f"WRF auxhist2 for dates > split-date (default: {DEFAULT_EXP05_DIR}).",
    )
    p.add_argument(
        "--split-date",
        type=str,
        default=SPLIT_DATE.strftime("%Y-%m-%d"),
        help="Last UTC calendar day served by --exp03-dir (default: 2025-09-06).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUT_CSV}).",
    )
    p.add_argument(
        "--datetime",
        action="append",
        default=None,
        metavar="YYYY-MM-DD HH:MM:SS",
        help="Extract only the given UTC datetime(s); default: all campaign synoptic slots.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    split_date = pd.Timestamp(args.split_date)

    with STATION_JSON.open(encoding="utf-8") as f:
        import json

        station_info = json.load(f)

    if args.datetime:
        target_slots = pd.DatetimeIndex([pd.Timestamp(s) for s in args.datetime])
    else:
        target_slots = cfg.METRIC_DATETIMES

    jobs = iter_campaign_nc_files(
        exp03_dir=args.exp03_dir,
        exp05_dir=args.exp05_dir,
        split_date=split_date,
        datetimes=target_slots,
    )
    missing = len(target_slots) - len(jobs)
    print(f"[Config] Target slots: {len(target_slots)}  found NC files: {len(jobs)}")
    if missing:
        print(f"[WARN] {missing} NetCDF file(s) missing across exp03/exp05 auxhist2")

    if not jobs:
        print("No WRF files to extract.", file=sys.stderr)
        return 1

    all_frames: list[pd.DataFrame] = []
    for dt, nc_path in jobs:
        src = "myExp03" if dt.normalize() <= split_date.normalize() else "myExp05"
        print(f"  WRF [lidar] {dt} ({src}) <- {nc_path.name} ...", end=" ", flush=True)
        try:
            df = extract_nc_file(nc_path, station_info, dt_hint=dt)
            all_frames.append(df)
            print(f"OK  rows={len(df)}")
        except Exception as exc:
            print(f"FAILED ({exc})")
            return 1

    out_df = pd.concat(all_frames, ignore_index=True)
    out_df = out_df.sort_values(["datetime", "obtid", "z_probe"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}  shape={out_df.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
