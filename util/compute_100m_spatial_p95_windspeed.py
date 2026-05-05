#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量读取 steady_experiments_finer_ABL/<case_id>/postProcessing/100m.csv，
排除目录名含 fvOpt_sensitivity_run 的敏感性算例；对每个切片用 U:0、U:1
计算水平风速后取空间 95% 分位数；从 case_id 前缀解析 UTC 时间，写出两列 CSV。

用法:
  python util/compute_100m_spatial_p95_windspeed.py
  python util/compute_100m_spatial_p95_windspeed.py --root steady_experiments_finer_ABL --out results/my_p95.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd

CASE_TIME_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})")
SENSITIVITY_MARK = "fvOpt_sensitivity_run"
DEFAULT_ROOT = "steady_experiments_finer_ABL"
DEFAULT_OUT = os.path.join(
    "results", "wrf_openfoam", "steady_ABL_100m_spatial_p95_windspeed.csv"
)
DEFAULT_CHUNKSIZE = 100_000


def parse_time_utc_from_case_dir(case_dir: str) -> Optional[pd.Timestamp]:
    base = os.path.basename(case_dir.rstrip(os.sep))
    m = CASE_TIME_RE.match(base)
    if not m:
        return None
    yr, mo, dy, hh, mm = m.groups()
    return pd.Timestamp(
        year=int(yr),
        month=int(mo),
        day=int(dy),
        hour=int(hh),
        minute=int(mm),
        tz="UTC",
    )


def spatial_p95_wind_speed(csv_path: str, chunksize: int) -> Optional[float]:
    if not os.path.isfile(csv_path):
        return None
    parts: list[np.ndarray] = []
    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            if "U:0" not in chunk.columns or "U:1" not in chunk.columns:
                return None
            u0 = chunk["U:0"].to_numpy(dtype=np.float64, copy=False)
            u1 = chunk["U:1"].to_numpy(dtype=np.float64, copy=False)
            ws = np.sqrt(u0 * u0 + u1 * u1)
            parts.append(ws)
        if not parts:
            return None
        all_ws = np.concatenate(parts)
        if all_ws.size == 0:
            return None
        return float(np.percentile(all_ws, 95))
    except Exception:
        return None


def iter_case_csvs(root: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Returns (included (case_dir, csv_path), excluded_case_dirs)."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return [], []

    included: list[tuple[str, str]] = []
    excluded_dirs: list[str] = []

    for name in sorted(os.listdir(root)):
        case_dir = os.path.join(root, name)
        if not os.path.isdir(case_dir):
            continue
        if SENSITIVITY_MARK in name:
            excluded_dirs.append(case_dir)
            continue
        csv_path = os.path.join(case_dir, "postProcessing", "100m.csv")
        if os.path.isfile(csv_path):
            included.append((case_dir, csv_path))

    return included, excluded_dirs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Per-case spatial 95th percentile of horizontal wind speed on 100m.csv slices; "
            "excludes fvOpt_sensitivity_run cases."
        )
    )
    ap.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help=f"Case root directory (default: {DEFAULT_ROOT})",
    )
    ap.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output CSV path (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
        help=f"pandas read_csv chunksize in rows (default: {DEFAULT_CHUNKSIZE})",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any case fails time parse or p95 compute",
    )
    args = ap.parse_args()

    included, excluded_dirs = iter_case_csvs(args.root)
    print(f"Root: {os.path.abspath(args.root)}")
    print(f"Included (has postProcessing/100m.csv, not sensitivity): {len(included)}")
    print(f"Excluded (dirname contains {SENSITIVITY_MARK}): {len(excluded_dirs)}")

    rows: list[dict] = []
    errors = 0

    for case_dir, csv_path in included:
        t_utc = parse_time_utc_from_case_dir(case_dir)
        if t_utc is None:
            print(f"WARN: skip (cannot parse time from dirname): {case_dir}", file=sys.stderr)
            errors += 1
            continue
        p95 = spatial_p95_wind_speed(csv_path, args.chunksize)
        if p95 is None:
            print(f"WARN: skip (cannot compute p95 / missing columns): {csv_path}", file=sys.stderr)
            errors += 1
            continue
        rows.append(
            {
                "time_utc": t_utc,
                "wind_speed_p95_m_s": p95,
            }
        )

    if args.strict and errors > 0:
        print(f"strict: {errors} case(s) failed", file=sys.stderr)
        return 1

    if not rows:
        print("No valid rows; output not written.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows).sort_values("time_utc").reset_index(drop=True)
    df["time_utc"] = df["time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    # Normalize +0000 -> +00:00 for readability
    df["time_utc"] = df["time_utc"].str.replace(r"\+0000$", "+00:00", regex=True)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
