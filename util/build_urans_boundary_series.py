#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build time-varying constant/boundaryData for URANS pimpleFoam cases.

Uses existing convert + construct scripts without modifying them.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
WRF_EXP05_START = (2025, 9, 7)
PATCHES = ("west", "east", "south", "north")
FIELDS = ("U", "k", "epsilon")
# OpenFOAM time dirs: (sim_time_s, snapshot_index 0|1|2)
TIME_SCHEDULE = (
    (0, 0),
    (570, 0),
    (600, 1),
    (1170, 1),
    (1200, 2),
    (1800, 2),
)


def _wrf_run_dir(wrf_iso: str) -> Path:
    y, m, d = (int(wrf_iso[0:4]), int(wrf_iso[5:7]), int(wrf_iso[8:10]))
    exp = "W_myExp05" if (y, m, d) >= WRF_EXP05_START else "W_myExp03"
    return _REPO_ROOT / exp / "WRF" / "run"


def parse_anchor(anchor: str) -> datetime:
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})", anchor.strip())
    if not m:
        raise ValueError(f"anchor 期望 YYYYMMDD_HHMM，得到 {anchor!r}")
    y, mo, d, hh, mm = (int(x) for x in m.groups())
    return datetime(y, mo, d, hh, mm, 0)


def wrf_iso_from_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H:%M:%S")


def raw_auxhist_path(wrf_iso: str) -> Path:
    return _wrf_run_dir(wrf_iso) / f"auxhist2_d03_{wrf_iso}"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def parse_boundary_vectors(path: Path) -> np.ndarray:
    text = path.read_text()
    m = re.search(r"\n(\d+)\s*\n\(\s*\n", text)
    if not m:
        raise ValueError(f"无法解析向量场: {path}")
    n = int(m.group(1))
    block = text[m.end() :]
    vals = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        if line == ")":
            break
        line = line.strip("()")
        parts = line.split()
        if len(parts) == 3:
            vals.append([float(x) for x in parts])
    arr = np.array(vals, dtype=float)
    if arr.shape[0] != n:
        raise ValueError(f"{path}: 期望 {n} 点，解析到 {arr.shape[0]}")
    return arr


def parse_boundary_scalars(path: Path) -> np.ndarray:
    text = path.read_text()
    m = re.search(r"\n(\d+)\s*\n\(\s*\n", text)
    if not m:
        raise ValueError(f"无法解析标量场: {path}")
    n = int(m.group(1))
    vals = []
    for line in text[m.end() :].splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        if line == ")":
            break
        vals.append(float(line))
    arr = np.array(vals, dtype=float)
    if arr.shape[0] != n:
        raise ValueError(f"{path}: 期望 {n} 点，解析到 {arr.shape[0]}")
    return arr


def patch_normal(patch: str) -> np.ndarray:
    if patch == "west":
        return np.array([-1.0, 0.0, 0.0])
    if patch == "east":
        return np.array([1.0, 0.0, 0.0])
    if patch == "south":
        return np.array([0.0, -1.0, 0.0])
    if patch == "north":
        return np.array([0.0, 1.0, 0.0])
    raise KeyError(patch)


def diagnose_snapshot(bd_root: Path, snap_tag: str) -> dict:
    report: dict = {"tag": snap_tag, "patches": {}, "max_speed": 0.0}
    for patch in PATCHES:
        u_path = bd_root / patch / "0" / "U"
        u = parse_boundary_vectors(u_path)
        speed = np.linalg.norm(u, axis=1)
        n = patch_normal(patch)
        un = u @ n
        if patch in ("east", "south"):
            inflow_frac = float(np.mean(un < -0.01))
            expected = "inflow"
        else:
            inflow_frac = float(np.mean(un > 0.01))
            expected = "outflow_or_inletOutlet"
        report["patches"][patch] = {
            "expected_role": expected,
            "speed_min": float(speed.min()),
            "speed_max": float(speed.max()),
            "speed_mean": float(speed.mean()),
            "wrong_sign_fraction": float(1.0 - inflow_frac) if patch in ("east", "south") else float(np.mean(un > 0.05)),
        }
        report["max_speed"] = max(report["max_speed"], float(speed.max()))
    return report


def ensure_points_match(ref: Path, other: Path) -> None:
    for patch in PATCHES:
        a = ref / patch / "points"
        b = other / patch / "points"
        if not filecmp.cmp(a, b, shallow=False):
            raise RuntimeError(f"points 不一致: {patch} {a} vs {b}")


def build_snapshots(
    case_dir: Path,
    anchor: datetime,
    snapshot_offsets_min: list[int],
) -> list[Path]:
    wrf_dir = case_dir / "wrf_snapshots" / "tmp"
    wrf_dir.mkdir(parents=True, exist_ok=True)
    out_cartesian: list[Path] = []
    convert_py = _REPO_ROOT / "util" / "convert_wrf_coord_and_do_3D_interp.py"

    for off in snapshot_offsets_min:
        t = anchor + timedelta(minutes=off)
        wrf_iso = wrf_iso_from_dt(t)
        raw = raw_auxhist_path(wrf_iso)
        if not raw.is_file():
            raise FileNotFoundError(f"缺少 WRF auxhist2: {raw}")
        link_name = f"auxhist2_d03_{wrf_iso}_tmp.nc"
        link_path = wrf_dir / link_name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(raw.resolve())
        run_cmd([sys.executable, str(convert_py), str(link_path)])
        cart = case_dir / "wrf_snapshots" / link_name.replace("_tmp.nc", "_1h-rolling_cartesian.nc")
        if not cart.is_file():
            raise FileNotFoundError(f"插值产物未找到: {cart}")
        out_cartesian.append(cart)
        print(f"  snapshot +{off} min -> {cart.name}", flush=True)
    return out_cartesian


def construct_boundary(scratch: Path, nc_path: Path) -> Path:
    scratch.mkdir(parents=True, exist_ok=True)
    construct_py = _REPO_ROOT / "util" / "construct_OF_boundary_arrays.py"
    run_cmd(
        [sys.executable, str(construct_py), "--input-nc", str(nc_path)],
        cwd=scratch,
    )
    return scratch / "boundaryData"


def assemble_series(case_dir: Path, snap_bds: list[Path]) -> Path:
    bd_case = case_dir / "constant" / "boundaryData"
    if bd_case.exists():
        shutil.rmtree(bd_case)
    bd_case.mkdir(parents=True)

    ref = snap_bds[0]
    for patch in PATCHES:
        (bd_case / patch).mkdir(parents=True, exist_ok=True)
        shutil.copy2(ref / patch / "points", bd_case / patch / "points")

    for sim_t, snap_i in TIME_SCHEDULE:
        src = snap_bds[snap_i]
        for patch in PATCHES:
            dest_dir = bd_case / patch / str(sim_t)
            dest_dir.mkdir(parents=True, exist_ok=True)
            for fld in FIELDS:
                shutil.copy2(src / patch / "0" / fld, dest_dir / fld)
    return bd_case


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="URANS 边界时间序列构建")
    p.add_argument("case_dir", type=Path, help="OpenFOAM 算例根目录")
    p.add_argument("anchor", help="锚定时次 YYYYMMDD_HHMM")
    p.add_argument(
        "--snapshots",
        default="0,10,20",
        help="相对锚定的分钟偏移，逗号分隔（默认 0,10,20）",
    )
    p.add_argument("--ramp", type=float, default=30.0, help="过渡时长(s)，仅写入元数据")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    anchor = parse_anchor(args.anchor)
    offsets = [int(x.strip()) for x in args.snapshots.split(",") if x.strip()]
    if len(offsets) != 3:
        raise SystemExit("--snapshots 需要恰好 3 个偏移（对应 HH:00/10/20）")

    if args.dry_run:
        print("dry-run", case_dir, anchor, offsets)
        return 0

    print(f"=== build URANS boundaryData in {case_dir} ===", flush=True)
    nc_list = build_snapshots(case_dir, anchor, offsets)
    scratch_root = case_dir / "wrf_scratch"
    snap_bds: list[Path] = []
    for i, nc in enumerate(nc_list):
        scratch = scratch_root / f"snap{i:02d}"
        if scratch.exists():
            shutil.rmtree(scratch)
        bd = construct_boundary(scratch, nc)
        snap_bds.append(bd)

    for j in range(1, len(snap_bds)):
        ensure_points_match(snap_bds[0], snap_bds[j])

    assemble_series(case_dir, snap_bds)
    diag = [diagnose_snapshot(bd, f"snap{i}") for i, bd in enumerate(snap_bds)]
    meta = {
        "anchor": args.anchor,
        "anchor_wrf_iso": wrf_iso_from_dt(anchor),
        "snapshot_offsets_min": offsets,
        "ramp_s": args.ramp,
        "time_schedule": TIME_SCHEDULE,
        "snapshots": diag,
        "watchdog_Umax_suggest": 2.5 * max(d["max_speed"] for d in diag),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = case_dir / "urans_boundary_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {meta_path} and constant/boundaryData/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
