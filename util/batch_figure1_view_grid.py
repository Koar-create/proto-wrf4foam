#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a 3×3×3 grid of Figure 1 PNGs varying:
  - half_width_m  (horizontal clip extent)
  - camera_distance_factor
  - analysis_z_max_m (with zmax_m fixed high so effective z_top ≈ analysis_z_max_m)

Outputs under results/microhazard/20250903_1400/ and writes figure1_grid_index.json there.

Usage (from repo root):
  python util/batch_figure1_view_grid.py
  python util/batch_figure1_view_grid.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Three levels per axis (tunable)
HALF_WIDTH_M = (200.0, 400.0, 600.0)
CAMERA_DISTANCE_FACTOR = (1.0, 1.5, 2.0)
ANALYSIS_Z_MAX_M = (200.0, 280.0, 360.0)
# Keep above analysis caps so z_top = min(zmax_m, analysis_z_max_m) == analysis_z_max_m
ZMAX_M_FIXED = 400.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_render_mod():
    path = _repo_root() / "util" / "render_3d_microhazard_pyvista.py"
    spec = importlib.util.spec_from_file_location("render_3d_microhazard_pyvista", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _png_name(half_w: float, cdf: float, az: float) -> str:
    cdf_s = f"{cdf:.1f}".replace(".", "p")
    return f"figure1_grid_hw{int(round(half_w)):04d}_cdf{cdf_s}_az{int(round(az)):04d}.png"


def _parse_args_local() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3×3×3 Figure 1 view parameter sweep.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/microhazard/20250903_1400).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned runs only.")
    return p.parse_args()


def main() -> None:
    args = _parse_args_local()
    root = _repo_root()
    out_dir = args.out_dir if args.out_dir is not None else (root / "results" / "microhazard" / "20250903_1400")
    out_dir = out_dir if out_dir.is_absolute() else (root / out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # render_3d_microhazard_pyvista._parse_args() reads sys.argv — isolate so our --dry-run etc. do not leak
    _argv = sys.argv[:]
    sys.argv = [_argv[0]]
    try:
        mod = _load_render_mod()
        base_args = mod._parse_args()
    finally:
        sys.argv = _argv
    case_dir = base_args.case_dir if base_args.case_dir.is_absolute() else (root / base_args.case_dir)
    buildings = (
        base_args.buildings_stl if base_args.buildings_stl.is_absolute() else (root / base_args.buildings_stl)
    )
    csv_100 = case_dir / "postProcessing" / "100m.csv"
    if not csv_100.is_file():
        print(f"ERROR: missing {csv_100}", file=sys.stderr)
        sys.exit(1)

    x, y, w = mod._read_100m_windspeed(csv_100)
    x0, y0, thr, n_cand = mod._cluster_hotspot_xy(
        x,
        y,
        w,
        urban_core_m=float(base_args.urban_core_m),
        p_pct=float(base_args.p_percentile),
        dbscan_eps=float(base_args.dbscan_eps_m),
        dbscan_min_samples=int(base_args.dbscan_min_samples),
    )

    w_w, h_h = [int(s.strip()) for s in str(base_args.window_size).split(",") if s.strip()]
    iso_pct = mod._parse_iso_percentiles(str(base_args.iso_percentiles))

    entries: List[Dict[str, Any]] = []
    n_ok, n_err = 0, 0

    for hw in HALF_WIDTH_M:
        for cdf in CAMERA_DISTANCE_FACTOR:
            for az in ANALYSIS_Z_MAX_M:
                name = _png_name(hw, cdf, az)
                out_png = out_dir / name
                z_top = min(float(ZMAX_M_FIXED), float(az))
                rec: Dict[str, Any] = {
                    "half_width_m": hw,
                    "camera_distance_factor": cdf,
                    "analysis_z_max_m": az,
                    "zmax_m": ZMAX_M_FIXED,
                    "effective_z_top_m": z_top,
                    "file": str(out_png.relative_to(root)).replace("\\", "/"),
                    "status": "pending",
                }
                if args.dry_run:
                    rec["status"] = "dry_run"
                    entries.append(rec)
                    print(f"[dry-run] {name}")
                    continue
                try:
                    mod._render_pyvista(
                        case_dir=case_dir,
                        buildings_stl=buildings,
                        out_png=out_png,
                        x0=x0,
                        y0=y0,
                        half_w=float(hw),
                        zmax=float(ZMAX_M_FIXED),
                        slice_z=float(base_args.slice_z_m),
                        window_size=(w_w, h_h),
                        cmap=str(base_args.cmap),
                        clim_low_percentile=float(base_args.clim_low_percentile),
                        clim_percentile=float(base_args.clim_percentile),
                        analysis_z_max_m=float(az),
                        iso_percentiles=iso_pct,
                        iso_opacity=float(base_args.iso_opacity),
                        seed_box_m=float(base_args.seed_box_m),
                        seed_z_max_m=float(base_args.seed_z_max_m),
                        seed_nx=int(base_args.seed_nx),
                        seed_ny=int(base_args.seed_ny),
                        seed_nz=int(base_args.seed_nz),
                        use_edl=not bool(base_args.no_edl),
                        building_wind_sampling=bool(base_args.building_wind),
                        tube_radius_m=float(base_args.tube_radius_m),
                        show_streamlines=not bool(base_args.no_streamlines),
                        camera_distance_factor=float(cdf),
                    )
                    rec["status"] = "ok"
                    n_ok += 1
                    print(f"OK {len(entries)+1}/27: {name}")
                except Exception as e:
                    rec["status"] = "error"
                    rec["error"] = f"{type(e).__name__}: {e}"
                    n_err += 1
                    print(f"ERR {len(entries)+1}/27: {name} -> {rec['error']}", file=sys.stderr)
                entries.append(rec)

    index_path = out_dir / "figure1_grid_index.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_relative": True,
        "axes": {
            "half_width_m": list(HALF_WIDTH_M),
            "camera_distance_factor": list(CAMERA_DISTANCE_FACTOR),
            "analysis_z_max_m": list(ANALYSIS_Z_MAX_M),
            "zmax_m_fixed": ZMAX_M_FIXED,
        },
        "hotspot": {"x0_m": x0, "y0_m": y0, "p_threshold_m_s": thr, "n_candidates": n_cand},
        "count": len(entries),
        "ok": n_ok,
        "errors": n_err,
        "images": entries,
    }
    if not args.dry_run:
        index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {index_path.relative_to(root)}")
    else:
        print(f"Would write index with {len(entries)} entries to {index_path}")


if __name__ == "__main__":
    main()
