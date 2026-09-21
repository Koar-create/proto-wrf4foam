#!/usr/bin/env python3
"""Summarize along-track wind speed and shear for UAV route Figures 12–13."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN_DIR = REPO_ROOT / "results/uav_route_wind_shear_multi_z/20250901_1000"
DEFAULT_OUT = Path(__file__).resolve().parent / "uav_route_stats_20250901_1000.csv"

HEIGHTS = (30, 60, 120)


def summarize_route_csv(path: Path, route_name: str) -> list[dict]:
    df = pd.read_csv(path)
    rows: list[dict] = []
    for z in HEIGHTS:
        ws_w = pd.to_numeric(df[f"WS_wrf_{z}"], errors="coerce")
        ws_c = pd.to_numeric(df[f"WS_cfd_{z}"], errors="coerce")
        sh_w = pd.to_numeric(df[f"Vector_Shear_wrf_{z}"], errors="coerce")
        sh_c = pd.to_numeric(df[f"Vector_Shear_cfd_{z}"], errors="coerce")

        wrf_mean = float(ws_w.mean())
        cfd_mean = float(ws_c.mean())
        pct_reduction = (
            100.0 * (wrf_mean - cfd_mean) / wrf_mean if wrf_mean > 0 else float("nan")
        )

        rows.append(
            {
                "route": route_name,
                "height_m": z,
                "n_points": len(df),
                "WS_wrf_min": round(float(ws_w.min()), 3),
                "WS_wrf_mean": round(wrf_mean, 3),
                "WS_wrf_max": round(float(ws_w.max()), 3),
                "WS_cfd_min": round(float(ws_c.min()), 3),
                "WS_cfd_mean": round(cfd_mean, 3),
                "WS_cfd_max": round(float(ws_c.max()), 3),
                "WS_cfd_mean_reduction_pct": round(pct_reduction, 1),
                "Shear_wrf_mean": round(float(sh_w.mean()), 4),
                "Shear_wrf_max": round(float(sh_w.max()), 4),
                "Shear_cfd_mean": round(float(sh_c.mean()), 4),
                "Shear_cfd_max": round(float(sh_c.max()), 4),
                "WS_cfd_below_1ms_frac": round(float((ws_c < 1.0).mean()), 3),
                "WS_cfd_below_0p5ms_frac": round(float((ws_c < 0.5).mean()), 3),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize UAV route along-track statistics.")
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    files = {
        "Route1_open_river": args.in_dir / "route1_along_track_multi_z30-60-120.csv",
        "Route2_street_canyon": args.in_dir / "route2_along_track_multi_z30-60-120.csv",
    }

    all_rows: list[dict] = []
    for route, path in files.items():
        if not path.is_file():
            print(f"[ERROR] Missing {path}")
            return 1
        all_rows.extend(summarize_route_csv(path, route))

    out = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[CSV] Wrote {len(out)} rows to {args.out}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
