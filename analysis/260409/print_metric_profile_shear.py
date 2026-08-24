#!/usr/bin/env python3
"""Layer-mean wind speed and mean dU/dz for Fig. 6 profile window (1 Sep 19:00–00:00 BJT)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import print_metric as pm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv"
DEFAULT_OUT = _SCRIPT_DIR / "metric_profile_shear_20250901_1900_0000_bjt.csv"

# BJT 19:00–00:00 on 1 Sep 2025 = UTC 11:00–16:00 inclusive (hourly).
PROFILE_START_UTC = "2025-09-01 11:00:00"
PROFILE_END_UTC = "2025-09-01 16:00:00"

LOW_Z_MIN = 52.0
LOW_Z_MAX = 300.0


def mean_dudz(heights: np.ndarray, ws: np.ndarray) -> float:
    """Mean |dU/dz| over consecutive valid profile levels (m/s per m)."""
    order = np.argsort(heights)
    h = heights[order]
    u = ws[order]
    valid = ~(np.isnan(h) | np.isnan(u))
    h, u = h[valid], u[valid]
    if len(h) < 2:
        return float("nan")
    dz = np.diff(h)
    du = np.diff(u)
    mask = dz > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(du[mask] / dz[mask])))


def compute_profile_shear(
    df: pd.DataFrame,
    *,
    start_utc: str,
    end_utc: str,
    z_min: float = LOW_Z_MIN,
    z_max: float = LOW_Z_MAX,
) -> pd.DataFrame:
    sub = df[df["qc_ok"] & df["ws_obs"].notna()].copy()
    t0 = pd.Timestamp(start_utc)
    t1 = pd.Timestamp(end_utc)
    sub = sub[(sub["datetime"] >= t0) & (sub["datetime"] <= t1)]

    records: list[dict] = []
    for site in sorted(sub["obtid"].unique()):
        for dt in sorted(sub["datetime"].unique()):
            g = sub[(sub["obtid"] == site) & (sub["datetime"] == dt)]
            low = g[(g["Height"] >= z_min) & (g["Height"] <= z_max)]
            if len(low) < 2:
                continue

            for src, col in [("obs", "ws_obs"), ("wrf", "ws_wrf"), ("cfd", "ws_cfd")]:
                h = low["Height"].to_numpy(dtype=float)
                ws = low[col].to_numpy(dtype=float)
                records.append(
                    {
                        "obtid": site,
                        "datetime_utc": dt,
                        "source": src,
                        "layer_z_min_m": z_min,
                        "layer_z_max_m": z_max,
                        "n_levels": len(low),
                        "mean_ws_m_s": round(float(np.nanmean(ws)), 3),
                        "mean_abs_dudz_s-1": round(mean_dudz(h, ws), 5),
                    }
                )

    out = pd.DataFrame(records)
    if out.empty:
        return out

    # Site-level aggregates across the six hourly profiles.
    agg_rows: list[dict] = []
    for site in sorted(out["obtid"].unique()):
        for src in ["obs", "wrf", "cfd"]:
            g = out[(out["obtid"] == site) & (out["source"] == src)]
            if g.empty:
                continue
            agg_rows.append(
                {
                    "obtid": site,
                    "datetime_utc": "AGGREGATE",
                    "source": src,
                    "layer_z_min_m": z_min,
                    "layer_z_max_m": z_max,
                    "n_levels": int(g["n_levels"].sum()),
                    "mean_ws_m_s": round(float(g["mean_ws_m_s"].mean()), 3),
                    "mean_abs_dudz_s-1": round(float(g["mean_abs_dudz_s-1"].mean()), 5),
                }
            )

    return pd.concat([out, pd.DataFrame(agg_rows)], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile-layer shear stats for Fig. 6 window.")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--start-utc", default=PROFILE_START_UTC)
    ap.add_argument("--end-utc", default=PROFILE_END_UTC)
    args = ap.parse_args()

    df = pm.quality_control(pm.load_and_preprocess(args.csv))
    result = compute_profile_shear(df, start_utc=args.start_utc, end_utc=args.end_utc)
    if result.empty:
        print("[ERROR] No profile shear stats computed.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"[CSV] Wrote {len(result)} rows to {args.out}")
    agg = result[result["datetime_utc"] == "AGGREGATE"]
    print("\nSite aggregates (52–300 m, six hourly profiles):")
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
