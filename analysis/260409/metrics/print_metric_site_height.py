#!/usr/bin/env python3
"""Per-station wind-speed metrics at Fig-4 comparison heights (120 m / 500 m)."""

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

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv"
DEFAULT_OUT = _SCRIPT_DIR / "metric_site_height_120_500.csv"

TARGET_HEIGHTS = {
    120.0: "120 m (nearest gate)",
    500.0: "500 m (nearest gate)",
}


def pearson_r(model: np.ndarray, obs: np.ndarray) -> float:
    valid = ~(np.isnan(model) | np.isnan(obs))
    if valid.sum() < 2:
        return float("nan")
    m = model[valid]
    o = obs[valid]
    if np.std(m) == 0 or np.std(o) == 0:
        return float("nan")
    return float(np.corrcoef(m, o)[0, 1])


def pick_nearest_height(heights: np.ndarray, h_req: float) -> float:
    idx = int(np.argmin(np.abs(heights - float(h_req))))
    return float(heights[idx])


def compute_site_height_metrics(
    df: pd.DataFrame,
    *,
    target_heights: dict[float, str] | None = None,
) -> pd.DataFrame:
    if target_heights is None:
        target_heights = TARGET_HEIGHTS

    sub = df[df["qc_ok"] & df["ws_obs"].notna()].copy()
    records: list[dict] = []

    for site in sorted(sub["obtid"].unique()):
        site_df = sub[sub["obtid"] == site]
        heights = pd.to_numeric(site_df["Height"], errors="coerce").dropna().unique()
        if len(heights) == 0:
            continue
        heights = np.asarray(heights, dtype=float)

        for h_req, h_label in target_heights.items():
            h_used = pick_nearest_height(heights, h_req)
            g = site_df[np.isclose(site_df["Height"].astype(float), h_used)]
            if len(g) < 2:
                continue

            obs = g["ws_obs"].values
            wrf = g["ws_wrf"].values
            cfd = g["ws_cfd"].values

            for model_name, sim in [("WRF", wrf), ("CFD", cfd)]:
                records.append(
                    {
                        "obtid": site,
                        "height_requested_m": h_req,
                        "height_used_m": h_used,
                        "height_label": h_label,
                        "model": model_name,
                        "n": len(g),
                        "R": round(pearson_r(sim, obs), 3),
                        "MBE": round(pm.mean_bias_error(sim, obs), 3),
                        "RMSE": round(pm.rmse(sim, obs), 3),
                        "IOA": round(pm.index_of_agreement(sim, obs), 3),
                    }
                )

            records.append(
                {
                    "obtid": site,
                    "height_requested_m": h_req,
                    "height_used_m": h_used,
                    "height_label": h_label,
                    "model": "CFD_vs_WRF",
                    "n": len(g),
                    "R": float("nan"),
                    "MBE": float("nan"),
                    "RMSE": float("nan"),
                    "IOA": float("nan"),
                    "SS": round(pm.skill_score(cfd, obs, wrf), 3),
                }
            )

    out = pd.DataFrame(records)
    if out.empty:
        return out

    # Reorder columns for readability.
    cols = [
        "obtid",
        "height_requested_m",
        "height_used_m",
        "height_label",
        "model",
        "n",
        "R",
        "MBE",
        "RMSE",
        "IOA",
        "SS",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute per-station metrics at 120 m and 500 m for Fig. 4 narrative."
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = pm.quality_control(pm.load_and_preprocess(args.csv))
    result = compute_site_height_metrics(df)
    if result.empty:
        print("[ERROR] No metrics computed.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"[CSV] Wrote {len(result)} rows to {args.out}")
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
