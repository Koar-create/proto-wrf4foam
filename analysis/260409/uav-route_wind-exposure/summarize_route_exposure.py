#!/usr/bin/env python3
"""Regime medians, Table 5, and the H1 / KF3 verdict for section 3.3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
OUT = _REPO / "results" / "uav_route_exposure_hourly"

METRICS = [
    "ws_wrf_mean",
    "ws_cfd_mean",
    "lambda_of_means",
    "frac_lambda_gt1",
    "frac_fast",
    "frac_decel",
    "p90_p10_cfd",
    "p90_p10_wrf",
    "shear_cfd_mean",
    "shear_cfd_p95",
    "shear_wrf_mean",
    "shear_wrf_p95",
]


def _median_range(series: pd.Series) -> dict[str, float]:
    v = pd.to_numeric(series, errors="coerce").dropna()
    if v.empty:
        return {"median": np.nan, "p10": np.nan, "p90": np.nan, "n": 0}
    return {
        "median": float(v.median()),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
        "n": int(v.size),
    }


def main() -> int:
    hourly = pd.concat(
        [pd.read_csv(p) for p in sorted((OUT / "hourly").glob("*.csv"))],
        ignore_index=True,
    )
    shear_path = OUT / "shear_hourly.csv"
    if shear_path.is_file():
        shear = pd.read_csv(shear_path)
        hourly = hourly.merge(shear, on=["case", "route", "height_m"], how="left")
    else:
        for col in ("shear_cfd_mean", "shear_cfd_p95", "shear_wrf_mean", "shear_wrf_p95"):
            hourly[col] = np.nan

    rows = []
    for keys, g in hourly.groupby(["route", "height_m", "regime"], sort=True):
        route, height, regime = keys
        row = {"route": route, "height_m": int(height), "regime": regime, "n_hours": int(len(g))}
        for metric in METRICS:
            if metric not in g.columns:
                continue
            stats = _median_range(g[metric])
            row[f"{metric}_median"] = stats["median"]
            row[f"{metric}_p10"] = stats["p10"]
            row[f"{metric}_p90"] = stats["p90"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = OUT / "regime_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Table 5: medians only, wide on regime, same quantities as the plan.
    show = [
        "ws_wrf_mean_median",
        "ws_cfd_mean_median",
        "lambda_of_means_median",
        "frac_lambda_gt1_median",
        "frac_fast_median",
        "frac_decel_median",
        "p90_p10_cfd_median",
        "p90_p10_wrf_median",
        "shear_cfd_mean_median",
        "shear_cfd_p95_median",
        "shear_wrf_mean_median",
        "shear_wrf_p95_median",
    ]
    table = summary.pivot_table(
        index=["route", "height_m"],
        columns="regime",
        values=[c for c in show if c in summary.columns],
        aggfunc="first",
    )
    # Flatten to regular | typhoon columns in a stable order.
    flat_rows = []
    for (route, height), _ in summary.groupby(["route", "height_m"], sort=True):
        rec = {"route": route, "height_m": int(height)}
        for regime in ("regular", "typhoon"):
            sub = summary[
                (summary["route"] == route)
                & (summary["height_m"] == height)
                & (summary["regime"] == regime)
            ]
            if sub.empty:
                continue
            s = sub.iloc[0]
            rec[f"n_hours_{regime}"] = int(s["n_hours"])
            for col in show:
                if col in s:
                    rec[f"{col.replace('_median', '')}_{regime}"] = s[col]
        flat_rows.append(rec)
    table5 = pd.DataFrame(flat_rows)
    table5_path = OUT / "table5_route_exposure.csv"
    table5.to_csv(table5_path, index=False)

    lines = ["# Table 5. Along-route exposure, median across hours.", ""]
    lines.append(
        "Each entry is the median of the hourly route statistic. "
        "Parent WRF and WRF-OpenFOAM share the section 3.2 5 m grid. "
        "Ranges in regime_summary.csv are the 10th–90th percentiles of that hourly series, "
        "not an independent-sample confidence interval."
    )
    lines.append("")
    hdr = [
        "Route",
        "z (m)",
        "Regime",
        "Hours",
        "WRF mean (m/s)",
        "CFD mean (m/s)",
        "mean Λ",
        "Λ>1 fraction",
        "Fast-patch fraction",
        "Decel-patch fraction",
        "CFD P90/P10",
        "WRF P90/P10",
        "CFD shear mean (1/s)",
        "CFD shear P95 (1/s)",
        "WRF shear mean (1/s)",
        "WRF shear P95 (1/s)",
    ]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for _, r in table5.iterrows():
        for regime, label in (("regular", "Regular"), ("typhoon", "Typhoon")):
            def fmt(key, nd=3):
                val = r.get(f"{key}_{regime}", np.nan)
                if pd.isna(val):
                    return ""
                return f"{val:.{nd}f}"

            lines.append(
                "| "
                + " | ".join(
                    [
                        r["route"],
                        str(int(r["height_m"])),
                        label,
                        str(int(r[f"n_hours_{regime}"])),
                        fmt("ws_wrf_mean", 2),
                        fmt("ws_cfd_mean", 2),
                        fmt("lambda_of_means", 2),
                        fmt("frac_lambda_gt1", 2),
                        fmt("frac_fast", 2),
                        fmt("frac_decel", 2),
                        fmt("p90_p10_cfd", 2),
                        fmt("p90_p10_wrf", 2),
                        fmt("shear_cfd_mean", 4),
                        fmt("shear_cfd_p95", 4),
                        fmt("shear_wrf_mean", 4),
                        fmt("shear_wrf_p95", 4),
                    ]
                )
                + " |"
            )
    (OUT / "table5_route_exposure.md").write_text("\n".join(lines) + "\n")

    def med(route, height, regime, col):
        sub = summary[
            (summary["route"] == route)
            & (summary["height_m"] == height)
            & (summary["regime"] == regime)
        ]
        return float(sub.iloc[0][col]) if len(sub) else float("nan")

    # H1 at 30 m: typhoon Route 1 Λ>1 rises, Route 2 stays deficit-dominated, gap widens,
    # WRF means of the two routes stay close.
    r1_reg = med("Route1_open_river", 30, "regular", "frac_lambda_gt1_median")
    r1_ty = med("Route1_open_river", 30, "typhoon", "frac_lambda_gt1_median")
    r2_reg = med("Route2_urban_canyon", 30, "regular", "frac_lambda_gt1_median")
    r2_ty = med("Route2_urban_canyon", 30, "typhoon", "frac_lambda_gt1_median")
    gap_reg = r1_reg - r2_reg
    gap_ty = r1_ty - r2_ty
    wrf_r1 = med("Route1_open_river", 30, "typhoon", "ws_wrf_mean_median")
    wrf_r2 = med("Route2_urban_canyon", 30, "typhoon", "ws_wrf_mean_median")
    wrf_close = abs(wrf_r1 - wrf_r2) / max(wrf_r1, 1e-6) < 0.15
    route2_deficit = med("Route2_urban_canyon", 30, "typhoon", "lambda_of_means_median") < 1.0
    h1 = (r1_ty > r1_reg + 0.05) and route2_deficit and (gap_ty > gap_reg) and wrf_close
    if h1:
        branch = (
            "H1 holds. KF3 can say that under typhoon forcing the open-river route "
            "develops a larger share of segments faster than WRF, the canyon route stays "
            "in deficit, and the gap between the routes widens, while WRF itself gives the two routes similar means."
        )
    else:
        branch = (
            "H1 does not hold in full. KF3 should fall back to: the route contrast is set by urban morphology "
            "and does not simply grow under typhoon forcing. Do not pre-commit a third key finding beyond what these medians show."
        )
    verdict = [
        "# H1 verdict (30 m, hourly medians)",
        "",
        f"- Route 1 Λ>1 fraction: regular {r1_reg:.3f}, typhoon {r1_ty:.3f}",
        f"- Route 2 Λ>1 fraction: regular {r2_reg:.3f}, typhoon {r2_ty:.3f}",
        f"- Gap (Route1 − Route2): regular {gap_reg:.3f}, typhoon {gap_ty:.3f}",
        f"- Typhoon WRF mean speed: Route 1 {wrf_r1:.2f} m/s, Route 2 {wrf_r2:.2f} m/s (close={wrf_close})",
        f"- Route 2 typhoon mean Λ: {med('Route2_urban_canyon', 30, 'typhoon', 'lambda_of_means_median'):.3f} (deficit={route2_deficit})",
        f"- H1 = {h1}",
        "",
        branch,
        "",
        "Same quantities at 60 m and 120 m are in table5_route_exposure.csv. "
        "Λ>1 is a downscaling increment relative to WRF, not a flight-risk class.",
    ]
    (OUT / "h1_verdict.md").write_text("\n".join(verdict) + "\n")
    print(f"[CSV] {summary_path}", flush=True)
    print(f"[CSV] {table5_path}", flush=True)
    print("\n".join(verdict), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
