"""Extend print_underestimation_failure_stats.py to the typhoon-affected window.

Quantifies, for the low-layer (52-300 m) station-mean hourly wind speed, how
often WRF underestimates the observed wind speed and how often the coupled
CFD run deepens that underestimation -- separately for the regular-weather
baseline and the typhoon-affected window defined in
analysis/260409/metrics/visualize_metric_sample_variants_metric_panels.py
(2025-09-07 03:00 to 2025-09-08 19:00 UTC, inclusive).

Data: data/260409/processed/merged_lidar_simulation_final.csv (1-5 Sep)
      data/260707/processed/merged_lidar_simulation_final.csv (6-9 Sep, incl. typhoon)
Usage: python analysis/260409/print_underestimation_failure_stats_typhoon.py
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_CSV = REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv"
LATER_CSV = REPO_ROOT / "data/260707/processed/merged_lidar_simulation_final.csv"

LOW_MIN_M, LOW_MAX_M = 52, 300
LST_OFFSET_HOURS = 8

# Same windows as analysis/260409/metrics/visualize_metric_sample_variants_metric_panels.py.
# Regular-weather is defined in docs/scs-wrf-of-manuscript/02-rough-writing.md (S2.1) as
# "all remaining hours" in the 1-9 Sep study window, i.e. the study window minus the
# typhoon-affected window below.
PERIOD_START = pd.Timestamp("2025-09-01 00:00:00", tz="UTC")
PERIOD_END = pd.Timestamp("2025-09-09 23:00:00", tz="UTC")
TYPHOON_START = pd.Timestamp("2025-09-07 03:00:00", tz="UTC")
TYPHOON_END = pd.Timestamp("2025-09-08 19:00:00", tz="UTC")


def _load_hourly_low_layer() -> pd.DataFrame:
    frames = []
    for path in (PRIMARY_CSV, LATER_CSV):
        d = pd.read_csv(path)
        d["datetime"] = pd.to_datetime(d["datetime"], utc=True)
        if "ws_cfd" not in d.columns:
            d["ws_cfd"] = (d["u_cfd"] ** 2 + d["v_cfd"] ** 2) ** 0.5
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["datetime", "obtid", "Height"], keep="first")

    low = df[(df["Height"] >= LOW_MIN_M) & (df["Height"] <= LOW_MAX_M)]
    hourly = low.groupby("datetime")[["ws_obs", "ws_wrf", "ws_cfd"]].mean()
    hourly["wrf_bias"] = hourly["ws_wrf"] - hourly["ws_obs"]
    hourly["cfd_bias"] = hourly["ws_cfd"] - hourly["ws_obs"]
    hourly["lst"] = hourly.index + pd.Timedelta(hours=LST_OFFSET_HOURS)
    return hourly


def report(name: str, sub: pd.DataFrame) -> None:
    n_total = len(sub)
    under = sub[sub["wrf_bias"] < 0]
    n_under = len(under)
    worsened = under[under["cfd_bias"] < under["wrf_bias"]]
    n_worsened = len(worsened)
    deepen = (worsened["cfd_bias"] - worsened["wrf_bias"]).mean() if n_worsened else float("nan")

    over = sub[sub["wrf_bias"] >= 0]
    n_over = len(over)
    over_improved = over[over["cfd_bias"].abs() < over["wrf_bias"].abs()]

    print(f"--- {name} ---")
    print(f"N_total={n_total}")
    print(
        f"WRF underestimates: N_under={n_under} ({n_under / n_total:.1%} of hours); "
        f"of those, CFD deepens the underestimation: N_worsened={n_worsened} "
        f"({(n_worsened / n_under if n_under else float('nan')):.1%} of N_under)"
    )
    print(f"  mean additional deepening among worsened hours (cfd_bias - wrf_bias): {deepen:+.3f} m/s")
    print(
        f"WRF overestimates or ties: N_over={n_over} ({n_over / n_total:.1%} of hours); "
        f"of those, CFD reduces |bias|: N_improved={len(over_improved)} "
        f"({(len(over_improved) / n_over if n_over else float('nan')):.1%} of N_over)"
    )
    print(f"  mean wrf_bias={sub['wrf_bias'].mean():+.3f} m/s  mean cfd_bias={sub['cfd_bias'].mean():+.3f} m/s")
    print()


def main() -> None:
    hourly = _load_hourly_low_layer()
    in_period = (hourly.index >= PERIOD_START) & (hourly.index <= PERIOD_END)
    in_typhoon = (hourly.index >= TYPHOON_START) & (hourly.index <= TYPHOON_END)
    reg = hourly[in_period & ~in_typhoon]
    typ = hourly[in_period & in_typhoon]

    report("Regular (all non-typhoon hours, 1-9 Sep, UTC)", reg)
    report("Typhoon-affected (07 03:00 - 08 19:00 UTC; N should be 41)", typ)

    print("All underestimated typhoon hours, sorted by wrf_bias (most negative first):")
    under_typ = typ[typ["wrf_bias"] < 0].sort_values("wrf_bias")
    cols = ["ws_obs", "ws_wrf", "ws_cfd", "wrf_bias", "cfd_bias", "lst"]
    print(under_typ[cols].to_string())


if __name__ == "__main__":
    main()
