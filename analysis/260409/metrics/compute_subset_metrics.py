"""Write subset wind-speed metrics for the 1–9 September study window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ANALYSIS = Path(__file__).resolve().parents[2]
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import batch
from metrics.layers import HEIGHT_BINS, LAYER_NAMES_EN, LAYER_SHORT, WS_MAX_CFD, WS_MAX_OBS
from metrics.stats import index_of_agreement, mean_bias_error, rmse, skill_score

SUMMARY_SPECS = [
    ("regular_weather", "All", "all_times"),
    ("regular_weather", "Daytime", "daytime"),
    ("regular_weather", "Nighttime", "nighttime"),
    ("strong_wind", "Strong-wind", "all_times"),
]


def paired_rows(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    out = pd.concat(frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True).dt.floor("h")
    out["ws_cfd"] = np.sqrt(out["u_cfd"] ** 2 + out["v_cfd"] ** 2)
    out["layer"] = pd.cut(out["Height"], bins=HEIGHT_BINS, labels=LAYER_NAMES_EN)
    ok = ((out["ws_obs"] <= WS_MAX_OBS) | out["ws_obs"].isna()) & (out["ws_cfd"] <= WS_MAX_CFD)
    return out.loc[ok & out["ws_obs"].notna() & out["ws_wrf"].notna() & out["ws_cfd"].notna()].copy()


def in_study_window(dt: pd.Series) -> pd.Series:
    return (dt >= batch.STUDY_START) & (dt <= batch.STUDY_END)


def in_strong_wind(dt: pd.Series) -> pd.Series:
    return (dt >= batch.STRONG_WIND_START) & (dt <= batch.STRONG_WIND_END)


def mask_lst_day_am(lst: pd.Series, day: int) -> pd.Series:
    return (lst.dt.month == 9) & (lst.dt.day == day) & (lst.dt.hour >= 7) & (lst.dt.hour <= 18)


def mask_lst_day_pm(lst: pd.Series, day: int) -> pd.Series:
    same_evening = (lst.dt.month == 9) & (lst.dt.day == day) & (lst.dt.hour >= 19)
    next_morning = (lst.dt.month == 9) & (lst.dt.day == day + 1) & (lst.dt.hour <= 6)
    return same_evening | next_morning


def metric_row(
    g: pd.DataFrame,
    *,
    sample_group: str,
    time_category: str,
    period: str,
    layer: str,
    sample_date: str | None = None,
) -> dict:
    o = g["ws_obs"].to_numpy(dtype=float)
    w = g["ws_wrf"].to_numpy(dtype=float)
    c = g["ws_cfd"].to_numpy(dtype=float)
    return {
        "sample_group": sample_group,
        "time_category": time_category,
        "sample_date": sample_date,
        "period": period,
        "layer": layer,
        "n": int(len(g)),
        "wrf_rmse": rmse(w, o),
        "cfd_rmse": rmse(c, o),
        "wrf_mbe": mean_bias_error(w, o),
        "cfd_mbe": mean_bias_error(c, o),
        "wrf_ioa": index_of_agreement(w, o),
        "cfd_ioa": index_of_agreement(c, o),
        "skill_score": skill_score(c, o, w),
    }


def summary_subset(paired: pd.DataFrame, time_category: str) -> pd.DataFrame:
    in_period = in_study_window(paired["datetime"])
    strong = in_strong_wind(paired["datetime"])
    if time_category == "Strong-wind":
        return paired.loc[in_period & strong]
    regular = paired.loc[in_period & ~strong]
    if time_category == "All":
        return regular
    hour = regular["datetime"].dt.hour
    if time_category == "Daytime":
        return regular.loc[hour.isin(batch.DAY_UTC_HOURS)]
    if time_category == "Nighttime":
        return regular.loc[hour.isin(batch.NIGHT_UTC_HOURS)]
    raise ValueError(f"Unsupported time category: {time_category}")


def build_rows(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for sample_group, time_category, period in SUMMARY_SPECS:
        part = summary_subset(paired, time_category)
        for full_name, short in LAYER_SHORT.items():
            g = part[part["layer"] == full_name]
            rows.append(
                metric_row(
                    g,
                    sample_group=sample_group,
                    time_category=time_category,
                    period=period,
                    layer=short,
                )
            )
            print(f"[summary] {time_category:11} {short:4} N={len(g):6d}")

    in_period = paired.loc[in_study_window(paired["datetime"])].copy()
    in_period["lst"] = in_period["datetime"] + pd.Timedelta(hours=8)
    lst = in_period["lst"]
    for day in batch.LST_DAYS:
        sample_date = f"2025-09-{day:02d}"
        for period, mask in (
            ("daytime", mask_lst_day_am(lst, day)),
            ("nighttime", mask_lst_day_pm(lst, day)),
        ):
            part = in_period.loc[mask]
            for full_name, short in LAYER_SHORT.items():
                g = part[part["layer"] == full_name]
                rows.append(
                    metric_row(
                        g,
                        sample_group="lst_day_period",
                        time_category="",
                        period=period,
                        layer=short,
                        sample_date=sample_date,
                    )
                )
                print(f"[daily]   {sample_date} {period:9} {short:4} N={len(g):6d}")
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write subset wind-speed metrics (regular / strong-wind / daily LST)."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        default=list(batch.MERGED_CSVS),
        help="Merged LiDAR–simulation CSV files to concatenate.",
    )
    parser.add_argument("--out", type=Path, default=batch.SUBSET_METRICS_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paired = paired_rows(args.csv)
    table = build_rows(paired)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  ({len(table)} rows)")


if __name__ == "__main__":
    main()
