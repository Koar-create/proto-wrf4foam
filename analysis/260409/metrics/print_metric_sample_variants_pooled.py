"""
print_metric_sample_variants_pooled.py — Sample subset metric table (all heights merged, no low/mid/high stratification).

On the basis of print_metric_sample_variants.py, each period/subset only outputs one row of pooled summary (grid-level N),
The metric columns are the same as the hierarchical table in print_metric.main().

Can print multiple blocks (use `--sections` to trim). The subset definitions are the same as in sample_variants.
Can apply `--max-height 1000|2000` to limit the maximum height (in meters) for inclusion in the statistics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import print_metric as pm  # noqa: E402


UTC_TIME_GROUPS: dict[str, list[int]] = {
    "Daytime / Convective (LST 11-17 -> UTC 03-09)": list(range(3, 10)),
    "Nighttime / LLJ-active (LST 23-04 -> UTC 15-20)": list(range(15, 21)),
    "Transition (remaining UTC hours)": [
        h for h in range(24)
        if h not in set(range(3, 10)) | set(range(15, 21))
    ],
}

CONSERVATIVE_UTC_HOURS = list(range(0, 12))

MAX_HEIGHT_CHOICES = (1000, 2000)


def pool_label(max_height_m: int) -> str:
    return f"All heights <= {max_height_m} m"


def _base_subset(df: pd.DataFrame, max_height_m: int) -> pd.DataFrame:
    sub = df[df["qc_ok"] & df["ws_obs"].notna()].copy()
    return sub[sub["Height"] <= max_height_m]


def _attach_utc_hour(sub: pd.DataFrame) -> pd.DataFrame:
    out = sub.copy()
    out["utc_hour"] = out["datetime"].dt.hour
    return out


def _attach_lst(sub: pd.DataFrame) -> pd.DataFrame:
    out = sub.copy()
    out["lst"] = out["datetime"] + pd.Timedelta(hours=8)
    return out


def mask_lst_daytime(lst: pd.Series, day: int) -> pd.Series:
    """LST calendar day (day-th September), hours 07–18 (same as plot-fig4-lst daytime 12 h window)."""
    return (lst.dt.month == 9) & (lst.dt.day == day) & (lst.dt.hour >= 7) & (lst.dt.hour <= 18)


def mask_lst_nighttime(lst: pd.Series, day: int) -> pd.Series:
    """LST calendar day (day-th September), hours 19–23 and 00–06 (same as plot-fig4-lst nighttime 12 h window)."""
    same_evening = (lst.dt.month == 9) & (lst.dt.day == day) & (lst.dt.hour >= 19)
    next_morning = (lst.dt.month == 9) & (lst.dt.day == day + 1) & (lst.dt.hour <= 6)
    return same_evening | next_morning


def print_pooled_summary(title: str, grp_df: pd.DataFrame, *, max_height_m: int) -> None:
    """Merge all heights into one row summary (same metric columns as print_metric hierarchical table)."""
    label = pool_label(max_height_m)
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()
    header = (
        f"{'Subset':<28} {'N':>7}  {'WRF_MBE':>8} {'WRF_RMSE':>9} "
        f"{'WRF_IoA':>8}  {'CFD_MBE':>8} {'CFD_RMSE':>9} "
        f"{'CFD_IoA':>8}  {'SS':>7}"
    )
    print(header)
    print("-" * len(header))

    if len(grp_df) < 5:
        print(f"{label:<28} {len(grp_df):>7}  (insufficient N, skipped)")
        return

    o, w, c = grp_df["ws_obs"].values, grp_df["ws_wrf"].values, grp_df["ws_cfd"].values
    print(
        f"{label:<28} {len(grp_df):>7}  "
        f"{pm.mean_bias_error(w, o):>+8.3f} {pm.rmse(w, o):>9.3f} "
        f"{pm.index_of_agreement(w, o):>8.3f}  "
        f"{pm.mean_bias_error(c, o):>+8.3f} {pm.rmse(c, o):>9.3f} "
        f"{pm.index_of_agreement(c, o):>8.3f}  "
        f"{pm.skill_score(c, o, w):>+7.3f}"
    )


def section_baseline(sub: pd.DataFrame, max_height_m: int) -> None:
    print_pooled_summary(
        "Height-pooled summary - ALL times (baseline)", sub, max_height_m=max_height_m
    )


def section_utc_groups(sub: pd.DataFrame, max_height_m: int) -> None:
    sub_h = _attach_utc_hour(sub)
    for label, hours in UTC_TIME_GROUPS.items():
        g = sub_h[sub_h["utc_hour"].isin(hours)]
        print_pooled_summary(f"UTC subgroup - {label}", g, max_height_m=max_height_m)


def section_lst_periods(sub: pd.DataFrame, max_height_m: int) -> None:
    sub_l = _attach_lst(sub)
    lst = sub_l["lst"]
    for day in (1, 2, 3):
        for period, mask in (
            ("Daytime (LST 07-18)", mask_lst_daytime(lst, day)),
            ("Nighttime (LST 19 - next day 06)", mask_lst_nighttime(lst, day)),
        ):
            g = sub_l[mask]
            print_pooled_summary(
                f"LST-grouped (align fig4-lst) - 2025-09-{day:02d} {period}",
                g,
                max_height_m=max_height_m,
            )


def section_by_utc_calendar_day(sub: pd.DataFrame, max_height_m: int) -> None:
    sub = sub.copy()
    sub["utc_date"] = sub["datetime"].dt.strftime("%Y-%m-%d")
    for d in ("2025-09-01", "2025-09-02", "2025-09-03"):
        g = sub[sub["utc_date"] == d]
        print_pooled_summary(f"UTC calendar day - {d}", g, max_height_m=max_height_m)


def section_conservative(sub: pd.DataFrame, max_height_m: int) -> None:
    sub_h = _attach_utc_hour(sub)
    g = sub_h[sub_h["utc_hour"].isin(CONSERVATIVE_UTC_HOURS)]
    print_pooled_summary(
        "Conservative subset - UTC 00-11 (~ LST 08-19; advice strategy 2 example)",
        g,
        max_height_m=max_height_m,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print height-pooled metrics for multiple sample definitions "
        "(UTC regimes, LST day×daytime/nighttime, by UTC day, conservative hours)."
    )
    p.add_argument(
        "--sections",
        nargs="+",
        choices=[
            "baseline",
            "utc_groups",
            "lst_periods",
            "by_utc_day",
            "conservative",
            "all",
        ],
        default=["all"],
        help="Which blocks to print. Use 'all' for every block (default).",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Override CSV path (default: print_metric.DATA_PATH).",
    )
    p.add_argument(
        "--max-height",
        type=int,
        choices=MAX_HEIGHT_CHOICES,
        default=2000,
        metavar="M",
        help="Upper height limit in metres: 1000 (1 km) or 2000 (2 km). Default: 2000.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sections = set(args.sections)
    if "all" in sections:
        sections = {
            "baseline",
            "utc_groups",
            "lst_periods",
            "by_utc_day",
            "conservative",
        }

    max_h = args.max_height
    path = args.data if args.data is not None else pm.DATA_PATH
    df_raw = pm.load_and_preprocess(path)
    df = pm.quality_control(df_raw)
    sub = _base_subset(df, max_h)

    print(f"[Config] Height cap: <= {max_h} m  (rows above cap excluded from all sections)")

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.3f}".format)

    if "baseline" in sections:
        section_baseline(sub, max_h)
    if "utc_groups" in sections:
        section_utc_groups(sub, max_h)
    if "lst_periods" in sections:
        section_lst_periods(sub, max_h)
    if "by_utc_day" in sections:
        section_by_utc_calendar_day(sub, max_h)
    if "conservative" in sections:
        section_conservative(sub, max_h)


if __name__ == "__main__":
    main()
