"""
print_metric_sample_variants.py — 在 print_metric.py 基础上按「样本子集」复算分层汇总表。

设计来源：
  1) advice-to-adjust-metric-sample.md：按 UTC 对流/LLJ/过渡期、保守时段、按日历日拆分；
  2) plot-fig4-lst.py：LST = UTC+8，LST「日」× AM(07–18)/PM(19–次日06) 与廓线图分组一致。

默认打印多组区块（可用 --sections 裁剪）。指标与 print_metric.main() 中表格一致（行级 N 为格点数）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 与 print_metric.py 同目录，保证 `python path/to/本脚本.py` 可 import
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import print_metric as pm  # noqa: E402


# ─── advice-to-adjust-metric-sample.md 策略一（UTC 小时，datetime 为 UTC）────
UTC_TIME_GROUPS: dict[str, list[int]] = {
    "Daytime (LST 07-18)": [23, *range(0, 11)],
    "Nighttime (LST 19 - next day 06)": list(range(11, 23)),
}

# 策略二：保守子集（文中原例 UTC 0–11 ≈ LST 08–19）
CONSERVATIVE_UTC_HOURS = list(range(0, 12))


def _base_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["qc_ok"] & df["ws_obs"].notna()].copy()


def _attach_utc_hour(sub: pd.DataFrame) -> pd.DataFrame:
    out = sub.copy()
    out["utc_hour"] = out["datetime"].dt.hour
    return out


def _attach_lst(sub: pd.DataFrame) -> pd.DataFrame:
    out = sub.copy()
    out["lst"] = out["datetime"] + pd.Timedelta(hours=8)
    return out


def mask_lst_day_am(lst: pd.Series, day: int) -> pd.Series:
    """LST 日历日 9 月 day 日，小时 07–18（与 plot-fig4-lst AM 一致）。"""
    return (lst.dt.month == 9) & (lst.dt.day == day) & (lst.dt.hour >= 7) & (lst.dt.hour <= 18)


def mask_lst_day_pm(lst: pd.Series, day: int) -> pd.Series:
    """LST 9 月 day 日 19–23 与 9 月 day+1 日 00–06（与 plot-fig4-lst PM 一致）。"""
    same_evening = (lst.dt.month == 9) & (lst.dt.day == day) & (lst.dt.hour >= 19)
    next_morning = (lst.dt.month == 9) & (lst.dt.day == day + 1) & (lst.dt.hour <= 6)
    return same_evening | next_morning


def print_layer_summary(title: str, grp_df: pd.DataFrame, *, skip_high: bool = False) -> None:
    """与 print_metric.main() 相同列的分层汇总（全站点混合）。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()
    header = (
        f"{'Layer':<22} {'N':>7}  {'WRF_MBE':>8} {'CFD_MBE':>8} "
        f"{'WRF_RMSE':>9} {'CFD_RMSE':>9} {'WRF_IoA':>8} "
        f"{'CFD_IoA':>8}  {'SS':>7}"
    )
    print(header)
    print("-" * len(header))

    for layer in pm.summary_layers(skip_high=skip_high):
        g = grp_df[grp_df["layer"] == layer]
        if len(g) < 5:
            continue
        o, w, c = g["ws_obs"].values, g["ws_wrf"].values, g["ws_cfd"].values
        print(
            f"{layer:<22} {len(g):>7}  "
            f"{pm.mean_bias_error(w, o):>+8.3f} {pm.mean_bias_error(c, o):>+8.3f} "
            f"{pm.rmse(w, o):>9.3f} {pm.rmse(c, o):>9.3f} "
            f"{pm.index_of_agreement(w, o):>8.3f} {pm.index_of_agreement(c, o):>8.3f}  "
            f"{pm.skill_score(c, o, w):>+7.3f}"
        )


def section_baseline(sub: pd.DataFrame, *, skip_high: bool = False) -> None:
    print_layer_summary("Layer-aggregated summary - ALL times (baseline)", sub, skip_high=skip_high)


def section_utc_groups(sub: pd.DataFrame, *, skip_high: bool = False) -> None:
    sub_h = _attach_utc_hour(sub)
    for label, hours in UTC_TIME_GROUPS.items():
        g = sub_h[sub_h["utc_hour"].isin(hours)]
        print_layer_summary(label, g, skip_high=skip_high)


def section_lst_periods(sub: pd.DataFrame, *, skip_high: bool = False) -> None:
    sub_l = _attach_lst(sub)
    lst = sub_l["lst"]
    for day in pm.metric_lst_days():
        for period, mask in (
            ("daytime (LST 07-18)", mask_lst_day_am(lst, day)),
            ("nighttime (LST 19 - next day 06)", mask_lst_day_pm(lst, day)),
        ):
            g = sub_l[mask]
            print_layer_summary(
                f"2025-09-{day:02d} {period}",
                g,
                skip_high=skip_high,
            )


def section_by_utc_calendar_day(sub: pd.DataFrame, *, skip_high: bool = False) -> None:
    sub = sub.copy()
    sub["utc_date"] = sub["datetime"].dt.strftime("%Y-%m-%d")
    for d in pm.metric_utc_dates():
        g = sub[sub["utc_date"] == d]
        print_layer_summary(f"UTC calendar day - {d}", g, skip_high=skip_high)


def section_conservative(sub: pd.DataFrame, *, skip_high: bool = False) -> None:
    sub_h = _attach_utc_hour(sub)
    g = sub_h[sub_h["utc_hour"].isin(CONSERVATIVE_UTC_HOURS)]
    print_layer_summary(
        "Conservative subset - LST 08-19",
        g,
        skip_high=skip_high,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print layer-aggregated metrics for multiple sample definitions "
        "(UTC regimes, LST day×AM/PM, by UTC day, conservative hours)."
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
        "--no-high",
        action="store_true",
        help="Omit the High (1000–2000 m) layer row from summary tables.",
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

    path = args.data if args.data is not None else pm.DATA_PATH
    df_raw = pm.load_and_preprocess(path)
    df = pm.quality_control(df_raw)
    sub = _base_subset(df)

    print(
        f"[Config] Time range: {pm.METRIC_START} .. {pm.METRIC_END}  "
        f"(N rows after QC: {len(sub):,})"
    )

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.3f}".format)

    if "baseline" in sections:
        section_baseline(sub, skip_high=args.no_high)
    if "utc_groups" in sections:
        section_utc_groups(sub, skip_high=args.no_high)
    if "lst_periods" in sections:
        section_lst_periods(sub, skip_high=args.no_high)
    if "by_utc_day" in sections:
        section_by_utc_calendar_day(sub, skip_high=args.no_high)
    if "conservative" in sections:
        section_conservative(sub, skip_high=args.no_high)


if __name__ == "__main__":
    main()
