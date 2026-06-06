"""
print_metric_site_variants.py — 在 print_metric.py 基础上按「站点子集」复算分层汇总表。

与 print_metric_sample_variants.py 对称：后者改时间范围，本脚本改站点范围，
用于识别 metric 表现优秀或糟糕的 LiDAR 站点。

策略：
  1) baseline：全站点（对照基线）；
  2) only_site：仅保留单个站点；
  3) exclude_site：排除单个站点（leave-one-out）；
  4) site_rank：各站单独汇总 + 与 baseline 对比的紧凑排名表（便于快速定位优劣站）。

指标与 print_metric.main() 中表格一致（行级 N 为格点数）。
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


def _base_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["qc_ok"] & df["ws_obs"].notna()].copy()


def _sites_in_data(sub: pd.DataFrame) -> list[str]:
    return sorted(sub["obtid"].unique())


def print_layer_summary(title: str, grp_df: pd.DataFrame, *, skip_high: bool = False) -> None:
    """与 print_metric.main() 相同列的分层汇总。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()
    header = (
        f"{'Layer':<22} {'N':>7}  {'WRF_MBE':>8} {'WRF_RMSE':>9} "
        f"{'WRF_IoA':>8}  {'CFD_MBE':>8} {'CFD_RMSE':>9} "
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
            f"{pm.mean_bias_error(w, o):>+8.3f} {pm.rmse(w, o):>9.3f} "
            f"{pm.index_of_agreement(w, o):>8.3f}  "
            f"{pm.mean_bias_error(c, o):>+8.3f} {pm.rmse(c, o):>9.3f} "
            f"{pm.index_of_agreement(c, o):>8.3f}  "
            f"{pm.skill_score(c, o, w):>+7.3f}"
        )


def _pooled_metrics(grp_df: pd.DataFrame) -> dict[str, float | int] | None:
    """全高度合并为一行指标（用于站点排名）。"""
    if len(grp_df) < 5:
        return None
    o, w, c = grp_df["ws_obs"].values, grp_df["ws_wrf"].values, grp_df["ws_cfd"].values
    return {
        "N": len(grp_df),
        "WRF_RMSE": pm.rmse(w, o),
        "WRF_IoA": pm.index_of_agreement(w, o),
        "CFD_RMSE": pm.rmse(c, o),
        "CFD_IoA": pm.index_of_agreement(c, o),
        "SS": pm.skill_score(c, o, w),
    }


def section_baseline(sub: pd.DataFrame, *, skip_high: bool = False) -> None:
    print_layer_summary("Layer-aggregated summary - ALL sites (baseline)", sub, skip_high=skip_high)


def section_only_site(sub: pd.DataFrame, sites: list[str], *, skip_high: bool = False) -> None:
    for site in sites:
        g = sub[sub["obtid"] == site]
        print_layer_summary(f"Only site - {site}", g, skip_high=skip_high)


def section_exclude_site(sub: pd.DataFrame, sites: list[str], *, skip_high: bool = False) -> None:
    for site in sites:
        g = sub[sub["obtid"] != site]
        print_layer_summary(f"Exclude site (leave-one-out) - drop {site}", g, skip_high=skip_high)


def section_site_rank(sub: pd.DataFrame, sites: list[str]) -> None:
    """各站单独 + leave-one-out 的紧凑对比（全高度合并）。"""
    baseline = _pooled_metrics(sub)
    if baseline is None:
        print("\n[site_rank] Baseline N < 5, skipped.")
        return

    rows: list[dict] = []
    rows.append({"Strategy": "baseline (all sites)", "Site": "-", **baseline})

    for site in sites:
        m = _pooled_metrics(sub[sub["obtid"] == site])
        if m is not None:
            rows.append({"Strategy": "only", "Site": site, **m})

    for site in sites:
        m = _pooled_metrics(sub[sub["obtid"] != site])
        if m is not None:
            rows.append({"Strategy": "exclude", "Site": site, **m})

    rank_df = pd.DataFrame(rows)
    rank_df["CFD_IoA_vs_baseline"] = rank_df["CFD_IoA"] - baseline["CFD_IoA"]
    rank_df["SS_vs_baseline"] = rank_df["SS"] - baseline["SS"]

    print("\n" + "=" * 70)
    print("  Site sensitivity rank (height-pooled, vs baseline)")
    print("=" * 70)
    print()
    cols = [
        "Strategy",
        "Site",
        "N",
        "CFD_RMSE",
        "CFD_IoA",
        "SS",
        "CFD_IoA_vs_baseline",
        "SS_vs_baseline",
    ]
    print(rank_df[cols].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    only_df = rank_df[rank_df["Strategy"] == "only"].sort_values("CFD_IoA", ascending=False)
    exclude_df = rank_df[rank_df["Strategy"] == "exclude"].sort_values("CFD_IoA", ascending=False)

    print("\n-- Per-site only (CFD_IoA high -> good, low -> bad) --")
    print(only_df[["Site", "CFD_IoA", "SS", "CFD_RMSE"]].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\n-- Leave-one-out exclude (CFD_IoA rises after drop -> removed site was bad) --")
    print(exclude_df[["Site", "CFD_IoA", "SS", "CFD_IoA_vs_baseline"]].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print layer-aggregated metrics for multiple site sample definitions "
        "(baseline, only one site, leave-one-out exclude, compact site rank)."
    )
    p.add_argument(
        "--sections",
        nargs="+",
        choices=[
            "baseline",
            "only_site",
            "exclude_site",
            "site_rank",
            "all",
        ],
        default=["all"],
        help="Which blocks to print. Use 'all' for every block (default).",
    )
    p.add_argument(
        "--sites",
        nargs="+",
        default=None,
        metavar="OBTID",
        help="Limit analysis to these obtid values (default: all sites in data).",
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
        sections = {"baseline", "only_site", "exclude_site", "site_rank"}

    path = args.data if args.data is not None else pm.DATA_PATH
    df_raw = pm.load_and_preprocess(path)
    df = pm.quality_control(df_raw)
    sub = _base_subset(df)

    all_sites = _sites_in_data(sub)
    if args.sites is not None:
        unknown = set(args.sites) - set(all_sites)
        if unknown:
            raise SystemExit(f"Unknown site(s): {sorted(unknown)}. Available: {all_sites}")
        sites = sorted(args.sites)
        sub = sub[sub["obtid"].isin(sites)].copy()
    else:
        sites = all_sites

    print(
        f"[Config] Time range: {pm.METRIC_START} .. {pm.METRIC_END}  "
        f"| Sites: {', '.join(sites)}  (N rows after QC: {len(sub):,})"
    )

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.3f}".format)

    if "baseline" in sections:
        section_baseline(sub, skip_high=args.no_high)
    if "only_site" in sections:
        section_only_site(sub, sites, skip_high=args.no_high)
    if "exclude_site" in sections:
        section_exclude_site(sub, sites, skip_high=args.no_high)
    if "site_rank" in sections:
        section_site_rank(sub, sites)


if __name__ == "__main__":
    main()
