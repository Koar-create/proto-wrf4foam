"""
print_metric_ref_sensitivity.py — 单时次 river/urban split 敏感性 CSV 的分层风速指标表。

参考 analysis/260409/print_metric_sample_variants.py 的表格风格，但：
  - 数据源仅含一个 datetime（不再按 UTC/LST/日历日划分子集）；
  - CFD 列拆为 reference（ws_cfd_ref）与 sensitivity（ws_cfd_sen）。

指标与 print_metric.py 一致；SS_ref / SS_sen 以 WRF 为 baseline；
SS_sen_vs_ref 以 CFD reference 为 baseline（敏感性相对参考的技巧得分）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_260409 = _REPO_ROOT / "analysis" / "260409"
if str(_260409) not in sys.path:
    sys.path.insert(0, str(_260409))

import print_metric as pm  # noqa: E402

DEFAULT_DATA = (
    _REPO_ROOT
    / "data"
    / "260513"
    / "processed"
    / "merged_lidar_simulation_final_river_urban_split_sensitivity_run.csv"
)


def load_and_preprocess(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])

    df["ws_cfd_ref"] = np.sqrt(df["u_cfd_ref"] ** 2 + df["v_cfd_ref"] ** 2)
    df["ws_cfd_sen"] = np.sqrt(df["u_cfd_sen"] ** 2 + df["v_cfd_sen"] ** 2)

    df["wd_obs"] = np.degrees(np.arctan2(-df["u_obs"], -df["v_obs"])) % 360
    df["wd_wrf"] = np.degrees(np.arctan2(-df["u_wrf"], -df["v_wrf"])) % 360
    df["wd_cfd_ref"] = np.degrees(np.arctan2(-df["u_cfd_ref"], -df["v_cfd_ref"])) % 360
    df["wd_cfd_sen"] = np.degrees(np.arctan2(-df["u_cfd_sen"], -df["v_cfd_sen"])) % 360

    df["layer"] = pd.cut(df["Height"], bins=pm.HEIGHT_BINS, labels=pm.LAYER_NAMES_EN)
    df["time_label"] = df["datetime"].astype(str)
    return df


def quality_control(df: pd.DataFrame) -> pd.DataFrame:
    """与 print_metric 类似：观测上限 + 两套 CFD 均需未发散，行才计入 qc_ok。"""
    obs_ok = (df["ws_obs"] <= pm.WS_MAX_OBS) | df["ws_obs"].isna()
    ref_ok = df["ws_cfd_ref"] <= pm.WS_MAX_CFD
    sen_ok = df["ws_cfd_sen"] <= pm.WS_MAX_CFD

    out = df.copy()
    out["qc_obs_ok"] = obs_ok
    out["qc_cfd_ref_ok"] = ref_ok
    out["qc_cfd_sen_ok"] = sen_ok
    out["qc_ok"] = obs_ok & ref_ok & sen_ok
    return out


def _base_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["qc_ok"] & df["ws_obs"].notna()].copy()


def print_layer_summary_dual_cfd(title: str, grp_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()
    header = (
        f"{'Layer':<22} {'N':>7}  "
        f"{'WRF_MBE':>8} {'WRF_RMSE':>9} {'WRF_IoA':>8}  "
        f"{'CFDref_MBE':>10} {'CFDref_RMSE':>11} {'CFDref_IoA':>10}  "
        f"{'CFDsen_MBE':>10} {'CFDsen_RMSE':>11} {'CFDsen_IoA':>10}  "
        f"{'SSref':>7} {'SSsen':>7} {'SSs_v_r':>7}"
    )
    print(header)
    print("-" * len(header))

    for layer in pm.LAYER_NAMES_EN:
        g = grp_df[grp_df["layer"] == layer]
        if len(g) < 5:
            continue
        o = g["ws_obs"].values
        w = g["ws_wrf"].values
        cr = g["ws_cfd_ref"].values
        cs = g["ws_cfd_sen"].values
        print(
            f"{layer:<22} {len(g):>7}  "
            f"{pm.mean_bias_error(w, o):>+8.3f} {pm.rmse(w, o):>9.3f} "
            f"{pm.index_of_agreement(w, o):>8.3f}  "
            f"{pm.mean_bias_error(cr, o):>+10.3f} {pm.rmse(cr, o):>11.3f} "
            f"{pm.index_of_agreement(cr, o):>10.3f}  "
            f"{pm.mean_bias_error(cs, o):>+10.3f} {pm.rmse(cs, o):>11.3f} "
            f"{pm.index_of_agreement(cs, o):>10.3f}  "
            f"{pm.skill_score(cr, o, w):>+7.3f} {pm.skill_score(cs, o, w):>+7.3f} "
            f"{pm.skill_score(cs, o, cr):>+7.3f}"
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print layer-aggregated wind metrics (WRF + CFD ref/sen) "
        "for single-time river_urban_split sensitivity merged CSV."
    )
    p.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Merged CSV path (default: data/260513/processed/...sensitivity_run.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    path = args.data.resolve()

    df_raw = load_and_preprocess(path)
    df = quality_control(df_raw)
    sub = _base_subset(df)

    times = sub["datetime"].drop_duplicates().sort_values()
    t_str = ", ".join(times.dt.strftime("%Y-%m-%d %H:%M:%S").astype(str))
    title = f"Layer-aggregated summary - ALL rows (single-time: {t_str})"

    pd.set_option("display.max_columns", 24)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)

    print(f"Data: {path}")
    print(f"Unique datetimes in filtered subset: {times.shape[0]}")
    print("SSref / SSsen: skill vs WRF; SSs_v_r: skill of sensitivity vs CFD reference.")

    print_layer_summary_dual_cfd(title, sub)


if __name__ == "__main__":
    main()
