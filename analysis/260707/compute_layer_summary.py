"""Layer-aggregated metrics for the 260707 batch. Logic lives in analysis.metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ANALYSIS = Path(__file__).resolve().parents[1]
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import batch
from metrics.layers import (  # noqa: F401
    HEIGHT_BINS,
    LAYER_NAMES_EN,
    WS_MAX_CFD,
    WS_MAX_OBS,
    summary_layers,
)
from metrics.stats import (
    circular_error_stats,  # noqa: F401
    compute_layer_summary,
    compute_metrics_table,
    fractional_bias,  # noqa: F401
    index_of_agreement,  # noqa: F401
    load_and_preprocess as _load_and_preprocess,
    mean_bias_error,  # noqa: F401
    quality_control,
    rmse,  # noqa: F401
    skill_score,  # noqa: F401
)
from metrics.style import COLOR_CFD, COLOR_OBS, COLOR_WRF  # noqa: F401

DATA_PATH = batch.DATA_PATH
LAYER_SUMMARY_CSV = batch.LAYER_SUMMARY_CSV
METRIC_START = batch.METRIC_START
METRIC_END = batch.METRIC_END
METRIC_DATETIMES = batch.METRIC_DATETIMES
TIME_LABELS = batch.TIME_LABELS
metric_utc_dates = batch.metric_utc_dates
metric_lst_days = batch.metric_lst_days


def load_and_preprocess(path: str | Path) -> pd.DataFrame:
    return _load_and_preprocess(path, time_labels=TIME_LABELS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print layer-aggregated wind speed metrics (WRF / OpenFOAM vs LiDAR)."
    )
    parser.add_argument(
        "--no-high",
        action="store_true",
        help="Omit the High (1000–2000 m) layer row from the summary table.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = quality_control(load_and_preprocess(DATA_PATH))
    compute_metrics_table(df)
    layer_summary = compute_layer_summary(df, skip_high=args.no_high)
    LAYER_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    layer_summary.to_csv(LAYER_SUMMARY_CSV, index=False)
    print(f"\n[CSV] Layer summary written to: {LAYER_SUMMARY_CSV}")

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.3f}".format)

    print("\n" + "=" * 70)
    print("  Layer-aggregated summary (all sites + all times)")
    print("=" * 70)
    print("\n-- All times --")
    header = f"{'Layer':<22} {'N':>5}  {'WRF_MBE':>8} {'WRF_RMSE':>9} "
    header += f"{'WRF_IoA':>8}  {'CFD_MBE':>8} {'CFD_RMSE':>9} {'CFD_IoA':>8}  {'SS':>7}"
    print(header)
    print("-" * len(header))
    for _, row in layer_summary.iterrows():
        print(
            f"{row['Layer']:<22} {int(row['N']):>5}  "
            f"{row['WRF_MBE']:>+8.3f} {row['WRF_RMSE']:>9.3f} "
            f"{row['WRF_IoA']:>8.3f}  "
            f"{row['CFD_MBE']:>+8.3f} {row['CFD_RMSE']:>9.3f} "
            f"{row['CFD_IoA']:>8.3f}  "
            f"{row['SS']:>+7.3f}"
        )


if __name__ == "__main__":
    main()
