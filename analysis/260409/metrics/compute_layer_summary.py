"""Write the control-run layer-aggregated wind-speed summary table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ANALYSIS = Path(__file__).resolve().parents[2]
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import batch
from metrics.stats import compute_layer_summary, load_and_preprocess, quality_control


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write layer-aggregated wind speed metrics (WRF / OpenFOAM vs LiDAR)."
    )
    parser.add_argument("--csv", type=Path, default=batch.DATA_PATH)
    parser.add_argument("--out", type=Path, default=batch.LAYER_SUMMARY_CSV)
    parser.add_argument(
        "--no-high",
        action="store_true",
        help="Omit the High (1000–2000 m) layer row from the summary table.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = quality_control(load_and_preprocess(args.csv, time_labels=batch.TIME_LABELS))
    layer_summary = compute_layer_summary(df, skip_high=args.no_high)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    layer_summary.to_csv(args.out, index=False)
    print(f"[CSV] Layer summary written to: {args.out}")

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
