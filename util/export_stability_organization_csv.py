#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export WRF inflow-boundary stability diagnostics to a CSV table.

Default input range:
  steady_experiments_finer_ABL/202509[01-05]_[00-23]00_two_boundaries_as_outlet

Default output:
  data/wrf_stability_20250901-20250905.csv

The output schema follows docs/WRF Atmospheric Stability Data Organization.csv,
but this script does not overwrite that reference file.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from diagnose_stability_diag import read_foam_field


CSV_HEADER = [
    "Record (UTC)",
    "Beijing Time",
    "Boundary Patch",
    "k_max (m²/s²)",
    "Avg k > 500m (m²/s²)",
    "Lt_max (m)",
    "Stability Regime",
]


@dataclass(frozen=True)
class StabilityMetrics:
    k_max: float
    k_aloft_mean: float
    lt_max: float
    regime: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_cases_root() -> Path:
    return repo_root() / "steady_experiments_finer_ABL"


def default_output_csv() -> Path:
    return repo_root() / "data" / "wrf_stability_20250901-20250905.csv"


def iter_records(start: str, end: str):
    """Yield hourly UTC records from YYYYMMDD through YYYYMMDD inclusive."""
    current = datetime.strptime(start, "%Y%m%d")
    final = datetime.strptime(end, "%Y%m%d") + timedelta(hours=23)
    while current <= final:
        yield current.strftime("%Y%m%d_%H%M")
        current += timedelta(hours=1)


def beijing_time_label(record_utc: str) -> str:
    dt_utc = datetime.strptime(record_utc, "%Y%m%d_%H%M")
    hour = (dt_utc + timedelta(hours=8)).hour
    suffix = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12 or 12
    return f"{hour_12}{suffix}"


def format_float(value: float, digits: int) -> str:
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def analyze_patch(patch_dir: Path) -> StabilityMetrics:
    pts = read_foam_field(str(patch_dir / "points"), is_vector=True)
    k_field = read_foam_field(str(patch_dir / "0" / "k"), is_vector=False)
    eps_field = read_foam_field(str(patch_dir / "0" / "epsilon"), is_vector=False)

    if any(v is None for v in [pts, k_field, eps_field]):
        raise FileNotFoundError(f"Missing points, k, or epsilon under {patch_dir}")

    z_values = np.round(pts[:, 2], decimals=1)
    z_unique = np.unique(z_values)

    k_prof = []
    eps_prof = []
    for z_value in z_unique:
        mask = z_values == z_value
        k_prof.append(np.mean(k_field[mask]))
        eps_prof.append(np.mean(eps_field[mask]))

    k_prof = np.asarray(k_prof)
    eps_prof = np.asarray(eps_prof)

    mask_aloft = z_unique > 500
    k_aloft_mean = float(np.mean(k_prof[mask_aloft])) if np.any(mask_aloft) else float(np.mean(k_prof))
    k_max = float(np.max(k_prof))

    c_mu = 0.09
    lt_prof = (c_mu**0.75) * (k_prof**1.5) / (eps_prof + 1e-15)
    lt_max = float(np.max(lt_prof))

    if k_aloft_mean < 0.05 and k_max < 0.6:
        regime = "Strongly Stable"
    elif k_aloft_mean > 0.2 or k_max > 1.5:
        regime = "Unstable / Convective"
    else:
        regime = "Neutral / Weakly Stable"

    return StabilityMetrics(
        k_max=k_max,
        k_aloft_mean=k_aloft_mean,
        lt_max=lt_max,
        regime=regime,
    )


def build_rows(cases_root: Path, start: str, end: str, patches: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for record in iter_records(start, end):
        case_dir = cases_root / f"{record}_two_boundaries_as_outlet"
        boundary_data_dir = case_dir / "constant" / "boundaryData"
        if not boundary_data_dir.is_dir():
            raise FileNotFoundError(f"Missing boundaryData directory: {boundary_data_dir}")

        for patch in patches:
            patch_dir = boundary_data_dir / patch
            if not patch_dir.is_dir():
                raise FileNotFoundError(f"Missing patch directory: {patch_dir}")

            metrics = analyze_patch(patch_dir)
            rows.append(
                [
                    record,
                    beijing_time_label(record),
                    patch.capitalize(),
                    format_float(metrics.k_max, 4),
                    format_float(metrics.k_aloft_mean, 4),
                    format_float(metrics.lt_max, 1),
                    metrics.regime,
                ]
            )
    return rows


def write_csv(output_csv: Path, rows: list[list[str]], overwrite: bool) -> None:
    if output_csv.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {output_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export WRF atmospheric stability diagnostics to a new CSV."
    )
    parser.add_argument("--cases-root", type=Path, default=default_cases_root())
    parser.add_argument("--out", type=Path, default=default_output_csv())
    parser.add_argument("--start", default="20250901", help="Start date in YYYYMMDD, inclusive")
    parser.add_argument("--end", default="20250905", help="End date in YYYYMMDD, inclusive")
    parser.add_argument("--patches", nargs="+", default=["east", "south"])
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing the output CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.cases_root, args.start, args.end, [p.lower() for p in args.patches])
    write_csv(args.out, rows, args.overwrite)
    print(f"Saved {len(rows)} rows: {args.out}")


if __name__ == "__main__":
    main()
