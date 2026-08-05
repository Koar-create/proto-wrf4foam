"""Shared settings for 20250906–13 OpenFOAM experiments.

Sep 6–8: include CFD cases that have completed (case/<5000> exists).
Sep 9–13: keep synoptic UTC 00/06/12/18 slots.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ID = "260707"

UTC_DAY_FIRST = 6
UTC_DAY_LAST = 13
SYNOPTIC_UTC_HOURS = (0, 6, 12, 18)

HOURLY_DAY_FIRST = 6
HOURLY_DAY_LAST = 8
CASES_ROOT = REPO_ROOT / "steady_experiments_finer_ABL"
CASE_SUFFIX = "_two_boundaries_as_outlet"
COMPLETION_MARKER = "5000"

METRIC_START = f"2025-09-{UTC_DAY_FIRST:02d} 00:00:00"
METRIC_END = f"2025-09-{UTC_DAY_LAST:02d} 18:00:00"


def completed_hourly_datetimes(
    day_first: int = HOURLY_DAY_FIRST,
    day_last: int = HOURLY_DAY_LAST,
    *,
    cases_root: Path = CASES_ROOT,
    marker: str = COMPLETION_MARKER,
) -> list[pd.Timestamp]:
    """Return UTC timestamps for Sep day_first–day_last cases with <marker> present."""
    slots: list[pd.Timestamp] = []
    for day in range(day_first, day_last + 1):
        for hour in range(24):
            stamp = f"202509{day:02d}_{hour:02d}00"
            case_dir = cases_root / f"{stamp}{CASE_SUFFIX}"
            if (case_dir / marker).is_dir():
                slots.append(pd.Timestamp(f"2025-09-{day:02d} {hour:02d}:00:00"))
    return slots


def build_metric_datetimes() -> pd.DatetimeIndex:
    slots = completed_hourly_datetimes()
    for day in range(HOURLY_DAY_LAST + 1, UTC_DAY_LAST + 1):
        for hour in SYNOPTIC_UTC_HOURS:
            slots.append(pd.Timestamp(f"2025-09-{day:02d} {hour:02d}:00:00"))
    return pd.DatetimeIndex(slots)


METRIC_DATETIMES = build_metric_datetimes()
TIME_LABELS = {
    dt.strftime("%Y-%m-%d %H:%M:%S"): f"{dt.day:02d}_{dt.strftime('%H00')} UTC"
    for dt in METRIC_DATETIMES
}

DATA_DIR = REPO_ROOT / "data" / CAMPAIGN_ID
ANALYSIS_DIR = REPO_ROOT / "analysis" / CAMPAIGN_ID
DATA_PATH = DATA_DIR / "processed" / "merged_lidar_simulation_final.csv"
RESULTS_TAG = CAMPAIGN_ID

SAMPLE_DATE_MIN = pd.Timestamp("2025-09-06")
SAMPLE_DATE_MAX = pd.Timestamp("2025-09-13")
CAMPAIGN_TITLE = "6–13 September 2025"
CAMPAIGN_TITLE_LONG = f"WRF vs. OpenFOAM wind-speed evaluation, {CAMPAIGN_TITLE}"


def metric_utc_dates() -> list[str]:
    return sorted({dt.strftime("%Y-%m-%d") for dt in METRIC_DATETIMES})


def metric_utc_days() -> list[int]:
    return sorted({dt.day for dt in METRIC_DATETIMES})


def metric_lst_days() -> list[int]:
    return sorted({(dt + pd.Timedelta(hours=8)).day for dt in METRIC_DATETIMES})


def cfd_filename(dt: pd.Timestamp) -> str:
    return f"CFD_lidar_simulation_{dt.strftime('%Y%m%d_%H00')}_two_boundaries_as_outlet.csv"
