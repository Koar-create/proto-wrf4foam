"""Shared settings for 20250906–13 OpenFOAM experiments (UTC 00/06/12/18, 32 cases)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ID = "260707"

UTC_DAY_FIRST = 6
UTC_DAY_LAST = 13
SYNOPTIC_UTC_HOURS = (0, 6, 12, 18)

METRIC_START = f"2025-09-{UTC_DAY_FIRST:02d} 00:00:00"
METRIC_END = f"2025-09-{UTC_DAY_LAST:02d} 18:00:00"


def build_metric_datetimes() -> pd.DatetimeIndex:
    slots = [
        pd.Timestamp(f"2025-09-{day:02d} {hour:02d}:00:00")
        for day in range(UTC_DAY_FIRST, UTC_DAY_LAST + 1)
        for hour in SYNOPTIC_UTC_HOURS
    ]
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
