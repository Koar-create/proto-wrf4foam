"""260409 batch paths, study windows, and control-run metric dates.

流程总览（图例、各条 compute/plot 链）：analysis/metrics/README.md

稿件 Fig.5：

    python analysis/260409/metrics/compute_subset_metrics.py
    python analysis/260409/metrics/plot_subset_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ANALYSIS = Path(__file__).resolve().parents[2]
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

REPO_ROOT = Path(__file__).resolve().parents[3]
BATCH_ID = "260409"

DATA_PATH = REPO_ROOT / "data" / BATCH_ID / "processed" / "merged_lidar_simulation_final.csv"
MERGED_CSVS = [
    DATA_PATH,
    REPO_ROOT / "data" / "260707" / "processed" / "merged_lidar_simulation_final.csv",
]

TABLES_DIR = REPO_ROOT / "results" / "metrics" / BATCH_ID / "tables"
FIGURES_DIR = REPO_ROOT / "results" / "metrics" / BATCH_ID / "figures"

LAYER_SUMMARY_CSV = TABLES_DIR / "ws_layer_summary.csv"
SUBSET_METRICS_CSV = TABLES_DIR / "ws_subset_metrics.csv"
SITE_HEIGHT_CSV = TABLES_DIR / "ws_site_height.csv"
PROFILE_SHEAR_CSV = TABLES_DIR / "ws_profile_shear.csv"
WD_CIRCULAR_CSV = TABLES_DIR / "wd_circular.csv"
WD_VECTOR_CSV = TABLES_DIR / "wd_vector.csv"
WD_VEER_CSV = TABLES_DIR / "wd_veer.csv"

# Control-run coverage used by compute_layer_summary (hourly through 5 Sep).
METRIC_START = "2025-09-01 00:00:00"
METRIC_END = "2025-09-05 23:00:00"
METRIC_DATETIMES = pd.date_range(METRIC_START, METRIC_END, freq="h")
TIME_LABELS = {
    dt.strftime("%Y-%m-%d %H:%M:%S"): f"{dt.day:02d}_{dt.strftime('%H00')} UTC"
    for dt in METRIC_DATETIMES
}

# Manuscript subset window (§2.1).
STUDY_START = pd.Timestamp("2025-09-01 00:00:00", tz="UTC")
STUDY_END = pd.Timestamp("2025-09-09 23:00:00", tz="UTC")
STRONG_WIND_START = pd.Timestamp("2025-09-07 03:00:00", tz="UTC")
STRONG_WIND_END = pd.Timestamp("2025-09-08 19:00:00", tz="UTC")
DAILY_PANEL_START = pd.Timestamp("2025-09-01")
DAILY_PANEL_END = pd.Timestamp("2025-09-09")
LST_DAYS = list(range(1, 10))
DAY_UTC_HOURS = [23, *range(0, 11)]
NIGHT_UTC_HOURS = list(range(11, 23))
SUMMARY_CATEGORIES = ["All", "Daytime", "Nighttime", "Strong-wind"]


def metric_utc_dates() -> list[str]:
    return sorted({dt.strftime("%Y-%m-%d") for dt in METRIC_DATETIMES})


def metric_lst_days() -> list[int]:
    return sorted({(dt + pd.Timedelta(hours=8)).day for dt in METRIC_DATETIMES})
