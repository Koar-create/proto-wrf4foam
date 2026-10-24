#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Resolve non-isothermal BBSF case paths from a time tag (YYYYMMDD_HHMM)."""
from __future__ import annotations

from pathlib import Path

_UTIL_DIR = Path(__file__).resolve().parent
NONISO_ROOT = _UTIL_DIR.parent
REPO_ROOT = NONISO_ROOT.parent.parent

KNOWN_TAGS = ("20250901_1000", "20250908_0000")

_CASE_META = {
    "20250901_1000": {
        "datetime_utc": "2025-09-01 10:00:00",
        "merged_csv": REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv",
    },
    "20250908_0000": {
        "datetime_utc": "2025-09-08 00:00:00",
        "merged_csv": REPO_ROOT / "data/260707/processed/merged_lidar_simulation_final.csv",
    },
}


def normalize_tag(tag: str) -> str:
    t = str(tag).strip().rstrip("/")
    t = t.split("/")[-1]
    if t.endswith("_bbsf"):
        t = t[: -len("_bbsf")]
    if t not in _CASE_META:
        known = ", ".join(KNOWN_TAGS)
        raise ValueError(f"unknown tag {tag!r}; expected one of: {known}")
    return t


def datetime_utc(tag: str) -> str:
    return _CASE_META[normalize_tag(tag)]["datetime_utc"]


def merged_csv(tag: str) -> Path:
    return _CASE_META[normalize_tag(tag)]["merged_csv"]


def case_dir(tag: str) -> Path:
    t = normalize_tag(tag)
    return NONISO_ROOT / f"{t}_bbsf"


def rel_case_dir(tag: str) -> str:
    t = normalize_tag(tag)
    return f"experiments/nonisothermal/{t}_bbsf"


def simplefoam_dir(tag: str) -> Path:
    t = normalize_tag(tag)
    return REPO_ROOT / "steady_experiments_finer_ABL" / f"{t}_two_boundaries_as_outlet"


def bbsf_lidar_csv(tag: str) -> Path:
    t = normalize_tag(tag)
    return case_dir(t) / "postProcessing" / f"CFD_lidar_simulation_{t}_bbsf.csv"


def postprocessing_dir(tag: str) -> Path:
    return case_dir(tag) / "postProcessing"


def figs_dir() -> Path:
    return REPO_ROOT / "results" / "nonisothermal"
