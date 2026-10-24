#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Resolve URANS pimpleFoam case paths from a time tag (YYYYMMDD_HHMM)."""
from __future__ import annotations

from pathlib import Path

_UTIL_DIR = Path(__file__).resolve().parent
URANS_ROOT = _UTIL_DIR.parent
REPO_ROOT = URANS_ROOT.parent.parent

KNOWN_TAGS = ("20250901_1000", "20250908_0000")

_CASE_META = {
    "20250901_1000": {
        "datetime_utc": "2025-09-01 10:00:00",
        "merged_csv": REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv",
        "slurm_job_name": "URANS_0901",
        "slurm_walltime": "12:00:00",
    },
    "20250908_0000": {
        "datetime_utc": "2025-09-08 00:00:00",
        "merged_csv": REPO_ROOT / "data/260707/processed/merged_lidar_simulation_final.csv",
        "slurm_job_name": "URANS_0908",
        "slurm_walltime": "24:00:00",
    },
}


def normalize_tag(tag: str) -> str:
    t = str(tag).strip().rstrip("/")
    t = t.split("/")[-1]
    if t.endswith("_pimplefoam"):
        t = t[: -len("_pimplefoam")]
    if t not in _CASE_META:
        known = ", ".join(KNOWN_TAGS)
        raise ValueError(f"unknown tag {tag!r}; expected one of: {known}")
    return t


def datetime_utc(tag: str) -> str:
    return _CASE_META[normalize_tag(tag)]["datetime_utc"]


def merged_csv(tag: str) -> Path:
    return _CASE_META[normalize_tag(tag)]["merged_csv"]


def slurm_job_name(tag: str) -> str:
    return _CASE_META[normalize_tag(tag)]["slurm_job_name"]


def slurm_walltime(tag: str) -> str:
    return _CASE_META[normalize_tag(tag)]["slurm_walltime"]


def case_dir(tag: str) -> Path:
    t = normalize_tag(tag)
    return URANS_ROOT / f"{t}_pimplefoam"


def rel_case_dir(tag: str) -> str:
    t = normalize_tag(tag)
    return f"experiments/urans/{t}_pimplefoam"


def simplefoam_dir(tag: str) -> Path:
    t = normalize_tag(tag)
    return REPO_ROOT / "steady_experiments_finer_ABL" / f"{t}_two_boundaries_as_outlet"


def postprocessing_dir(tag: str) -> Path:
    return case_dir(tag) / "postProcessing"


def figs_dir() -> Path:
    return REPO_ROOT / "results" / "urans"
