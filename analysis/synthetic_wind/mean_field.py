"""Temporal interpolation of hourly RANS mean fields (PCHIP)."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

import numpy as np
from scipy.interpolate import PchipInterpolator

from .config import list_case_ids


def case_id_to_datetime(case_id: str) -> datetime:
    m = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})_", case_id)
    if not m:
        raise ValueError(f"Cannot parse datetime from {case_id}")
    y, mo, d, h, mi = map(int, m.groups())
    return datetime(y, mo, d, h, mi)


def build_time_axis(case_ids: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    """Return (case_ids, seconds since first case)."""
    ids = case_ids or list_case_ids()
    ids = sorted(ids, key=case_id_to_datetime)
    t0 = case_id_to_datetime(ids[0])
    seconds = np.array(
        [(case_id_to_datetime(cid) - t0).total_seconds() for cid in ids],
        dtype=np.float64,
    )
    return ids, seconds


class MeanFieldInterpolator:
    """PCHIP interpolation of U,k,epsilon,nut across hourly snapshots."""

    def __init__(self, case_data: dict[str, dict]):
        """
        case_data: case_id -> dict with U,k,epsilon,nut arrays (same grid).
        """
        self.case_ids, self.times = build_time_axis(list(case_data.keys()))
        self._data = case_data
        self._fields = ("U", "k", "epsilon", "nut")

    def at_time(self, t_sec: float) -> dict[str, np.ndarray]:
        """Interpolate all fields at continuous time (seconds from first case)."""
        out = {}
        for field in self._fields:
            stack = np.stack([self._data[cid][field] for cid in self.case_ids], axis=0)
            # interpolate each voxel along time axis
            flat = stack.reshape(len(self.case_ids), -1)
            interp = PchipInterpolator(self.times, flat, axis=0, extrapolate=True)
            out[field] = interp(t_sec).reshape(stack.shape[1:])
        return out

    def at_datetime(self, dt: datetime) -> dict[str, np.ndarray]:
        t0 = case_id_to_datetime(self.case_ids[0])
        return self.at_time((dt - t0).total_seconds())

    def window_times(
        self,
        center_case_id: str,
        duration_s: float,
        dt_s: float,
    ) -> np.ndarray:
        """Time offsets (seconds from series start) for a window centred on one hour."""
        t_center = self.times[self.case_ids.index(center_case_id)]
        n = int(round(duration_s / dt_s)) + 1
        return t_center + np.arange(n, dtype=np.float64) * dt_s - 0.5 * duration_s
