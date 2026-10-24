"""M0.4 sampling-match operator: replay LiDAR timestamps on LES probes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from paths import CACHE, LES_WIN_A, LIDAR_DBS, OBS_WINDOW, STATIONS


def load_lidar(force: bool = False) -> pd.DataFrame:
    dest = CACHE / "lidar_dbs.pkl"
    if dest.is_file() and not force:
        return pd.read_pickle(dest)
    d = pd.read_csv(LIDAR_DBS, parse_dates=["datetime"])
    d = d.dropna(subset=["U", "V", "WindSpd"])
    d = d[(d.WindSpd < 15.0) & (d.VerticalSpd.abs() < 5.0)]
    h0 = {k: v["h0"] for k, v in STATIONS.items()}
    d["zAGL"] = d.Height + d.obtid.map(h0)
    d = d.sort_values(["obtid", "datetime", "Height"]).reset_index(drop=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    d.to_pickle(dest)
    return d


def window(d: pd.DataFrame, t0=OBS_WINDOW[0], t1=OBS_WINDOW[1]) -> pd.DataFrame:
    return d[(d.datetime >= t0) & (d.datetime < t1)].copy()


def longitudinal(u, v):
    um, vm = np.nanmean(u), np.nanmean(v)
    m = np.hypot(um, vm)
    if m < 1e-9:
        return u - um, v - vm
    return ((u - um) * um + (v - vm) * vm) / m, (-(u - um) * vm + (v - vm) * um) / m


def station_gate(d: pd.DataFrame, station: str, height: float) -> pd.DataFrame:
    g = d[(d.obtid == station) & (np.isclose(d.Height, height))]
    return g.sort_values("datetime")


def timestamps_rel_s(g: pd.DataFrame, t0=OBS_WINDOW[0]) -> np.ndarray:
    t0 = pd.Timestamp(t0)
    return (g.datetime - t0).dt.total_seconds().to_numpy()


def sample_match_once(t_les, u_les, t_obs_rel, phi, T_win=600.0, t_les0=None, scan_s=0.0):
    """Map a 600 s lidar timestamp fragment onto a LES window.

    t_obs_rel : seconds from the observation-window start (length of 1 h).
    phi       : start of the 600 s fragment within the 1 h observation window.
    t_les0    : LES window start (default LES_WIN_A[0]).
    """
    if t_les0 is None:
        t_les0 = LES_WIN_A[0]
    sel = (t_obs_rel >= phi) & (t_obs_rel < phi + T_win)
    tau = t_obs_rel[sel] - phi
    t_q = t_les0 + tau
    u = np.asarray(u_les, float)
    if scan_s and scan_s > 0:
        dt = float(np.median(np.diff(t_les)))
        k = max(int(round(scan_s / dt)), 1)
        if k > 1:
            ker = np.ones(k) / k
            u = np.convolve(u, ker, mode="same")
    return t_q, np.interp(t_q, t_les, u, left=np.nan, right=np.nan)


def sample_match_mc(
    t_les, u_les, t_obs_rel,
    n=200, T_win=600.0, t_les0=None, t_les1=None,
    scan_s=0.0, rng=None, min_n=20,
):
    """200 Monte-Carlo placements of a 600 s lidar fragment on the LES window.

    Returns list of (t_q, u_q) realisations that have at least min_n samples.
    """
    rng = np.random.default_rng(rng)
    if t_les0 is None:
        t_les0 = LES_WIN_A[0]
    if t_les1 is None:
        t_les1 = LES_WIN_A[1]
    T_les = t_les1 - t_les0
    T_win = min(T_win, T_les)
    t_span = float(np.nanmax(t_obs_rel) - np.nanmin(t_obs_rel))
    phi_max = max(t_span - T_win, 0.0)
    out = []
    for _ in range(n):
        phi = float(rng.uniform(0.0, phi_max)) if phi_max > 0 else 0.0
        t_q, u_q = sample_match_once(
            t_les, u_les, t_obs_rel, phi, T_win=T_win, t_les0=t_les0, scan_s=scan_s,
        )
        if np.isfinite(u_q).sum() >= min_n:
            out.append((t_q, u_q))
    return out


def mc_stat_distribution(realisations, stat_fn):
    """Apply stat_fn(u_q) to each realisation; return median and 5–95 %."""
    vals = []
    for _, u_q in realisations:
        v = stat_fn(u_q)
        if np.isfinite(v):
            vals.append(float(v))
    vals = np.asarray(vals, float)
    if vals.size == 0:
        return dict(median=np.nan, p05=np.nan, p95=np.nan, n=0)
    return dict(
        median=float(np.median(vals)),
        p05=float(np.percentile(vals, 5)),
        p95=float(np.percentile(vals, 95)),
        n=int(vals.size),
    )
