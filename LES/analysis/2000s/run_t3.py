#!/usr/bin/env python3
"""T3: bias / RMSE / VDI-style hit rate of window means vs LiDAR (B-grade)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from lesio import load_les_probes, load_urans_probes
from paths import LES_WIN_A, OUT, PROBE_Z, STATIONS, URANS_WIN
from run_p2 import _win_stats, load_hourly_cfd_wrf, obs_profile
from sampling import load_lidar, window


def hit_rate(pred, obs, D=0.25, W=0.25):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    ok = np.isfinite(pred) & np.isfinite(obs)
    if not ok.any():
        return np.nan, 0
    p, o = pred[ok], obs[ok]
    hit = (np.abs(p - o) / np.maximum(np.abs(o), 1e-6) <= D) | (np.abs(p - o) <= W)
    return float(hit.mean()), int(ok.sum())


def metrics(pred, obs):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    ok = np.isfinite(pred) & np.isfinite(obs)
    if ok.sum() < 2:
        return dict(bias=np.nan, rmse=np.nan, n=int(ok.sum()), hit=np.nan)
    d = pred[ok] - obs[ok]
    hr, n = hit_rate(pred, obs)
    return dict(bias=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())), n=n, hit=hr)


def main():
    les = load_les_probes()
    ur = load_urans_probes()
    lidar_w = window(load_lidar())
    hourly = load_hourly_cfd_wrf()
    stL = _win_stats(les["t"], les["U"], *LES_WIN_A)
    stU = _win_stats(ur["t"], ur["U"], *URANS_WIN)
    z_src = np.asarray(PROBE_Z, float)
    rows = []
    for name in STATIONS:
        o = obs_profile(lidar_w, name)
        i0 = STATIONS[name]["i0"]
        les_s = np.interp(o.z, z_src, stL["ws_s"][i0:i0 + 11], left=np.nan, right=np.nan)
        ur_s = np.interp(o.z, z_src, stU["ws_s"][i0:i0 + 11], left=np.nan, right=np.nan)
        for layer, m in (("z<=300", o.z <= 300), (">300", o.z > 300), ("60-600", np.ones(len(o), bool))):
            for model, pred in (("LES_1400-2000", les_s), ("URANS_1200-1800", ur_s)):
                met = metrics(pred[m], o.S.to_numpy()[m])
                rows.append(dict(station=name, layer=layer, model=model, **met))
            if hourly is not None and "ws_cfd" in hourly.columns:
                hh = hourly[hourly.obtid == name] if "obtid" in hourly.columns else hourly
                if len(hh) and "Height" in hh.columns:
                    zobs = hh.Height + STATIONS[name]["h0"]
                    # interpolate hourly onto lidar profile z
                    rans = np.interp(o.z, zobs, hh.ws_cfd, left=np.nan, right=np.nan)
                    met = metrics(rans[m], o.S.to_numpy()[m])
                    rows.append(dict(station=name, layer=layer, model="RANS_hourly", **met))
                    if "ws_wrf" in hh.columns:
                        wrf = np.interp(o.z, zobs, hh.ws_wrf, left=np.nan, right=np.nan)
                        met = metrics(wrf[m], o.S.to_numpy()[m])
                        rows.append(dict(station=name, layer=layer, model="WRF_hourly", **met))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "T3_hit_rate.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
