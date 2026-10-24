#!/usr/bin/env python3
"""Redraw all results/les/2000s panels from existing caches (no VTP / vDBS loop)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd

from lesio import load_les_probes, load_urans_probes
from paths import CACHE, FIGS, OUT
from sampling import load_lidar, window
from run_p1 import fig_f01_real, fig_f07, fig_m1_heatmaps
from run_p2 import fig_f02, fig_f03_f08, fig_f05_routes, load_hourly_cfd_wrf
from run_p3 import main as p3_main


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    les = load_les_probes()
    ur = load_urans_probes()
    df = pd.read_csv(OUT / "m1_stationarity.csv")
    print("m1 / F01 / F07 …", flush=True)
    fig_m1_heatmaps(df)
    fig_f01_real(les, df)
    fig_f07(CACHE / "m2_front_x975_z100.npz")
    print("F02 / F03 / F08 / F05 …", flush=True)
    lidar_w = window(load_lidar())
    fig_f02(les, ur, lidar_w, load_hourly_cfd_wrf())
    fig_f03_f08(les, ur, lidar_w)
    fig_f05_routes()
    print("F04 …", flush=True)
    sys.path.insert(0, str(HERE / "figs"))
    from fig_f04_vdbs import main as f04_main
    f04_main()
    print("F06 …", flush=True)
    p3_main()
    print("wrote", FIGS)


if __name__ == "__main__":
    main()
