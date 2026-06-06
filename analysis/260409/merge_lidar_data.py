#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge WRF / CFD / LiDAR 1h-rolling tables into one CSV (260409 layout)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cfd-dir",
        type=Path,
        default=root / "data" / "260409" / "raw" / "cfd" / "control",
        help="Directory containing CFD_lidar_simulation_*.csv",
    )
    p.add_argument(
        "--wrf-csv",
        type=Path,
        default=root / "data" / "260409" / "raw" / "wrf" / "WRF_lidar_simulation_1h-rolling.csv",
        help="WRF 1h-rolling LiDAR-site CSV",
    )
    p.add_argument(
        "--lidar-csv",
        type=Path,
        default=root / "data" / "260409" / "raw" / "lidar" / "lidar_1h-rolling.csv",
        help="Observed LiDAR 1h-rolling CSV",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "260409" / "processed" / "merged_lidar_simulation_final.csv",
        help="Merged output CSV path",
    )
    return p.parse_args()


# 目标时次：CFD control 目前覆盖到 2025-09-05 23:00。
target_datetimes = pd.date_range("2025-09-01 00:00:00", "2025-09-05 23:00:00", freq="h")
target_times = target_datetimes.strftime("%Y-%m-%d %H:%M:%S").tolist()

cfd_files = [
    f"CFD_lidar_simulation_{dt.strftime('%Y%m%d_%H00')}_two_boundaries_as_outlet.csv"
    for dt in target_datetimes
]


def load_and_preprocess(
    cfd_dir: Path,
    wrf_csv: Path,
    lidar_csv: Path,
    output_file: Path,
) -> pd.DataFrame:
    print("Step 1: Loading WRF and Lidar data...")
    df_wrf = pd.read_csv(wrf_csv)
    df_lidar = pd.read_csv(lidar_csv)

    df_wrf["datetime"] = pd.to_datetime(df_wrf["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df_lidar["datetime"] = pd.to_datetime(df_lidar["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    df_wrf = df_wrf[df_wrf["datetime"].isin(target_times)].copy()
    df_lidar = df_lidar[df_lidar["datetime"].isin(target_times)].copy()

    print("Step 2: Loading and merging CFD data...")
    cfd_list = []
    for f in cfd_files:
        path = cfd_dir / f
        tmp = pd.read_csv(path)
        tmp["datetime"] = pd.to_datetime(tmp["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        cfd_list.append(tmp)
    df_cfd = pd.concat(cfd_list, ignore_index=True)

    print("Step 3: Aligning vertical layers by index...")
    for df in (df_wrf, df_cfd, df_lidar):
        h_col = "z_probe" if "z_probe" in df.columns else "Height"
        df.sort_values(by=["datetime", "obtid", h_col], inplace=True)
        df["layer_idx"] = df.groupby(["datetime", "obtid"]).cumcount()

    print("Step 4: Merging tables...")
    df_wrf = df_wrf.rename(
        columns={
            "U_wrf": "u_wrf",
            "V_wrf": "v_wrf",
            "WS_wrf": "ws_wrf",
            "z_probe": "Height",
        }
    )

    df_cfd_sub = df_cfd[
        ["datetime", "obtid", "layer_idx", "U_cfd", "V_cfd", "W_cfd"]
    ].rename(columns={"U_cfd": "u_cfd", "V_cfd": "v_cfd", "W_cfd": "w_cfd"})
    merged = pd.merge(df_wrf, df_cfd_sub, on=["datetime", "obtid", "layer_idx"], how="inner")

    df_lidar_sub = df_lidar[
        ["datetime", "obtid", "layer_idx", "U", "V", "WindSpd"]
    ].rename(columns={"U": "u_obs", "V": "v_obs", "WindSpd": "ws_obs"})
    merged = pd.merge(merged, df_lidar_sub, on=["datetime", "obtid", "layer_idx"], how="inner")

    merged["ws_cfd"] = (merged["u_cfd"] ** 2 + merged["v_cfd"] ** 2) ** 0.5

    final_columns = [
        "datetime",
        "obtid",
        "Height",
        "lon",
        "lat",
        "u_obs",
        "v_obs",
        "ws_obs",
        "u_wrf",
        "v_wrf",
        "ws_wrf",
        "u_cfd",
        "v_cfd",
        "w_cfd",
        "ws_cfd",
    ]
    final_df = merged[final_columns].copy()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Merge Complete! Final shape: {final_df.shape}")
    print(f"Saving to: {output_file}")
    final_df.to_csv(output_file, index=False)
    return final_df


def main() -> None:
    args = parse_args()
    final_data = load_and_preprocess(
        cfd_dir=args.cfd_dir,
        wrf_csv=args.wrf_csv,
        lidar_csv=args.lidar_csv,
        output_file=args.output,
    )
    print("Preview of first 5 rows:")
    print(final_data.head())


if __name__ == "__main__":
    main()
