#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge WRF / CFD / LiDAR 1h-rolling tables into one CSV (260707 layout)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import campaign_config as cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cfd-dir",
        type=Path,
        default=cfg.DATA_DIR / "raw" / "cfd" / "control",
        help="Directory containing CFD_lidar_simulation_*.csv",
    )
    p.add_argument(
        "--wrf-csv",
        type=Path,
        default=cfg.DATA_DIR / "raw" / "wrf" / "WRF_lidar_simulation_1h-rolling.csv",
        help="WRF 1h-rolling LiDAR-site CSV",
    )
    p.add_argument(
        "--lidar-csv",
        type=Path,
        default=cfg.DATA_DIR / "raw" / "lidar" / "lidar_1h-rolling.csv",
        help="Observed LiDAR 1h-rolling CSV",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=cfg.DATA_PATH,
        help="Merged output CSV path",
    )
    return p.parse_args()


# 目标时次：20250906–13，UTC 00/06/12/18（共 32 个实验；缺失 CFD 文件会跳过并告警）。
target_datetimes = cfg.METRIC_DATETIMES
target_times = target_datetimes.strftime("%Y-%m-%d %H:%M:%S").tolist()

cfd_files = [cfg.cfd_filename(dt) for dt in target_datetimes]

EXCLUDED_OBTIDS = {"GAW105"}


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
    missing = []
    for f in cfd_files:
        path = cfd_dir / f
        if not path.is_file():
            missing.append(f)
            continue
        tmp = pd.read_csv(path)
        tmp["datetime"] = pd.to_datetime(tmp["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        cfd_list.append(tmp)
    if missing:
        print(f"[WARN] Skipped {len(missing)} missing CFD file(s), e.g. {missing[0]}")
    if not cfd_list:
        raise FileNotFoundError(f"No CFD files found under {cfd_dir}")
    df_cfd = pd.concat(cfd_list, ignore_index=True)

    for df in (df_wrf, df_lidar, df_cfd):
        df.drop(df[df["obtid"].isin(EXCLUDED_OBTIDS)].index, inplace=True)

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
            "eps_wrf": "epsilon_wrf",
        }
    )

    df_cfd_sub = df_cfd[
        ["datetime", "obtid", "layer_idx", "U_cfd", "V_cfd", "W_cfd", "k_cfd", "eps_cfd"]
    ].rename(columns={"U_cfd": "u_cfd", "V_cfd": "v_cfd", "W_cfd": "w_cfd", "eps_cfd": "epsilon_cfd"})
    merged = pd.merge(df_wrf, df_cfd_sub, on=["datetime", "obtid", "layer_idx"], how="inner")

    df_lidar_sub = df_lidar[
        ["datetime", "obtid", "layer_idx", "U", "V", "WindSpd", "k", "epsilon"]
    ].rename(columns={"U": "u_obs", "V": "v_obs", "WindSpd": "ws_obs", "k": "k_obs", "epsilon": "epsilon_obs"})
    merged = pd.merge(merged, df_lidar_sub, on=["datetime", "obtid", "layer_idx"], how="inner")

    merged["ws_cfd"] = (merged["u_cfd"] ** 2 + merged["v_cfd"] ** 2) ** 0.5

    # 气象风向：风的来向，顺时针从正北起算
    # wd = (270 - atan2d(V, U)) % 360
    merged["wd_obs"] = (270 - np.degrees(np.arctan2(merged["v_obs"], merged["u_obs"]))) % 360
    merged["wd_wrf"] = (270 - np.degrees(np.arctan2(merged["v_wrf"], merged["u_wrf"]))) % 360
    merged["wd_cfd"] = (270 - np.degrees(np.arctan2(merged["v_cfd"], merged["u_cfd"]))) % 360

    final_columns = [
        "datetime",
        "obtid",
        "Height",
        "lon",
        "lat",
        "u_obs",
        "v_obs",
        "ws_obs",
        "wd_obs",
        "k_obs",
        "epsilon_obs",
        "u_wrf",
        "v_wrf",
        "ws_wrf",
        "wd_wrf",
        "k_wrf",
        "epsilon_wrf",
        "u_cfd",
        "v_cfd",
        "w_cfd",
        "ws_cfd",
        "wd_cfd",
        "k_cfd",
        "epsilon_cfd",
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
