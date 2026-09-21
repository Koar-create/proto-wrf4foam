"""Quantify how often WRF underestimates the low-layer (52-300 m) hourly mean
wind speed, and how often the coupled CFD run makes that underestimation worse.

Also reports the single clearest failure case (2025-09-02 17:00 LST) used in
Section 4 of the manuscript.

Data: data/260409/processed/merged_lidar_simulation_final.csv (1-5 Sep 2025 baseline).
Usage: python analysis/260409/print_underestimation_failure_stats.py
"""
import pandas as pd

DATA_PATH = "data/260409/processed/merged_lidar_simulation_final.csv"
LOW_MIN_M, LOW_MAX_M = 52, 300
LST_OFFSET_HOURS = 8


def main():
    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])

    low = df[(df["Height"] >= LOW_MIN_M) & (df["Height"] <= LOW_MAX_M)]
    hourly = low.groupby("datetime")[["ws_obs", "ws_wrf", "ws_cfd"]].mean()
    hourly["wrf_bias"] = hourly["ws_wrf"] - hourly["ws_obs"]
    hourly["cfd_bias"] = hourly["ws_cfd"] - hourly["ws_obs"]
    hourly["lst"] = hourly.index + pd.Timedelta(hours=LST_OFFSET_HOURS)

    n_total = len(hourly)
    under = hourly[hourly["wrf_bias"] < 0]
    n_under = len(under)
    worsened = under[under["cfd_bias"] < under["wrf_bias"]]
    n_worsened = len(worsened)

    print(f"Total hourly samples (low-layer, station-mean): {n_total}")
    print(f"Hours where WRF underestimates low-layer mean wind speed: {n_under}")
    print(
        f"Of those, hours where CFD bias is more negative than WRF bias "
        f"(coupling worsens underestimation): {n_worsened}"
    )
    print()

    target_utc = pd.Timestamp("2025-09-02 09:00:00")
    if target_utc in hourly.index:
        row = hourly.loc[target_utc]
        print("2025-09-02 09:00 UTC = 17:00 LST case (low-layer station mean):")
        print(
            f"  obs={row['ws_obs']:.3f} m/s, wrf={row['ws_wrf']:.3f} m/s, "
            f"cfd={row['ws_cfd']:.3f} m/s, wrf_bias={row['wrf_bias']:.3f}, "
            f"cfd_bias={row['cfd_bias']:.3f}"
        )
    else:
        print("Target hour 2025-09-02 09:00 UTC not found in dataset.")


if __name__ == "__main__":
    main()
