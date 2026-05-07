---
name: project-layout-data-results-analysis
description: Documents the current repo layout for moved datasets and outputs under data/, analysis/, and results/. Use when the user asks where files live, how to reference moved CSV/PNG/IPYNB paths, or how to run plotting/merge scripts after a data re-organization.
---

# Repo layout: `data/` + `analysis/` + `results/`

This repository organizes **inputs**, **notebooks**, and **generated figures** into three top-level directories:

- `data/`: datasets (raw + processed tables)
- `analysis/`: notebooks and one-off analysis scripts
- `results/`: generated figures (can be regenerated)

## Current locations (authoritative)

### `data/`

- `data/260409/processed/merged_lidar_simulation_final.csv`
- `data/260409/raw/lidar/lidar_1h-rolling.csv`
- `data/260409/raw/wrf/WRF_lidar_simulation_1h-rolling.csv`
- `data/260409/raw/cfd/control/CFD_lidar_simulation_<YYYYMMDD>_<HHMM>_two_boundaries_as_outlet.csv`
- `data/260409/raw/cfd/sensitivity/CFD_lidar_simulation_<YYYYMMDD>_<HHMM>_two_boundaries_as_outlet-fvOpt_sensitivity_run.csv`

- `data/260413/processed/merged_lidar_simulation_final_nighttime_only.csv`
- `data/260413/processed/merged_lidar_simulation_final_Sp_magn_test.csv`

### `analysis/`

- `analysis/260409/validation_analysis_*.ipynb`
- `analysis/260413-sensitivity/validation_analysis*.ipynb`
- `analysis/260409/merge_lidar_data.py`
- `analysis/260413-sensitivity/merge_lidar_data.py`

### `results/`

Hovmöller PNGs are stored here (by batch):

- `results/hovmoller/260409/hovmoller_ws_obs_wrf_cfd_<obtid>_y<ymax>.png`
- `results/hovmoller/260413-sensitivity/hovmoller_ws_obs_wrf_cfd_ref_sen_<obtid>_y<ymax>.png`

Other figure families follow the same convention:

- `results/taylor_diagram/<batch>/`
- `results/ws_composite_profile/<batch>/`
- `results/ws_station_profile/<batch>/`

## Using scripts after the move (important)

Some scripts still have **legacy defaults** pointing to the old `260409/` and `260413-sensitivity_run_analysis/` folders.

When in doubt, always pass explicit paths:

- Standard merged CSV: `--csv data/260409/processed/merged_lidar_simulation_final.csv`
- Sensitivity merged CSV: `--csv data/260413/processed/merged_lidar_simulation_final_nighttime_only.csv`
- Output PNG: `--out results/hovmoller/<batch>/<filename>.png`

### Examples (Hovmöller)

- Standard (3 panels):
  - `python scripts/plot_hovmoller_lidar_wrf_cfd.py --csv data/260409/processed/merged_lidar_simulation_final.csv --obtid GAW111 --ymax 2000 --out results/hovmoller/260409/hovmoller_ws_obs_wrf_cfd_GAW111_y2000.png`

- Sensitivity (4 panels):
  - `python scripts/plot_hovmoller_lidar_wrf_cfd.py --sensitivity --csv data/260413/processed/merged_lidar_simulation_final_nighttime_only.csv --obtid GAW103 --ymax 2000 --out results/hovmoller/260413-sensitivity/hovmoller_ws_obs_wrf_cfd_ref_sen_GAW103_y2000.png`

