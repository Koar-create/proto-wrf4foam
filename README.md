# WRF–OpenFOAM coupling and urban CFD validation

This repository supports a **WRF-driven urban CFD workflow** (OpenFOAM RANS, including `simpleFoam` and non-isothermal solvers) and **LiDAR observation handling**. It is aimed at comparing mesoscale meteorology, building-resolving CFD, and tower-based remote sensing on a shared grid and time basis.

## The three roots that tie everything together

Most scripts assume a layout where **WRF preprocessing**, **OpenFOAM case banks**, and **written project science / ops** stay next to this repo (names are historical; adjust paths in code or via environment if yours differ).

### `steady_experiments_finer_ABL/` (OpenFOAM steady case bank)

This directory is the **primary OpenFOAM experiment tree** for the finer ABL steady RANS ensemble: one subdirectory per case (for example `20250901_0000_two_boundaries_as_outlet`, optional `-fvOpt_sensitivity_run` suffixes). Typical contents per case include:

- `constant/`, `system/`, time directories — standard OpenFOAM layout.
- `constant/boundaryData/` — inlet/outlet patches used by **`scripts/task3_extract_inflow.py`** and by WRF→CFD diagnostics.
- `postProcessing/` — sampled line/slice CSVs consumed by utilities such as `util/visualize_x-y_wind_field.py` and `util/calculate_csv_mean_diag.py`.

At the **batch** level, `steady_experiments_finer_ABL/WRF Atmospheric Stability Data Organization.csv` documents stability / LLJ metadata and feeds **`util/plot_wrf_stability_organization_csv.py`** (and related methodology text under `docs/`). A copy also exists under `docs/` for workflows that keep large case trees outside Git.

### `W_myExp03/` (WRF sidecar experiment folder)

`W_myExp03` is the **WRF “experiment 03” workspace** used as the mesoscale input bridge: many `util/*.py` scripts hard-code or default to paths like `$HOME/WRF-OpenFOAM-Coupling/W_myExp03` (Linux) or repo-relative `W_myExp03/...` (Windows in some tools). Under it you normally maintain, among others:

- `auxhist2/` — processed WRF auxhist grids (e.g. `tmp/*.nc`, rolling Cartesian NetCDF, horizontal interpolation products) that **`convert_wrf_coord_and_do_3D_interp.py`**, **`do_vert_interp.py`**, **`process_auxhist2_hourly_avg.py`**, and **`construct_OF_boundary_arrays*.py`** read before writing OpenFOAM `boundaryData`.

If your checkout lives elsewhere, **edit the `project_path` / `WRF_ROOT` blocks** at the top of those utilities or symlink `W_myExp03` into the expected location.

### `docs/` (constraints, methodology, operations)

`docs/` holds **human-written project material** that is not implied by code alone:

- **`docs/project/Global_Constraints.md`** — global assumptions and constraints for the study.
- **`docs/methodology/`** — ABL stability / LLJ detection write-ups (Chinese and English variants; some marked deprecated).
- **`docs/ops/`** — operator notes, SVG roadmaps, staged instructions (`待使用的指令*.md`).
- **`docs/reference-candidate/`** — reference PDFs, JSON catalogs, and small analysis scripts for literature you compare against (e.g. coupling methodology).

These paths are listed in **`.gitignore`** in some clones so they never leave your machine; keep them in your working tree anyway if you rely on the documented workflow.

## What is in this repo

| Area | Role |
|------|------|
| **`scripts/`** | End-to-end helpers: WRF/LiDAR merge, plotting, shell drivers for post-processing. |
| **`util/`** | WRF→Cartesian conversion and vertical interpolation, OpenFOAM `boundaryData` construction, diagnostics, residual plots, WRF–CFD comparison figures. |
| **`data/`** | Versioned *layout* for inputs and processed tables (raw WRF/CFD/LiDAR CSVs and merged products); large files are usually git-ignored. |
| **`analysis/`** | Notebooks and batch-specific analysis scripts (e.g. sensitivity studies). |
| **`results/`** | Regenerated figures (Hovmöller, Taylor diagrams, profiles, etc.). Family comparison plots for URANS/BBSF/LES live under `results/urans/`, `results/nonisothermal/`, and `results/les/2000s/`. |

**Outside this table but central:** `steady_experiments_finer_ABL/` (OpenFOAM simpleFoam case bank), `experiments/` (extra cases: `nonisothermal/*_bbsf`, `urans/*_pimplefoam`, sensitivity, …), `LES/` (large-eddy cases under `LES/experiments/`, analysis under `LES/analysis/2000s/`, ops scripts under `LES/scripts/`), `W_myExp03/`/`W_myExp05/` (WRF auxhist pipeline), and `docs/` (science + ops) — see [The three roots](#the-three-roots-that-tie-everything-together) above.

Each CFD family uses the same support layout: `scripts/` (run/postprocess/plot), `util/case_paths.py`, OpenFOAM cases at the family root, solver logs under `<case>/logs/`, samples in `<case>/postProcessing/`, and cross-case figures in `results/<family>/` (see `experiments/urans/README.md`, `experiments/nonisothermal/README.md`, and `LES/docs/README`).

Large OpenFOAM trees, WRF outputs, and bulky artifacts (`.csv`, `.png`, `.h5`, entire `docs/` or `W_myExp*` in some setups, etc.) are often excluded from Git per `.gitignore`; you still need them on disk where the scripts expect them.

## WRF → CFD boundary and preprocessing

Utilities under `util/` implement steps such as:

- Hourly / rolling WRF auxhist processing (`process_auxhist2_hourly_avg.py`, `convert_wrf_coord_and_do_3D_interp.py`).
- Vertical interpolation to CFD levels (`do_vert_interp.py`).
- Construction of OpenFOAM inlet `boundaryData` from processed fields (`construct_OF_boundary_arrays.py`, `*_bbsf.py`, `*_with-top.py`).

Boundary and WRF utilities overwhelmingly assume a **`W_myExp03`** tree under the same parent as this repo (often `$HOME/WRF-OpenFOAM-Coupling/W_myExp03` on Linux). **Set paths explicitly** or edit the `project_path` / `WRF_ROOT` header in each script to match your machine. For platform-oriented notes, see `util/platform_classification.json` and `scripts/platform_classification.json`.

## Observations and validation

- **`scripts/merge_lidar_data.py`** — Merges WRF, CFD, and LiDAR 1 h rolling tables into one CSV (defaults under `data/260409/`; override with `--cfd-dir`, `--wrf-csv`, `--lidar-csv`, `--output`).
- **`scripts/plot_hovmoller_lidar_wrf_cfd.py`** — Hovmöller-style comparison plots; pass explicit `--csv` / `--out` after any data re-organization (see project skill under `.cursor/skills/` for canonical paths).

## Dependencies

There is no single `requirements.txt`; needs depend on the script:

- **General Python**: `numpy`, `pandas`, `xarray`, `matplotlib`, `h5py`, `scipy` (as used by each module).
- **Stage 2 / ParaView**: ParaView Python (`pvbatch`), `scipy`, `h5py`.
- **OpenFOAM**: matching version for your case setup; boundary utilities expect native OpenFOAM directory layout.

Use `conda`/`venv` and install packages as you hit import errors, or freeze a minimal environment once you know which sub-pipeline you run.

## Layout reference for moved data

After reorganizing batches under `data/`, `analysis/`, and `results/`, prefer **explicit CLI paths** over hard-coded legacy folders. A concise path map lives in:

`.cursor/skills/project-layout-data-results-analysis/SKILL.md`

## License

No project-wide license file is provided in-tree; add one if you redistribute this work.
