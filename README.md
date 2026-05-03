# WRF–OpenFOAM coupling and urban CFD validation

This repository supports a **WRF-driven urban CFD workflow** (OpenFOAM RANS, including `simpleFoam` / `buoyFoam` variants), **LiDAR observation handling**, and a **standardized surrogate-training dataset** built from many steady-state cases. It is aimed at comparing mesoscale meteorology, building-resolving CFD, and tower-based remote sensing on a shared grid and time basis.

## The three roots that tie everything together

Most scripts assume a layout where **WRF preprocessing**, **OpenFOAM case banks**, and **written project science / ops** stay next to this repo (names are historical; adjust paths in code or via environment if yours differ).

### `steady_experiments_finer_ABL/` (OpenFOAM steady case bank)

This directory is the **primary OpenFOAM experiment tree** for the finer ABL steady RANS ensemble: one subdirectory per case (for example `20250901_0000_two_boundaries_as_outlet`, optional `-fvOpt_sensitivity_run` suffixes). Typical contents per case include:

- `constant/`, `system/`, time directories — standard OpenFOAM layout.
- `constant/boundaryData/` — inlet/outlet patches used by **`scripts/task3_extract_inflow.py`** and by WRF→CFD diagnostics.
- `processed_hdf5/<case_id>.h5` — produced by **`scripts/stage2_interpolate.py`** (after stage 1b export), then aggregated by **`scripts/task2_collect_fields.py`** into `surrogate_dataset/fields/`.
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
| **`scripts/`** | End-to-end helpers: WRF/LiDAR merge, ParaView `pvbatch` grid interpolation (stage 2), surrogate tasks 1–5, plotting, shell drivers for post-processing. |
| **`util/`** | WRF→Cartesian conversion and vertical interpolation, OpenFOAM `boundaryData` construction, diagnostics, residual plots, WRF–CFD comparison figures. |
| **`data/`** | Versioned *layout* for inputs and processed tables (raw WRF/CFD/LiDAR CSVs and merged products); large files are usually git-ignored. |
| **`analysis/`** | Notebooks and batch-specific analysis scripts (e.g. sensitivity studies). |
| **`results/`** | Regenerated figures (Hovmöller, Taylor diagrams, profiles, etc.). |
| **`surrogate_dataset/`** | Curated HDF5 fields, per-case inflow JSON, building geometry encoding, and `index.csv` for ML / surrogate model training. See `surrogate_dataset/README_产出清单.md` for the full manifest (Chinese). |

**Outside this table but central:** `steady_experiments_finer_ABL/` (OpenFOAM cases), `W_myExp03/` (WRF auxhist pipeline), and `docs/` (science + ops) — see [The three roots](#the-three-roots-that-tie-everything-together) above.

Large OpenFOAM trees, WRF outputs, and bulky artifacts (`.csv`, `.png`, `.h5`, entire `docs/` or `W_myExp*` in some setups, etc.) are often excluded from Git per `.gitignore`; you still need them on disk where the scripts expect them.

## Surrogate dataset pipeline (high level)

1. **Stage 1b** — Export a per-case point cloud from each case under **`steady_experiments_finer_ABL/<case_id>/`** (ParaView / `pvbatch`; see `scripts/stage1b_inspect_pvbatch.py` and your HPC notes).
2. **Stage 2** — `scripts/stage2_interpolate.py` cleans outliers and resamples `U` and `k` onto a regular 3-D grid; writes **`steady_experiments_finer_ABL/<case_id>/processed_hdf5/<case_id>.h5`** (default `--outdir` is relative to the case).
3. **Tasks 1–5** (run from repo root with Python 3):
   - **Task 1** — `scripts/task1_building_encoding.py`: STL → `surrogate_dataset/geometry/building_encoding_*.npy` (UDF + building mask).
   - **Task 2** — `scripts/task2_collect_fields.py`: collect valid per-case HDF5 files into `surrogate_dataset/fields/`.
   - **Task 3** — `scripts/task3_extract_inflow.py`: parse `constant/boundaryData` → `surrogate_dataset/inflow/<case_id>_inflow.json`.
   - **Task 4** — `scripts/task4_make_index.py`: build `surrogate_dataset/index.csv` and train/val/test splits.
   - **Task 5** — `scripts/task5_sample_space_map.py`: sample-space diagnostics figure.

Case IDs follow a timestamped naming convention (e.g. `YYYYMMDD_HHMM_two_boundaries_as_outlet`); sensitivity runs may append `-fvOpt_sensitivity_run` suffixes. Some cases are intentionally skipped (see scripts for ignore rules).

## WRF → CFD boundary and preprocessing

Utilities under `util/` implement steps such as:

- Hourly / rolling WRF auxhist processing (`process_auxhist2_hourly_avg.py`, `convert_wrf_coord_and_do_3D_interp.py`).
- Vertical interpolation to CFD levels (`do_vert_interp.py`).
- Construction of OpenFOAM inlet `boundaryData` from processed fields (`construct_OF_boundary_arrays.py`, `*_buoyFoam.py`, `*_with-top.py`).

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
