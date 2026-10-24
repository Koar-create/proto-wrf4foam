# URANS (`pimpleFoam`)

OpenFOAM v2412 transient RANS cases: `20250901_1000_pimplefoam`, `20250908_0000_pimplefoam`.

## Layout

| Path | Role |
|------|------|
| `scripts/` | `run.slurm`, `reconstruct.slurm`, plotting |
| `util/case_paths.py` | Resolve case dirs and `results/urans/` outputs |
| `util/check_urans_physics.py` | Post-run log checks |
| `<tag>_pimplefoam/` | OpenFOAM case (`logs/`, `postProcessing/`, `check/`) |
| `results/urans/` (repo root) | Cross-case comparison PNG/CSV |

Scaffold new cases: `util/setup_urans_pimplefoam_case.py` + `util/build_urans_boundary_series.py`.

## Submit

```bash
sbatch experiments/urans/scripts/run.slurm 20250901_1000
sbatch experiments/urans/scripts/run.slurm 20250908_0000
# smoke: sbatch --export=ALL,ENDTIME=60,CLEAN=1,TAG=smoke,NTASKS=64 ...
sbatch experiments/urans/scripts/reconstruct.slurm 20250901_1000
```

## Plots

```bash
python experiments/urans/scripts/plot_xy_z100_t1800.py
python experiments/urans/scripts/plot_lidar_ws_timeseries.py
python experiments/urans/scripts/visualize_xz_wrf_simpleFoam_urans.py 20250901_1000
```

Defaults write to `results/urans/`.

## Check

```bash
python experiments/urans/util/check_urans_physics.py experiments/urans/20250901_1000_pimplefoam --log logs/log.pimpleFoam.full
```
