# Non-isothermal steady (`buoyantBoussinesqSimpleFoam`)

Cases: `20250901_1000_bbsf`, `20250908_0000_bbsf`.

## Layout

| Path | Role |
|------|------|
| `scripts/` | `run.slurm`, `postprocess.sh`, plotting |
| `util/case_paths.py` | Tags, merged CSV paths, `results/nonisothermal/` |
| `<tag>_bbsf/` | OpenFOAM case (`logs/`, `postProcessing/`, `check/`) |
| `results/nonisothermal/` (repo root) | Cross-case comparison figures |

Setup: `util/setup_bbsf_case.sh <YYYYMMDD_HHMM> <HFX> <bbsf.nc>`.

## Submit

```bash
sbatch experiments/nonisothermal/scripts/run.slurm 20250901_1000
sbatch experiments/nonisothermal/scripts/run.slurm 20250908_0000
```

## Post-process & plots

```bash
bash experiments/nonisothermal/scripts/postprocess.sh 20250901_1000
python experiments/nonisothermal/scripts/visualize_xz_wrf_simpleFoam_bbsf.py 20250901_1000
python experiments/nonisothermal/scripts/plot_ws_composite_stations_simpleFoam_bbsf.py 20250901_1000
```

Comparison PNGs default to `results/nonisothermal/`. LiDAR merge CSV stays in `<case>/postProcessing/`.

## Physics checks

```bash
python util/check_bbsf_physics.py experiments/nonisothermal/20250901_1000_bbsf --log logs/log.buoyantBoussinesqSimpleFoam.stageB
python util/check_bbsf_profiles.py experiments/nonisothermal/20250901_1000_bbsf --time 6000
```
