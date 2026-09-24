#!/bin/bash
# Resample missing 30/60/120 m slices. One ParaView read per case (~32 s, ~2 GB).
# Four cases at a time stays within the login node's free memory.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PVBATCH="${HOME}/miniconda3/envs/paraview-env/bin/pvbatch"
SCRIPT="${REPO}/util/export_xy_slices_pvbatch.py"
LIST="${REPO}/results/wrf_openfoam/local_wind_ratio/case_list.txt"
LOGDIR="${REPO}/results/wrf_openfoam/local_wind_ratio/resample_logs"
mkdir -p "${LOGDIR}"

if [[ ! -s "${LIST}" ]]; then
  echo "No pending cases in ${LIST}"
  exit 0
fi

export PVBATCH SCRIPT LOGDIR
# Each line is: foam_path z [z ...]
xargs -P 4 -L 1 -a "${LIST}" bash -c '
  foam="$1"
  shift
  case_name="$(basename "$(dirname "$foam")")"
  echo "START ${case_name} $(date +%H:%M:%S)"
  "$PVBATCH" "$SCRIPT" "$foam" "$@" > "${LOGDIR}/${case_name}.log" 2>/dev/null
  echo "DONE  ${case_name} $(date +%H:%M:%S)"
' _
echo "Resample batch finished"
