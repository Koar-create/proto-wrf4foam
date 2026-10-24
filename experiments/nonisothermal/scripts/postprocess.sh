#!/bin/bash
# BBSF 后处理（对齐 scripts/postprocessing.sh，按本求解器改过）。
# 用法: bash experiments/nonisothermal/scripts/postprocess.sh <YYYYMMDD_HHMM>
#   可选环境变量: LOG  RECON_TIME
# 不用 set -u：OpenFOAM bashrc 会引用未绑定变量。

PROJECT_ROOT=${PROJECT_ROOT:-$HOME/WRF-OpenFOAM-Coupling}
UTIL=$PROJECT_ROOT/util
NONISO=$PROJECT_ROOT/experiments/nonisothermal
RESULTS=$PROJECT_ROOT/results/nonisothermal
mkdir -p "$RESULTS"

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <YYYYMMDD_HHMM>   # 20250901_1000 or 20250908_0000"
    exit 1
fi
TAG=$1
TAG=${TAG%_bbsf}
EXPE=$NONISO/${TAG}_bbsf
REL=experiments/nonisothermal/${TAG}_bbsf

CONDA_PATH=$HOME/miniconda3
source "$CONDA_PATH/etc/profile.d/conda.sh"
source "$HOME/OpenFOAM/OpenFOAM-v2412/etc/bashrc"
set -eo pipefail

cd "$EXPE" || { echo "Directory $EXPE not found"; exit 1; }

if [ -z "${LOG:-}" ]; then
    for cand in \
        logs/log.buoyantBoussinesqSimpleFoam.stageB \
        log.buoyantBoussinesqSimpleFoam.stageB \
        logs/log.buoyantBoussinesqSimpleFoam.stageA \
        log.buoyantBoussinesqSimpleFoam.stageA \
        logs/log.buoyantBoussinesqSimpleFoam \
        log.buoyantBoussinesqSimpleFoam; do
        if [ -f "$cand" ] && grep -q "Finalising parallel run" "$cand"; then
            LOG=$cand
            break
        fi
    done
    if [ -z "${LOG:-}" ]; then
        LOG=logs/log.buoyantBoussinesqSimpleFoam
    fi
fi

if [ -z "${RECON_TIME:-}" ]; then
    if [ -f 6000/U ]; then
        RECON_TIME=6000
    elif [ -f 3000/U ]; then
        RECON_TIME=3000
    else
        RECON_TIME=6000
    fi
fi

if [ ! -f myExpxx.foam ]; then
    echo -e "\e[1;31mmyExpxx.foam not exist, create one.\e[0m"
    touch myExpxx.foam
fi

if ! grep -q "Finalising parallel run" "$LOG"; then
    echo -e "\e[1;34mexperiment crashed or smth else.\e[0m"
    exit 1
fi

if [ -f "${RECON_TIME}/U" ]; then
    echo -e "\e[1;33mreconstructed ${RECON_TIME}/U already exists, skip reconstructPar.\e[0m"
else
    reconstructPar -time "$RECON_TIME"
fi

foamLog "$LOG"

# check_scientific_usability_diag.sh 认 p_0；本求解器是 p_rgh
if [ -f logs/p_rgh_0 ] && [ ! -e logs/p_0 ]; then
    ln -s p_rgh_0 logs/p_0
fi

python "$UTIL/plot_residuals_buoy_steady.py" "$EXPE/logs"

export QT_QPA_PLATFORM=offscreen
conda activate paraview-env
pvbatch "$UTIL/export_z_slice_as_csv.py" myExpxx.foam 100
pvbatch "$UTIL/export_y_slice_as_csv.py" myExpxx.foam 800
conda deactivate

cd "$PROJECT_ROOT"
python "$UTIL/visualize_x-y_wind_field.py" "$REL" \
    --output "$RESULTS/wind_field_${TAG}_bbsf.png"
python "$PROJECT_ROOT/analysis/260409/ws-xz-plane/visualize_WRF_CFD_xz_two_panel.py" "$REL" \
    --layout horizon \
    --output "$RESULTS/xz_wrf_cfd_${TAG}_bbsf.png"
bash "$UTIL/check_scientific_usability_diag.sh" "$EXPE/${LOG}"

echo -e "\e[1;32mAll procedures done.\e[0m"
echo -e "\e[1;90mNothing else to do, exit.\e[0m"
