#!/bin/bash
PROJECT_ROOT=$HOME/WRF-OpenFOAM-Coupling
UTIL=$PROJECT_ROOT/util

# 1. initialize Conda and OpenFOAM environment variable
CONDA_PATH=$HOME/miniconda3
source $CONDA_PATH/etc/profile.d/conda.sh
source $HOME/OpenFOAM/OpenFOAM-v2412/etc/bashrc

# check if parameter provided
if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_path>"
    exit 1
fi

# use input parameter as EXPE
EXPE=$PROJECT_ROOT/$1

cd $EXPE || { echo "Directory $EXPE not found"; exit 1; }

if [ ! -f myExpxx.foam ]; then
    echo -e "\e[1;31mmyExpxx.foam not exist, create one.\e[0m"
    touch myExpxx.foam
fi

LOG=log.simpleFoam
# LOG=log.buoyFoam
if tail -n 5 ${LOG} | grep -q "Finalising parallel run"; then
    reconstructPar -time 5000
    foamLog ${LOG}
    python $UTIL/plot_residuals_steady.py $EXPE/logs
    conda activate paraview-env
    # pvbatch $UTIL/export_z_slice_as_csv.py myExpxx.foam 10
    pvbatch   $UTIL/export_z_slice_as_csv.py myExpxx.foam 100
    pvbatch   $UTIL/export_y_slice_as_csv.py myExpxx.foam 800
    conda deactivate
    cd $PROJECT_ROOT
    python $UTIL/visualize_x-y_wind_field.py $1
    python $UTIL/visualize_x-z_wind_field.py $1
    bash $PROJECT_ROOT/check_scientific_usability_diag.sh $EXPE/${LOG}
    echo -e "\e[1;32mAll procedures done.\e[0m"
else
    echo -e "\e[1;34mexperiment crashed or smth else.\e[0m"
    exit 1
fi

echo -e "\e[1;90mNothing else to do, exit.\e[0m"
