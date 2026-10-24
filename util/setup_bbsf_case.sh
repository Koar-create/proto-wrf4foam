#!/bin/bash
# 搭建 buoyantBoussinesqSimpleFoam 算例（热启动自参考 simpleFoam 第 5000 步）。
# 用法: bash util/setup_bbsf_case.sh <YYYYMMDD_HHMM> <HFX W/m2> <bbsf.nc 路径>
set -euo pipefail
TAG=$1; HFX=$2; NC=$3
ROOT=$HOME/WRF-OpenFOAM-Coupling
REF=$ROOT/steady_experiments_finer_ABL/${TAG}_two_boundaries_as_outlet
CASE=$ROOT/experiments/nonisothermal/${TAG}_bbsf
mkdir -p "$CASE"/{0,constant,system}
cd "$CASE"

# 网格与几何：软链接到参考算例所指向的原始位置
for n in polyMesh triSurface extendedFeatureEdgeMesh; do
  [ -e constant/$n ] || ln -s "$(readlink -f "$REF/constant/$n")" constant/$n
done
# 字典：拷自参考算例（不拷 transportProperties/physicalProperties/momentumTransport）
cp -n "$REF"/constant/turbulenceProperties "$REF"/constant/fvOptions constant/
cp -n "$REF"/system/{controlDict,fvSchemes,fvSolution,decomposeParDict} system/
# 0/ 模板（边界条件、$kInlet 宏），不要 p（BBSF 由 p_rgh 算出）
for f in U k epsilon nut; do cp -n "$REF/0/$f" 0/$f; done
touch myExpxx.foam

# 边界数据 + 热力场 + transportProperties + g
python3 "$ROOT/util/construct_OF_boundary_arrays_bbsf.py" --input-nc "$NC" --hfx "$HFX" --case-dir "$CASE"
# 内场热启动
python3 "$ROOT/util/init_bbsf_internal_fields.py" --case "$CASE" --ref-case "$REF" --input-nc "$NC"
