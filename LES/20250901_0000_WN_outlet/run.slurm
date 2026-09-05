#!/bin/bash
# 单节点 128 进程 / a128m512u（cpu3，EPYC 7763 双 NUMA）。
# 灵活修改 nodelist, 通过 `sinfo | grep a128m512u` 提前理解idle node 的编号。避免使用 cpu3-[3-10], 只用 cpu3-[11-17,20]。
# --exclusive：整机独占，避免未独占时邻居作业争用内存/IO 导致墙钟差数量级（与节点型号无关）。
# mpirun --bind-to core --map-by numa：128 核按 NUMA 绑定，各 cpu3 节点行为一致。
# 冒烟：pimpleFoam LES WALE，物理 5 s。独占一节点以免邻居作业拖垮 IO。
# 自适应 dt，但 writeControl=adjustableRunTime，输出落在整数秒。
# 100 s 生产：从 latestTime(=5) 续跑。节点仅 cpu3-[11-17,20]。
# 冒烟在 cpu3-11 约 113 s_wall/s_sim → 95 s 约 3.0 h，给 8 h 余量。
#SBATCH --account=root
#SBATCH --partition=a128m512u
#SBATCH --qos=a128m512u
#SBATCH --job-name=LES_100s
#SBATCH --output=LES_100s.%j
#SBATCH --nodelist=cpu3-[11-17,20]
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --time=8:00:00

source "$HOME/OpenFOAM/OpenFOAM-v2412/etc/bashrc"

case_path="${HOME}/WRF-OpenFOAM-Coupling/LES/20250901_0000_WN_outlet"
cd "${case_path}" || exit 1

LOG=log.pimpleFoam.100s
: > "${LOG}"

echo "===== Job started at $(date) =====" >> "${LOG}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}" >> "${LOG}"
echo "SLURM_NODELIST=${SLURM_NODELIST}" >> "${LOG}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-<unset>}" >> "${LOG}"
echo "pwd=$(pwd)" >> "${LOG}"
echo "application=pimpleFoam  endTime=$(foamDictionary -entry endTime -value system/controlDict)" >> "${LOG}"
echo "" >> "${LOG}"

echo "---- NUMA topology ----" >> "${LOG}"
numactl -H >> "${LOG}" 2>&1
echo "" >> "${LOG}"

echo "---- ulimit ----" >> "${LOG}"
ulimit -s unlimited 2>> "${LOG}" || true
ulimit -a >> "${LOG}"
echo "" >> "${LOG}"

NTASKS="${SLURM_NTASKS:-128}"
export OMP_NUM_THREADS=1

MPIRUN_COMMON=(
    -n "${NTASKS}"
    --bind-to core
    --map-by numa:PE=1
    --rank-by numa
)

if [ ! -d processor0 ]; then
    echo "---- decomposePar ----" >> "${LOG}"
    decomposePar -force 2>&1 | tee -a "${LOG}"
    echo "" >> "${LOG}"
else
    echo "---- processor0 exists, skip decomposePar ----" >> "${LOG}"
fi

echo "---- Starting pimpleFoam ----" >> "${LOG}"
echo "mpirun opts: ${MPIRUN_COMMON[*]}" >> "${LOG}"
mpirun "${MPIRUN_COMMON[@]}" pimpleFoam -parallel 2>&1 | tee -a "${LOG}"
echo "===== Job finished at $(date) =====" >> "${LOG}"
