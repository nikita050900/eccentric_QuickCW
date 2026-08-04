#!/bin/bash
#SBATCH --job-name=G2D1_broad_UL_loki_4core
#SBATCH --output=/scratch/na00078/projects/eccentric_QuickCW/logs/G2D1_broad_UL_loki_4core.out
#SBATCH --error=/scratch/na00078/projects/eccentric_QuickCW/logs/G2D1_broad_UL_loki_4core.err
#SBATCH -p sbs0016
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
cd /scratch/na00078/projects/eccentric_QuickCW
source /shared/software/conda/etc/profile.d/conda.sh
conda activate /scratch/na00078/conda_envs/QuickCW
export PYTHONPATH=/scratch/na00078/projects/eccentric_QuickCW:$PYTHONPATH
export NUMBA_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
echo "JOBID=$SLURM_JOB_ID NODE=$(hostname) ALLOC_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK NUMBA=$NUMBA_NUM_THREADS OMP=$OMP_NUM_THREADS TEMPO2=$TEMPO2"
lscpu | grep -iE "model name|socket\(s\)|core\(s\) per socket|thread\(s\) per core"
echo "START: $(date)"
/usr/bin/time -v /scratch/na00078/conda_envs/QuickCW/bin/python /scratch/na00078/projects/eccentric_QuickCW/drivers/IPTA_MDC2/runQuickMCMC_IPTA_MDC2_G2D2_UL_broad_FGW_1m_lasttoa_4core.py /scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_broad_UL_loki_100M_lastTOA_4core.h5
echo "END: $(date)"
