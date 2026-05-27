#!/usr/bin/env bash
set -euo pipefail
PY="/scratch/na00078/conda_envs/QuickCW/bin/python"
detect_jobid=$(
  sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=NGC3115_gauss_dL_detect
#SBATCH --output=logs/NGC3115_gauss_dL_detect.out
#SBATCH --error=logs/NGC3115_gauss_dL_detect.err
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
hostname
echo "START: \$(date)"
export PYTHONPATH=/scratch/na00078/projects/eccentric_QuickCW:\$PYTHONPATH
/usr/bin/time -v $PY /scratch/na00078/projects/eccentric_QuickCW/drivers/targets/runQuickMCMC_NGC3115_gauss_dL.py \
    --save_filename NGC3115_gauss_dL_detect_broad.h5 \
    --amplitude_prior detection
echo "END: \$(date)"
EOF
)
echo "Submitted detection job: ${detect_jobid}"
sbatch --dependency=afterok:${detect_jobid} <<EOF
#!/bin/bash
#SBATCH --job-name=NGC3115_gauss_dL_UL
#SBATCH --output=logs/NGC3115_gauss_dL_UL.out
#SBATCH --error=logs/NGC3115_gauss_dL_UL.err
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
hostname
echo "START: \$(date)"
export PYTHONPATH=/scratch/na00078/projects/eccentric_QuickCW:\$PYTHONPATH
/usr/bin/time -v $PY /scratch/na00078/projects/eccentric_QuickCW/drivers/targets/runQuickMCMC_NGC3115_gauss_dL.py \
    --save_filename NGC3115_gauss_dL_UL_broad.h5 \
    --amplitude_prior UL
echo "END: \$(date)"
EOF
echo "Submitted UL job (afterok:${detect_jobid})"