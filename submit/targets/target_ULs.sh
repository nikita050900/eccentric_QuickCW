#!/usr/bin/env bash
set -euo pipefail
PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

sbatch <<EOF
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
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=NGC1316_gauss_dL_UL
#SBATCH --output=logs/NGC1316_gauss_dL_UL.out
#SBATCH --error=logs/NGC1316_gauss_dL_UL.err
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
hostname
echo "START: \$(date)"
export PYTHONPATH=/scratch/na00078/projects/eccentric_QuickCW:\$PYTHONPATH
/usr/bin/time -v $PY /scratch/na00078/projects/eccentric_QuickCW/drivers/targets/runQuickMCMC_NGC1316_gauss_dL.py \
    --save_filename NGC1316_gauss_dL_UL_broad.h5 \
    --amplitude_prior UL
echo "END: \$(date)"
EOF