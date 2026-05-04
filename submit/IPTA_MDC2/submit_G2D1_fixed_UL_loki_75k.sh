#!/usr/bin/env bash
set -euo pipefail

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=G2D1_fixed_UL_loki_75k
#SBATCH --output=logs/G2D1_fixed_UL_loki_75k.out
#SBATCH --error=logs/G2D1_fixed_UL_loki_75k.err
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

hostname
echo "START: \$(date)"

export PYTHONPATH=/scratch/na00078/projects/eccentric_QuickCW:\$PYTHONPATH

/usr/bin/time -v $PY drivers/IPTA_MDC2/runQuickMCMC_IPTA_MDC2_G2D1_UL_fixed_FGW.py \
    /scratch/na00078/projects/IPTA_MDC2/h5_files/G2D1_fixed_UL_loki_75k_\$(date +%d_%b_%Y).h5

echo "END: \$(date)"
EOF
