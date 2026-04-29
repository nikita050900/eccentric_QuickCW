#!/usr/bin/env bash
set -euo pipefail

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=G2D2_fixed_detect_loki
#SBATCH --output=G2D2_fixed_detect_loki.out
#SBATCH --error=G2D2_fixed_detect_loki.err
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

hostname
echo "START: \$(date)"

/usr/bin/time -v $PY runQuickMCMC_IPTA_MDC2_G2D2_DETECT_FIXED_FGW.py \
    /scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_fixed_detect_loki_\$(date +%d_%b_%Y).h5

echo "END: \$(date)"
EOF