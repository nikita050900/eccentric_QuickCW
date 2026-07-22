#!/usr/bin/env bash
set -euo pipefail
# Run M: G2D1 (GWB-only), Upper Limit, fixed fGW, QuickCW-dL (Loki), 100M samples, last-TOA tref.
# Place in submit/IPTA_MDC2/ and run from there (logs/ is relative). Timing via /usr/bin/time -v.
PY="/scratch/na00078/conda_envs/QuickCW/bin/python"
sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=G2D1_fixed_UL_loki_100M_lastTOA_ntol_10
#SBATCH --output=logs/G2D1_fixed_UL_loki_100M_lastTOA_ntol_10.out
#SBATCH --error=logs/G2D1_fixed_UL_loki_100M_lastTOA_ntol_10.err
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
hostname
echo "START: \$(date)"
export PYTHONPATH=/scratch/na00078/projects/eccentric_QuickCW:\$PYTHONPATH
/usr/bin/time -v $PY /scratch/na00078/projects/eccentric_QuickCW/drivers/IPTA_MDC2/runQuickMCMC_IPTA_MDC2_G2D2_UL_fixed_FGW_1m_lasttoa.py \
/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D1_fixed_UL_loki_100M_lastTOA_ntol_10_\$(date +%d_%b_%Y).h5
echo "END: \$(date)"
EOF