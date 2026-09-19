#!/usr/bin/env bash
# Submit all 8 production CURN runs: {NGC3115,NGC1316} x {detection,UL} x {withJ1713,noJ1713}
# Priors frozen 2026-08-17: NGC3115 10.0+/-1.0 Mpc, NGC1316 19.3+/-1.93 Mpc, mc [6,10]
set -euo pipefail
PY="/scratch/na00078/conda_envs/QuickCW/bin/python"
DRIVER="/scratch/na00078/projects/eccentric_QuickCW/drivers/targets/runQuickMCMC_broad_prod.py"
cd /scratch/na00078/projects/eccentric_QuickCW/submit/targets
for TARGET in NGC3115 NGC1316; do
  for PRIOR in detection UL; do
    for PSRSET in withJ1713 noJ1713; do
      DROPFLAG=""
      if [ "$PSRSET" = "noJ1713" ]; then DROPFLAG="--drop_j1713"; fi
      NAME="${TARGET}_${PRIOR}_${PSRSET}"
      jid=$(sbatch --parsable << EOF
#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --output=logs/${NAME}.out
#SBATCH --error=logs/${NAME}.err
#SBATCH -p sbs0016
#SBATCH --time=10-00:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
hostname
echo "START: \$(date)"
export PYTHONPATH=/scratch/na00078/projects/eccentric_QuickCW:\$PYTHONPATH
export NUMBA_NUM_THREADS=4
export TEMPO2=/scratch/na00078/conda_envs/QuickCW/share/tempo2
/usr/bin/time -v $PY $DRIVER --target ${TARGET} --amplitude_prior ${PRIOR} ${DROPFLAG}
echo "END: \$(date)"
EOF
)
      echo "submitted ${NAME} as job ${jid}"
    done
  done
done
