#!/bin/bash
#SBATCH --job-name=3C66B_master
#SBATCH --output=3C66B_master.log
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1

# Path to your QuickCW python executable (Python 3.6+)
PYTHON_PATH="/scratch/na00078/conda_envs/QuickCW/bin/python"

# Submit detection job first
detect_jobid=$(sbatch --job-name=3C66B_detect \
    --output=3C66B_detect_2k_iterations.out \
    --parsable \
    --wrap="$PYTHON_PATH runQuickMCMC_3C66B_detect.py")

echo "Submitted detection job with JobID: $detect_jobid"