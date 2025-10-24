#!/bin/bash
#SBATCH --job-name=NGC3115_master
#SBATCH --output=NGC3115_master.log
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1

# Record start time
start_time=$(date +%s)
echo "===================================================="
echo "Job started at: $(date)"
echo "===================================================="

# Path to your QuickCW python executable (Python 3.6+)
PYTHON_PATH="/scratch/na00078/conda_envs/QuickCW/bin/python"

# Submit detection job first
detect_jobid=$(sbatch --job-name=NGC3115_detect \
    --output=NGC3115_detect.out \
    --parsable \
    --wrap="$PYTHON_PATH runQuickMCMC_NGC3115_detect.py")

echo "Submitted detection job with JobID: $detect_jobid"

# Wait for job to finish if needed
# scontrol wait $detect_jobid   # uncomment if you want to block until completion

# Record end time and compute runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

echo "===================================================="
echo "Job finished at: $(date)"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"
echo "===================================================="
