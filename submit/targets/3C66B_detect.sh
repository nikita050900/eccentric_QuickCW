#!/bin/bash
#SBATCH --job-name=3C66B_master
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
#SBATCH --output=3C66B_master.log

# ==================== START INFO ==================== #
start_time=$(date)
echo "=========================================="
echo " Job started at: $start_time"
echo " Node: $(hostname)"
echo " Partition: $SLURM_JOB_PARTITION"
echo "=========================================="

# Path to your QuickCW python executable (Python 3.6+)
PYTHON_PATH="/scratch/na00078/conda_envs/QuickCW/bin/python"

# Submit detection job to the SAME partition
detect_jobid=$(sbatch \
    --job-name=3C66B_detect \
    --partition=$SLURM_JOB_PARTITION \
    --output=3C66B_detect_sarah_node.out \
    --parsable \
    --wrap="$PYTHON_PATH runQuickMCMC_3C66B_detect.py")

echo "Submitted detection job with JobID: $detect_jobid"

# ==================== END INFO ==================== #
end_time=$(date)
echo "=========================================="
echo " Job ended at: $end_time"

# Compute runtime
start_sec=$(date -d "$start_time" +%s)
end_sec=$(date -d "$end_time" +%s)
runtime=$((end_sec - start_sec))
runtime_min=$(echo "$runtime / 60" | bc)

echo " Total runtime: ${runtime} seconds (${runtime_min} minutes)"
echo "=========================================="
