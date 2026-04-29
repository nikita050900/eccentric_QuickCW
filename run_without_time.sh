#!/bin/bash
#SBATCH --job-name=3C66B_no_time
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
#SBATCH -o 3C66B_no_time.out

# ==================== START INFO ==================== #
start_time=$(date)
echo "=========================================="
echo " Job started at: $start_time"
echo " Node: $(hostname)"
echo " Partition: $SLURM_JOB_PARTITION"
echo "=========================================="

PYTHON_PATH="/scratch/na00078/conda_envs/QuickCW/bin/python"
H5FILE="/scratch/na00078/projects/eccentric_QuickCW/h5_files/3C66B_detect_no_time.h5"

$PYTHON_PATH runQuickMCMC_3C66B_detect.py "$H5FILE"

# ==================== END INFO ==================== #
end_time=$(date)
echo "=========================================="
echo " Job ended at: $end_time"

start_sec=$(date -d "$start_time" +%s)
end_sec=$(date -d "$end_time" +%s)
runtime=$((end_sec - start_sec))
runtime_min=$(echo "$runtime / 60" | bc)
runtime_hr=$(echo "$runtime / 3600" | bc)

echo " Total runtime: ${runtime} seconds (${runtime_min} minutes, ${runtime_hr} hours)"
echo "=========================================="
