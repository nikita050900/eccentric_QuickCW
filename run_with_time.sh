#!/bin/bash
#SBATCH --job-name=3C66B_with_time_30hr
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH -o 3C66B_with_time_30hr.out

# ==================== START INFO ==================== #
start_time=$(date)
echo "=========================================="
echo " Job started at: $start_time"
echo " Node: $(hostname)"
echo " Partition: $SLURM_JOB_PARTITION"
echo "=========================================="

# Path to QuickCW Python executable
PYTHON_PATH="/scratch/na00078/conda_envs/QuickCW/bin/python"

# HDF5 output file for this run
H5FILE="/scratch/na00078/projects/eccentric_QuickCW/h5_files/3C66B_detect_with_time_30hr.h5"

# Execute the run, passing the HDF5 filename to Python
$PYTHON_PATH runQuickMCMC_3C66B_detect.py "$H5FILE"

# ==================== END INFO ==================== #
end_time=$(date)
echo "=========================================="
echo " Job ended at: $end_time"

# Compute runtime
start_sec=$(date -d "$start_time" +%s)
end_sec=$(date -d "$end_time" +%s)
runtime=$((end_sec - start_sec))
runtime_min=$(echo "$runtime / 60" | bc)
runtime_hr=$(echo "$runtime / 3600" | bc)

echo " Total runtime: ${runtime} seconds (${runtime_min} minutes, ${runtime_hr} hours)"
echo "=========================================="
