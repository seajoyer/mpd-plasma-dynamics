#!/bin/bash
#SBATCH --job-name=slurm_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=120:00:00
#SBATCH --mem=4G
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

# Set OMP_NUM_THREADS for safety
export OMP_NUM_THREADS=48

# Run your executable with 48 as the command-line argument
./build/cpp_openmp_project 48
