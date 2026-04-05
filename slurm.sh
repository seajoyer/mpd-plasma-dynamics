#!/bin/bash
#SBATCH --job-name=mpd-plasma-hybrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=2
#SBATCH --time=04:30:00
#SBATCH --mem=8G
#SBATCH --output=output/log/mpd-plasma-%j.out
#SBATCH --error=output/log/mpd-plasma-%j.err

export OMP_PLACES=cores
export OMP_WAIT_POLICY=passive
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close

CONFIG_FILE="${CONFIG_FILE:-config.yaml}"

echo "=== SLURM Debug Info ==="
echo "SLURM_NTASKS:         ${SLURM_NTASKS}"
echo "SLURM_CPUS_PER_TASK:  ${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS:      ${OMP_NUM_THREADS}"
echo "OMP_PLACES:           ${OMP_PLACES}"
echo "OMP_PROC_BIND:        ${OMP_PROC_BIND}"
echo "OMP_WAIT_POLICY:      ${OMP_WAIT_POLICY}"
echo "SLURM_JOB_NODELIST:   ${SLURM_JOB_NODELIST}"
echo "CONFIG_FILE:          ${CONFIG_FILE}"
echo "Date: $(date)"
echo "======================"

srun --mpi=pmix \
     --cpus-per-task=${SLURM_CPUS_PER_TASK} \
     --cpu-bind=verbose,cores \
     ./build/mpd-plasma-dynamics "${CONFIG_FILE}"

echo "Job completed at $(date)"
echo "Used ${SLURM_NTASKS} MPI tasks with ${OMP_NUM_THREADS} OpenMP threads each on ${SLURM_JOB_NODELIST}"
