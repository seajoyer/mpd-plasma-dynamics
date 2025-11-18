#!/bin/bash
#SBATCH --job-name=mpd-plasma-hybrid
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1  # 1 MPI task per node (total: 2 MPI tasks)
#SBATCH --cpus-per-task=32   # 32 OpenMP threads per MPI task (total: 64 CPUs)
#SBATCH --time=00:30:00      # Short for testing; increase to 02:00:00 later
#SBATCH --mem=4G
#SBATCH --output=log/mpd-plasma-%j.out
#SBATCH --error=log/mpd-plasma-%j.err

# Load modules (adjust if needed; ensures OpenMP/MPI compatibility)
# module purge
# module load gcc  # Or intel if you compiled with it

# Set OpenMP environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=false  # Disable OpenMP binding to avoid MPI conflicts

# Echo variables for debugging (will appear in .out log)
echo "=== SLURM Debug Info ==="
echo "SLURM_NTASKS: ${SLURM_NTASKS}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "Date: $(date)"
echo "======================"

# Optional: Disable some OpenMPI MCA params for better hybrid perf
# export OMPI_MCA_hwloc_base_binding_policy=none

# Launch the hybrid job
mpirun -np ${SLURM_NTASKS} --bind-to none ./build/mpd-plasma-dynamics ${OMP_NUM_THREADS}

# Job summary
echo "Job completed at $(date)"
echo "Used ${SLURM_NTASKS} MPI tasks with ${OMP_NUM_THREADS} OpenMP threads each on ${SLURM_JOB_NODELIST}"
