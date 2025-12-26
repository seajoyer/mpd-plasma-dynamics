#!/bin/bash
#SBATCH --job-name=mpd-plasma
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time=1:00:00
#SBATCH --mem=10G
#SBATCH --output=log/mpd-plasma-%j.out
#SBATCH --error=log/mpd-plasma-%j.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=active

echo "=== SLURM Debug Info ==="
echo "SLURM_NTASKS: ${SLURM_NTASKS}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "SLURM_NTASKS_PER_NODE: ${SLURM_NTASKS_PER_NODE}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "OMP_PLACES: ${OMP_PLACES}"
echo "OMP_PROC_BIND: ${OMP_PROC_BIND}"
echo "OMP_WAIT_POLICY: ${OMP_WAIT_POLICY}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "Date: $(date)"
echo "======================"

# Construct the mpirun command once
MPIRUN_CMD="mpirun -np ${SLURM_NTASKS} \
  --map-by ppr:${SLURM_NTASKS_PER_NODE}:node:PE=${SLURM_CPUS_PER_TASK} \
  --bind-to core \
  ./build/mpd-plasma-dynamics ${OMP_NUM_THREADS} --end_time 4.0 --check-freq 100"

# Print the command that will be executed
echo "Executing the following command:"
echo "  ${MPIRUN_CMD}"
echo ""

# Run the command
eval "${MPIRUN_CMD}"

# Job summary
echo ""
echo "Job completed at $(date)"
echo "Used ${SLURM_NTASKS} MPI tasks with ${OMP_NUM_THREADS} OpenMP threads each on ${SLURM_JOB_NODELIST}"
