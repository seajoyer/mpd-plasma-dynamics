#!/bin/bash
#SBATCH --job-name=mpd-plasma-hybrid
#SBATCH --nodes=1
<<<<<<< HEAD
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=10:00:00
#SBATCH --mem=10G
=======
#SBATCH --ntasks-per-node=1  # 1 MPI task per node (total: 2 MPI tasks)
#SBATCH --cpus-per-task=48  # 32 OpenMP threads per MPI task (total: 64 CPUs)
#SBATCH --time=02:00:00
#SBATCH --mem=4G
>>>>>>> 1643640 (Struggle to change stream configuration)
#SBATCH --output=log/mpd-plasma-%j.out
#SBATCH --error=log/mpd-plasma-%j.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=active

echo "=== SLURM Debug Info ==="
echo "SLURM_NTASKS: ${SLURM_NTASKS}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "OMP_PLACES: ${OMP_PLACES}"
echo "OMP_PROC_BIND: ${OMP_PROC_BIND}"
echo "OMP_WAIT_POLICY: ${OMP_WAIT_POLICY}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "Date: $(date)"
echo "======================"

# mpirun -np "${SLURM_NTASKS}" --bind-to none ./build/mpd-plasma-dynamics "${OMP_NUM_THREADS}" --animate --anim-freq 20000 --format vtk

# mpirun -np "${SLURM_NTASKS}" --map-by numa:PE="${SLURM_CPUS_PER_TASK}" --bind-to core ./build/mpd-plasma-dynamics "${OMP_NUM_THREADS}" --converge 1e-4 --animate --anim-freq 50000 --check-freq 100

# mpirun -np "${SLURM_NTASKS}" --map-by numa:PE="${SLURM_CPUS_PER_TASK}" --bind-to core ./build/mpd-plasma-dynamics "${OMP_NUM_THREADS}" --end-time 4.0 --check-freq 100

# Kirill's job

mpirun -np "${SLURM_NTASKS}" --bind-to none ./build/mpd-plasma-dynamics "${OMP_NUM_THREADS}" --converge 1e-4 --animate --anim-freq 50000 --check-freq 100

# Job summary
echo "Job completed at $(date)"
echo "Used ${SLURM_NTASKS} MPI tasks with ${OMP_NUM_THREADS} OpenMP threads each on ${SLURM_JOB_NODELIST}"
