#include <cstdlib>
#include <cstdio>
#include <omp.h>

#include "config.hpp"
#include "mpi_manager.hpp"
#include "grid.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "diagnostics.hpp"
#include "io_manager.hpp"

int main(int argc, char* argv[]) {

    // ----------------------------------------------------------------
    // 1. Configuration
    // ----------------------------------------------------------------

    const char* config_path = (argc > 1) ? argv[1] : "config.yaml";

    SimConfig cfg;
    cfg.load(config_path);

    // MPIManager initialises MPI and builds the 2-D Cartesian topology.
    MPIManager mpi(argc, argv, cfg);

    if (mpi.rank == 0) {
        std::printf("Config file         : %s\n", config_path);
        std::printf("Grid                : %d x %d  (L x M)\n",
                    cfg.L_max_global, cfg.M_max);
        std::printf("Time step / end     : %.6e / %.4f\n", cfg.dt, cfg.T);
        std::printf("MPI ranks           : %d  (%d x %d Cartesian)\n",
                    mpi.size, mpi.dims[0], mpi.dims[1]);

        if (cfg.convergence_threshold > 0.0)
            std::printf("Convergence check   : threshold = %.2e, every %d steps\n",
                        cfg.convergence_threshold, cfg.check_frequency);

        if (cfg.vtk_step > 0)
            std::printf("VTK output every    : %d steps\n", cfg.vtk_step);
    }

    if (cfg.openmp_threads > 0)
        omp_set_num_threads(cfg.openmp_threads);

    // ----------------------------------------------------------------
    // 2. Build grid and initialise fields
    // ----------------------------------------------------------------

    // Grid and Fields are constructed with local dimensions including ghosts.
    Grid grid(cfg,
              mpi.local_L_with_ghosts, mpi.l_start,
              mpi.local_M_with_ghosts, mpi.m_start);

    Fields fields(mpi.local_L_with_ghosts, mpi.local_M_with_ghosts,
                  cfg.convergence_threshold > 0.0);

    fields.init_physical    (cfg, grid, mpi.l_start);
    fields.init_conservative(grid);

    if (cfg.convergence_threshold > 0.0)
        fields.save_prev();

    // ----------------------------------------------------------------
    // 3. Construct solver and I/O manager
    // ----------------------------------------------------------------

    Solver    solver(cfg, mpi, grid, fields);
    IOManager io    (cfg, mpi);

    // ----------------------------------------------------------------
    // 4. Time loop
    // ----------------------------------------------------------------

    double t          = 0.0;
    int    step_count = 0;
    bool   converged  = false;

    // Periodic console checkpoint: probe cell at global (l=20, m=40).
    // Determine which rank owns this cell under the 2-D decomposition.
    constexpr int check_l_global = 20;
    constexpr int check_m_global = 40;

    // Does this rank own the checkpoint cell?
    const bool owns_checkpoint =
        (check_l_global >= mpi.l_start && check_l_global <= mpi.l_end) &&
        (check_m_global >= mpi.m_start && check_m_global <= mpi.m_end);

    // Local indices for the checkpoint cell (ghost offset = 1).
    const int check_l_local = owns_checkpoint
                              ? (check_l_global - mpi.l_start + 1) : -1;
    const int check_m_local = owns_checkpoint
                              ? (check_m_global - mpi.m_start + 1) : -1;

    // Print table header from rank 0 (which may or may not own the cell;
    // the actual value is gathered via a reduce below).
    if (mpi.rank == 0) {
        std::printf("\n%-14s %-14s %-14s %-14s %-14s %-14s\n",
                    "t", "rho", "v_z", "v_phi", "e", "H_phi");
        std::printf("%-14s %-14s %-14s %-14s %-14s %-14s\n",
                    "--------------", "--------------", "--------------",
                    "--------------", "--------------", "--------------");
    }

    // Write step 0 before the loop starts.
    if (cfg.vtk_step > 0)
        io.write_frame(0, fields, grid);

    const double begin = mpi.wtime();

    while (t < cfg.T && !converged) {

        solver.advance();
        t += cfg.dt;
        ++step_count;

        // ---- convergence check ----
        if (cfg.convergence_threshold > 0.0
                && step_count % cfg.check_frequency == 0) {

            const double change = Diagnostics::solution_change(
                fields, mpi.local_L, mpi.local_M);

            if (mpi.rank == 0)
                std::printf("Step %d, t=%.6f, relative change: %.6e\n",
                            step_count, t, change);

            if (change < cfg.convergence_threshold) {
                converged = true;
                if (mpi.rank == 0)
                    std::printf("Converged at t=%.6f after %d steps\n",
                                t, step_count);
            }
            fields.save_prev();
        }

        // ---- CFL check every 100 steps ----
        if (step_count % 100 == 0)
            Diagnostics::check_cfl(fields, cfg, mpi,
                                   mpi.local_L, mpi.local_M, step_count);

        // ---- VTK animation frame ----
        if (cfg.vtk_step > 0 && step_count % cfg.vtk_step == 0)
            io.write_frame(step_count, fields, grid);

        // ---- periodic console checkpoint ----
        // The checkpoint cell may live on any rank in the 2-D topology.
        // Each quantity is gathered to rank 0 via Allreduce with MPI_SUM:
        // only the owning rank contributes a non-zero value.
        if (step_count % 1000 == 0) {
            double local_vals[5] = {0, 0, 0, 0, 0};
            if (owns_checkpoint) {
                local_vals[0] = fields.rho  [check_l_local][check_m_local];
                local_vals[1] = fields.v_z  [check_l_local][check_m_local];
                local_vals[2] = fields.v_phi[check_l_local][check_m_local];
                local_vals[3] = fields.e    [check_l_local][check_m_local];
                local_vals[4] = fields.H_phi[check_l_local][check_m_local];
            }
            double global_vals[5];
            MPI_Reduce(local_vals, global_vals, 5, MPI_DOUBLE, MPI_SUM,
                       0, MPI_COMM_WORLD);

            if (mpi.rank == 0) {
                std::printf("%-14.6f %-14.6f %-14.6f %-14.6f %-14.6f %-14.6f\n",
                            t,
                            global_vals[0], global_vals[1], global_vals[2],
                            global_vals[3], global_vals[4]);
            }
        }
    }

    if (mpi.rank == 0)
        std::printf("\nCalculation time : %.3f sec\n", mpi.wtime() - begin);

    // ----------------------------------------------------------------
    // 5. Write final VTK frame (always, regardless of vtk_step)
    // ----------------------------------------------------------------
    io.write_frame(step_count, fields, grid);

    if (mpi.rank == 0)
        std::printf("Final VTK written : %s/step_%04d.vtk\n",
                    io.run_dir().c_str(), step_count);

    return 0;
}
