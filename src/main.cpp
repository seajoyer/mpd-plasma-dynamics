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

    // Determine config file path: first positional argument or default.
    const char* config_path = (argc > 1) ? argv[1] : "config.yaml";

    SimConfig cfg;
    cfg.load(config_path);

    // MPIManager must be constructed before any printing so rank is known.
    MPIManager mpi(argc, argv, cfg);

    if (mpi.rank == 0) {
        std::printf("Config file         : %s\n", config_path);
        std::printf("Grid                : %d x %d  (L x M)\n",
                    cfg.L_max_global, cfg.M_max);
        std::printf("Time step / end     : %.6e / %.4f\n", cfg.dt, cfg.T);
        std::printf("MPI ranks           : %d\n", mpi.size);

        if (cfg.convergence_threshold > 0.0)
            std::printf("Convergence check   : threshold = %.2e, every %d steps\n",
                        cfg.convergence_threshold, cfg.check_frequency);

        if (cfg.vtk_step > 0)
            std::printf("VTK output every    : %d steps\n", cfg.vtk_step);
    }

    // Apply OpenMP thread count (0 → keep whatever OMP_NUM_THREADS set).
    if (cfg.openmp_threads > 0)
        omp_set_num_threads(cfg.openmp_threads);

    // ----------------------------------------------------------------
    // 2. Build grid and initialise fields
    // ----------------------------------------------------------------
    Grid   grid(cfg, mpi.local_L_with_ghosts, mpi.l_start);
    Fields fields(mpi.local_L_with_ghosts, cfg.M_max + 1,
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

    // Print periodic-checkpoint table header before the first row.
    constexpr int check_l_global = 20;
    constexpr int check_m        = 40;

    if (mpi.rank == 0
            && check_l_global >= mpi.l_start
            && check_l_global <= mpi.l_end) {
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
                fields, mpi.local_L, cfg.M_max);

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
                                   mpi.local_L, step_count);

        // ---- VTK animation frame ----
        if (cfg.vtk_step > 0 && step_count % cfg.vtk_step == 0)
            io.write_frame(step_count, fields, grid);

        // ---- periodic console checkpoint ----
        if (mpi.rank == 0 && step_count % 1000 == 0) {
            if (check_l_global >= mpi.l_start && check_l_global <= mpi.l_end) {
                const int lc = check_l_global - mpi.l_start + 1;
                std::printf("%-14.6f %-14.6f %-14.6f %-14.6f %-14.6f %-14.6f\n",
                            t,
                            fields.rho  [lc][check_m],
                            fields.v_z  [lc][check_m],
                            fields.v_phi[lc][check_m],
                            fields.e    [lc][check_m],
                            fields.H_phi[lc][check_m]);
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
