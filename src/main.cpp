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

    // When stdout is redirected to a file (as SLURM always does) the C
    // runtime switches from line-buffered to fully-buffered mode.  Every
    // printf then silently accumulates in a kernel buffer and only appears
    // in the log once that buffer fills up — producing multi-minute latency
    // and burst prints.  Setting unbuffered mode on rank 0 fixes this with
    // negligible overhead because rank 0 only emits infrequent diagnostic
    // lines.
    if (mpi.rank == 0)
        std::setvbuf(stdout, nullptr, _IONBF, 0);

    if (mpi.rank == 0) {
        std::printf("Config file         : %s\n", config_path);
        std::printf("Grid                : %d x %d  (L x M)\n",
                    cfg.L_max_global, cfg.M_max);
        std::printf("Initial dt / T end  : %.6e / %.4f\n", cfg.dt, cfg.T);
        std::printf("MPI ranks           : %d  (%d x %d Cartesian)\n",
                    mpi.size, mpi.dims[0], mpi.dims[1]);

        if (cfg.adaptive_dt) {
            std::printf("Adaptive dt         : ON  (CFL=%.2f, growth=%.2f, "
                        "dt_min=%.1e, dt_max=%.1e)\n",
                        cfg.cfl_number, cfg.dt_growth_factor,
                        cfg.dt_min, cfg.dt_max);
        } else {
            std::printf("Adaptive dt         : OFF (fixed dt=%.6e)\n", cfg.dt);
        }

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

    // dt for the upcoming step.  Initialised from config; updated each step
    // when adaptive_dt is enabled.
    double dt = cfg.dt;

    // Periodic console checkpoint: probe cell at global (l=20, m=40).
    constexpr int check_l_global = 20;
    constexpr int check_m_global = 40;

    const bool owns_checkpoint =
        (check_l_global >= mpi.l_start && check_l_global <= mpi.l_end) &&
        (check_m_global >= mpi.m_start && check_m_global <= mpi.m_end);

    const int check_l_local = owns_checkpoint
                              ? (check_l_global - mpi.l_start + 1) : -1;
    const int check_m_local = owns_checkpoint
                              ? (check_m_global - mpi.m_start + 1) : -1;

    if (mpi.rank == 0) {
        std::printf("\n%-14s %-14s %-14s %-14s %-14s %-14s %-14s\n",
                    "t", "dt", "rho", "v_z", "v_phi", "e", "H_phi");
        std::printf("%-14s %-14s %-14s %-14s %-14s %-14s %-14s\n",
                    "--------------", "--------------", "--------------",
                    "--------------", "--------------", "--------------",
                    "--------------");
    }

    // Write step 0 before the loop starts.
    if (cfg.vtk_step > 0)
        io.write_frame(0, fields, grid);

    const double begin = mpi.wtime();

    while (t < cfg.T && !converged) {

        // Clamp dt so we land exactly on T without overshooting.
        if (t + dt > cfg.T)
            dt = cfg.T - t;

        solver.advance(dt);
        t          += dt;
        ++step_count;

        // ---- adaptive dt: recompute from updated fields ------------------
        // We compute the new dt *after* the step so we can use the freshly
        // updated wave speeds.  The growth-rate cap in compute_dt() prevents
        // the step from jumping too far if the wave speed dropped sharply.
        if (cfg.adaptive_dt) {
            dt = Diagnostics::compute_dt(fields, cfg,
                                         mpi.local_L, mpi.local_M, mpi, dt);
        }

        // ---- convergence check -------------------------------------------
        if (cfg.convergence_threshold > 0.0
                && step_count % cfg.check_frequency == 0) {

            const double change = Diagnostics::solution_change(
                fields, mpi.local_L, mpi.local_M);

            if (mpi.rank == 0)
                std::printf("Step %d, t=%.6f, dt=%.6e, relative change: %.6e\n",
                            step_count, t, dt, change);

            if (change < cfg.convergence_threshold) {
                converged = true;
                if (mpi.rank == 0)
                    std::printf("Converged at t=%.6f after %d steps\n",
                                t, step_count);
            }
            fields.save_prev();
        }

        // ---- CFL sanity check every 100 steps ----------------------------
        // With adaptive_dt this should never fire; it acts as a safety net.
        if (step_count % 100 == 0)
            Diagnostics::check_cfl(fields, cfg, mpi,
                                   mpi.local_L, mpi.local_M,
                                   dt, step_count);

        // ---- VTK animation frame ----------------------------------------
        if (cfg.vtk_step > 0 && step_count % cfg.vtk_step == 0)
            io.write_frame(step_count, fields, grid);

        // ---- periodic console checkpoint --------------------------------
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
                std::printf("%-14.6f %-14.6e %-14.6f %-14.6f %-14.6f %-14.6f %-14.6f\n",
                            t, dt,
                            global_vals[0], global_vals[1], global_vals[2],
                            global_vals[3], global_vals[4]);
            }
        }
    }

    if (mpi.rank == 0)
        std::printf("\nCalculation time : %.3f sec  (%d steps)\n",
                    mpi.wtime() - begin, step_count);

    // ----------------------------------------------------------------
    // 5. Write final VTK frame (always, regardless of vtk_step)
    // ----------------------------------------------------------------
    io.write_frame(step_count, fields, grid);

    if (mpi.rank == 0)
        std::printf("Final VTK written : %s/step_%04d.vtk\n",
                    io.run_dir().c_str(), step_count);

    return 0;
}
