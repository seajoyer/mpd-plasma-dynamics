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
    SimConfig cfg;
    cfg.init();

    // MPIManager must be constructed before cfg.parse_args so rank is known.
    MPIManager mpi(argc, argv, cfg);

    cfg.parse_args(argc, argv, mpi.rank);

    // Legacy: argv[1] carries the OpenMP thread count
    const int omp_threads = (argc > 1) ? std::atoi(argv[1]) : 1;
    omp_set_num_threads(omp_threads);

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

    const double begin = mpi.wtime();

    while (t < cfg.T && !converged) {

        solver.advance();

        // ---- convergence check ----
        if (cfg.convergence_threshold > 0.0
                && step_count % cfg.check_frequency == 0) {

            const double change = Diagnostics::solution_change(
                fields, mpi.local_L, cfg.M_max);

            if (mpi.rank == 0)
                printf("Step %d, t=%.6f, relative change: %.6e\n",
                       step_count, t, change);

            if (change < cfg.convergence_threshold) {
                converged = true;
                if (mpi.rank == 0)
                    printf("Converged at t=%.6f after %d steps\n",
                           t, step_count);
            }
            fields.save_prev();
        }

        // ---- CFL check every 100 steps ----
        if (step_count % 100 == 0)
            Diagnostics::check_cfl(fields, cfg, mpi,
                                   mpi.local_L, step_count);

        // ---- animation output ----
        if (cfg.animate == 1 && (int)(t * 10000) % 1000 == 0)
            io.write_animate_frame((int)(t * 10000), fields, grid);

        // ---- periodic console checkpoint ----
        if (mpi.rank == 0 && step_count % 1000 == 0) {
            constexpr int check_l_global = 20;
            constexpr int check_m        = 40;
            if (check_l_global >= mpi.l_start && check_l_global <= mpi.l_end) {
                const int lc = check_l_global - mpi.l_start + 1;
                printf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
                       t,
                       fields.rho  [lc][check_m],
                       fields.v_z  [lc][check_m],
                       fields.v_phi[lc][check_m],
                       fields.e    [lc][check_m],
                       fields.H_phi[lc][check_m]);
            }
        }

        t += cfg.dt;
        ++step_count;
    }

    if (mpi.rank == 0)
        printf("Calculation time : %lf sec\n", mpi.wtime() - begin);

    // ----------------------------------------------------------------
    // 5. Gather and write output
    // ----------------------------------------------------------------
    io.gather_global(fields, grid);
    io.write_vtk    ("output_MHD.vtk");
    io.write_tecplot("28-2D_MHD_LF_rzphi_MPD_MPI.plt");

    return 0;
}
