#include <mpi.h>
#include <omp.h>
#include <yaml-cpp/yaml.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "config.hpp"
#include "diagnostics.hpp"
#include "fields.hpp"
#include "geometry_registry.hpp"
#include "grid.hpp"
#include "ics/expression_ic.hpp"
#include "ics/vtk_ic.hpp"
#include "iinitial_condition.hpp"
#include "integrals.hpp"
#include "io_manager.hpp"
#include "mpi_manager.hpp"
#include "solver.hpp"
#include "verification.hpp"

auto main(int argc, char* argv[]) -> int {
    // ----------------------------------------------------------------
    // 1. Register all built-in geometry types.
    // ----------------------------------------------------------------
    RegisterAllGeometries();

    // ----------------------------------------------------------------
    // 2. Configuration
    // ----------------------------------------------------------------
    const char* config_path = (argc > 1) ? argv[1] : "config.yaml";

    SimConfig cfg;
    cfg.Load(config_path);

    MPIManager mpi(argc, argv, cfg);

    if (mpi.rank == 0) {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
    }

    if (mpi.rank == 0) {
        std::printf("Config file         : %s\n", config_path);
        std::printf("Grid                : %d x %d  (L x M)\n", cfg.L_max, cfg.M_max);
        std::printf("Initial dt / T end  : %.6e / %.4f\n", cfg.dt, cfg.T);
        std::printf("MPI ranks           : %d  (%d x %d Cartesian)\n", mpi.size,
                    mpi.dims[0], mpi.dims[1]);
        std::printf("Geometry            : %s\n", cfg.geometry.type.c_str());

        if (cfg.adaptive_dt) {
            std::printf(
                "Adaptive dt         : ON  (CFL=%.2f, growth=%.2f, "
                "dt_min=%.1e, dt_max=%.1e)\n",
                cfg.cfl_number, cfg.dt_growth_factor, cfg.dt_min, cfg.dt_max);
        } else {
            std::printf("Adaptive dt         : OFF (fixed dt=%.6e)\n", cfg.dt);
        }

        if (cfg.convergence_threshold > 0.0) {
            std::printf("Convergence check   : threshold = %.2e, every %d steps\n",
                        cfg.convergence_threshold, cfg.check_frequency);
        }

        if (cfg.vtk_step > 0) {
            std::printf("VTK output every    : %d steps\n", cfg.vtk_step);
        }

        // Summarise the diagnostics toggles so it's obvious from the run
        // log when any of them are off.
        const auto& d = cfg.diagnostics;
        if (!d.cfl_check || !d.conserved_integrals || !d.console_checkpoint ||
            !d.outlet_diagnostics || !d.startup_verification || !d.write_mpi_rank) {
            std::printf("Diagnostics         : cfl=%s integrals=%s console=%s "
                        "outlet=%s startup_verif=%s mpi_rank=%s\n",
                        d.cfl_check            ? "on" : "off",
                        d.conserved_integrals  ? "on" : "off",
                        d.console_checkpoint   ? "on" : "off",
                        d.outlet_diagnostics   ? "on" : "off",
                        d.startup_verification ? "on" : "off",
                        d.write_mpi_rank       ? "on" : "off");
        }
    }

    if (cfg.openmp_threads > 0) {
        omp_set_num_threads(cfg.openmp_threads);
    }

    // ----------------------------------------------------------------
    // Verification check 1: ghost exchange round-trip.
    // ----------------------------------------------------------------
    if (cfg.diagnostics.startup_verification) {
        if (mpi.rank == 0) {
            std::printf("\n--- Layer-1 verification ---\n");
        }
        const bool ghost_ok = Verification::CheckGhostExchange(mpi, cfg);
        if (!ghost_ok) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // ----------------------------------------------------------------
    // 3. Build geometry
    // ----------------------------------------------------------------
    YAML::Node geom_params;
    if (!cfg.geometry.params_yaml.empty()) {
        geom_params = YAML::Load(cfg.geometry.params_yaml);
    }

    auto geometry = GeometryRegistry::Instance().Create(cfg.geometry.type, geom_params);

    // ----------------------------------------------------------------
    // 4. Build initial condition
    // ----------------------------------------------------------------
    std::unique_ptr<IInitialCondition> ic;

    if (!cfg.initial_conditions.vtk_file.empty()) {
        if (mpi.rank == 0) {
            std::printf("IC                  : VTK restart  '%s'\n",
                        cfg.initial_conditions.vtk_file.c_str());
        }
        ic = std::make_unique<VtkIC>(cfg.initial_conditions.vtk_file);
    } else {
        YAML::Node ic_params;
        if (!cfg.initial_conditions.params_yaml.empty()) {
            ic_params = YAML::Load(cfg.initial_conditions.params_yaml);
        }
        if (mpi.rank == 0) {
            std::printf("IC                  : expression\n");
        }
        ic = std::make_unique<ExpressionIC>(ic_params);
    }

    // ----------------------------------------------------------------
    // 5. Build grid and initialise fields
    // ----------------------------------------------------------------
    Grid grid(cfg, mpi.local_L_with_ghosts, mpi.l_start, mpi.local_M_with_ghosts,
              mpi.m_start, *geometry);

    Fields fields(mpi.local_L_with_ghosts, mpi.local_M_with_ghosts,
                  cfg.convergence_threshold > 0.0);

    fields.InitPhysical(*ic, cfg, grid, mpi.l_start);
    fields.InitConservative(grid);

    if (cfg.convergence_threshold > 0.0) {
        fields.SavePrev();
    }

    if (cfg.diagnostics.startup_verification) {
        Verification::CheckRadialSymmetry(fields, mpi);
    }

    Verification::ConservedIntegrals integrals_ref{};

    if (cfg.diagnostics.conserved_integrals) {
        integrals_ref = Verification::ComputeIntegrals(fields, grid, cfg, mpi);
        Verification::PrintIntegralsHeader(mpi.rank);
        Verification::PrintIntegrals(integrals_ref, 0, 0.0, mpi.rank);

        if (mpi.rank == 0) {
            std::printf("----------------------------\n\n");
        }
    }

    // ----------------------------------------------------------------
    // 6. Construct solver and I/O manager
    // ----------------------------------------------------------------
    Solver solver(cfg, mpi, grid, fields);
    IOManager io(cfg, mpi);

    // ----------------------------------------------------------------
    // 7. Time loop
    // ----------------------------------------------------------------
    double t = 0.0;
    int step_count = 0;
    bool converged = false;
    double dt = cfg.dt;

    // Cached max wave speed for the steady-flow ComputeDt optimisation.
    // Initialised to -1 so the first call always runs a full Allreduce.
    double prev_max_speed = -1.0;

    constexpr int check_l_global = 20;
    constexpr int check_m_global = 40;

    const bool owns_checkpoint =
        (check_l_global >= mpi.l_start && check_l_global <= mpi.l_end) &&
        (check_m_global >= mpi.m_start && check_m_global <= mpi.m_end);

    const int check_l_local = owns_checkpoint ? (check_l_global - mpi.l_start + 1) : -1;
    const int check_m_local = owns_checkpoint ? (check_m_global - mpi.m_start + 1) : -1;

    if (mpi.rank == 0 && cfg.diagnostics.console_checkpoint) {
        std::printf("%-14s %-14s %-14s %-14s %-14s %-14s %-14s\n", "t", "dt", "rho",
                    "v_z", "v_phi", "e", "H_phi");
        std::printf("%-14s %-14s %-14s %-14s %-14s %-14s %-14s\n", "--------------",
                    "--------------", "--------------", "--------------",
                    "--------------", "--------------", "--------------");
    }

    if (cfg.vtk_step > 0) {
        io.WriteFrame(0, fields, grid);
    }

    const double begin = mpi.Wtime();

    while (t < cfg.T && !converged) {
        if (t + dt > cfg.T) {
            dt = cfg.T - t;
        }

        solver.Advance(dt);
        t += dt;
        ++step_count;

        constexpr int kDtRecomputeStride = 10;
        if (cfg.adaptive_dt && step_count % kDtRecomputeStride == 0) {
            dt = Diagnostics::ComputeDt(fields, cfg, mpi.local_L, mpi.local_M,
                                        mpi, dt, prev_max_speed);
        }

        if (cfg.convergence_threshold > 0.0 && step_count % cfg.check_frequency == 0) {
            const double change =
                Diagnostics::SolutionChange(fields, mpi.local_L, mpi.local_M);
            if (mpi.rank == 0) {
                std::printf("Step %d, t=%.6f, dt=%.6e, relative change: %.6e\n",
                            step_count, t, dt, change);
            }

            if (change < cfg.convergence_threshold) {
                converged = true;
                if (mpi.rank == 0) {
                    std::printf("Converged at t=%.6f after %d steps\n", t, step_count);
                }
            }
            fields.SavePrev();
        }

        if (cfg.diagnostics.cfl_check && step_count % 100 == 0) {
            Diagnostics::CheckCfl(fields, cfg, mpi, mpi.local_L, mpi.local_M, dt,
                                  step_count);
        }

        if (cfg.vtk_step > 0 && step_count % cfg.vtk_step == 0) {
            io.WriteFrame(step_count, fields, grid);
        }

        if (step_count % 1000 == 0) {
            if (cfg.diagnostics.console_checkpoint) {
                double local_vals[5] = {0, 0, 0, 0, 0};
                if (owns_checkpoint) {
                    local_vals[0] = fields.rho  [check_l_local][check_m_local];
                    local_vals[1] = fields.v_z  [check_l_local][check_m_local];
                    local_vals[2] = fields.v_phi[check_l_local][check_m_local];
                    local_vals[3] = fields.e    [check_l_local][check_m_local];
                    local_vals[4] = fields.H_phi[check_l_local][check_m_local];
                }
                double global_vals[5];
                MPI_Reduce(local_vals, global_vals, 5, MPI_DOUBLE, MPI_SUM, 0,
                           MPI_COMM_WORLD);
                if (mpi.rank == 0) {
                    std::printf("%-14.6f %-14.6e %-14.6f %-14.6f %-14.6f %-14.6f %-14.6f\n",
                                t, dt, global_vals[0], global_vals[1], global_vals[2],
                                global_vals[3], global_vals[4]);
                }
            }

            if (cfg.diagnostics.conserved_integrals) {
                const auto integrals_now =
                    Verification::ComputeIntegrals(fields, grid, cfg, mpi);
                Verification::PrintIntegrals(integrals_now, step_count, t, mpi.rank);
            }
        }
    }

    if (mpi.rank == 0) {
        std::printf("\nCalculation time : %.3f sec  (%d steps)\n", mpi.Wtime() - begin,
                    step_count);
    }

    io.WriteFrame(step_count, fields, grid);

    if (mpi.rank == 0) {
        std::printf("Final VTK written : %s/step_%04d.vtk\n", io.RunDir().c_str(),
                    step_count);
    }

    if (cfg.diagnostics.conserved_integrals) {
        const auto integrals_final =
            Verification::ComputeIntegrals(fields, grid, cfg, mpi);
        Verification::ReportDrift(integrals_ref, integrals_final, step_count, t,
                                  mpi.rank);
    }

    if (cfg.diagnostics.outlet_diagnostics) {
        const double mass_flux = Diagnostics::GetMassFlux(fields, grid, cfg, mpi);
        const double thrust    = Diagnostics::GetThrust  (fields, grid, cfg, mpi);

        if (mpi.rank == 0) {
            std::printf("\n--- Outlet-plane diagnostics ---\n");
            std::printf("Mass flux : %.6e\n", mass_flux);
            std::printf("Thrust    : %.6e\n", thrust);
            std::printf("--------------------------------\n");
        }
    }

    return 0;
}
