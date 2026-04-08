#include "diagnostics.hpp"

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace Diagnostics {

// ============================================================
// max_wave_speed
// ============================================================

auto MaxWaveSpeed(const Fields& f, const SimConfig& cfg,
                      int local_L, int local_M,
                      const MPIManager& mpi) -> double {
    double local_max = 0.0;

    // Iterate over interior cells only: [1..local_L][1..local_M].
    // Ghost cells are excluded because they may hold stale or boundary
    // values that would produce an artificially large wave speed.
    #pragma omp parallel for collapse(2) reduction(max : local_max)
    for (int l = 1; l <= local_L; ++l) {
        for (int m = 1; m <= local_M; ++m) {
            const double cs = std::sqrt(cfg.gamma * f.p[l][m] / f.rho[l][m]);

            const double ca = std::sqrt((f.H_z[l][m]*f.H_z[l][m]
                                       + f.H_r[l][m]*f.H_r[l][m]
                                       + f.H_phi[l][m]*f.H_phi[l][m])
                                       / f.rho[l][m]);

            const double v = std::sqrt(f.v_z[l][m]*f.v_z[l][m]
                                     + f.v_r[l][m]*f.v_r[l][m]);

            local_max = std::max(local_max, v + cs + ca);
        }
    }

    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    return global_max;
}

// ============================================================
// compute_dt
// ============================================================

auto ComputeDt(const Fields& f, const SimConfig& cfg,
                  int local_L, int local_M,
                  const MPIManager& mpi,
                  double dt_current) -> double {
    const double speed = MaxWaveSpeed(f, cfg, local_L, local_M, mpi);

    // CFL-limited step: dt = C * dx / max_speed
    // A small epsilon prevents division by zero in a perfectly quiescent field.
    const double dx     = std::min(cfg.dz, cfg.dy);
    const double dt_cfl = cfg.cfl_number * dx / (speed + 1.0e-10);

    // Limit growth to prevent sudden jumps when wave speeds drop sharply.
    const double dt_grown = dt_current * cfg.dt_growth_factor;

    // Apply growth cap first, then hard bounds.
    return std::clamp(std::min(dt_cfl, dt_grown), cfg.dt_min, cfg.dt_max);
}

// ============================================================
// solution_change
// ============================================================

auto SolutionChange(const Fields& f, int local_L, int local_M) -> double {
    double sum_diff = 0.0;
    double sum_curr = 0.0;

    // Interior cells only.
    #pragma omp parallel for collapse(2) reduction(+ : sum_diff, sum_curr)
    for (int l = 1; l <= local_L; ++l) {
        for (int m = 1; m <= local_M; ++m) {
            auto sq = [](double x) { return x * x; };

            const double d_rho  = f.rho  [l][m] - f.rho_prev  [l][m];
            const double d_vz   = f.v_z  [l][m] - f.v_z_prev  [l][m];
            const double d_vr   = f.v_r  [l][m] - f.v_r_prev  [l][m];
            const double d_vphi = f.v_phi[l][m] - f.v_phi_prev[l][m];
            const double d_Hz   = f.H_z  [l][m] - f.H_z_prev  [l][m];
            const double d_Hr   = f.H_r  [l][m] - f.H_r_prev  [l][m];
            const double d_Hphi = f.H_phi[l][m] - f.H_phi_prev[l][m];

            sum_diff += sq(d_rho) + sq(d_vz)  + sq(d_vr)  + sq(d_vphi)
                      + sq(d_Hz) + sq(d_Hr)   + sq(d_Hphi);

            sum_curr += sq(f.rho[l][m])   + sq(f.v_z[l][m])
                      + sq(f.v_r[l][m])   + sq(f.v_phi[l][m])
                      + sq(f.H_z[l][m])   + sq(f.H_r[l][m])
                      + sq(f.H_phi[l][m]);
        }
    }

    double g_diff = 0.0, g_curr = 0.0;
    MPI_Allreduce(&sum_diff, &g_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sum_curr, &g_curr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    const double norm_diff = std::sqrt(g_diff);
    const double norm_curr = std::sqrt(g_curr);
    return (norm_curr > 1e-15) ? (norm_diff / norm_curr) : norm_diff;
}

// ============================================================
// check_cfl
// ============================================================

void CheckCfl(const Fields& f, const SimConfig& cfg,
               const MPIManager& mpi,
               int local_L, int local_M,
               double dt, int step_count) {
    const double speed  = MaxWaveSpeed(f, cfg, local_L, local_M, mpi);
    const double dx     = std::min(cfg.dz, cfg.dy);
    const double dt_max = cfg.cfl_number * dx / (speed + 1.0e-10);

    if (dt > dt_max && mpi.rank == 0 && step_count % 1000 == 0) {
        std::printf("WARNING [step %d]: dt=%.6e exceeds CFL limit dt_max=%.6e "
                    "(max_speed=%.3f, CFL=%.2f)\n",
                    step_count, dt, dt_max, speed, dt / dt_max * cfg.cfl_number);
    }
}

} // namespace Diagnostics
