#include "diagnostics.hpp"

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace Diagnostics {

// ============================================================
// max_wave_speed
// ============================================================

double max_wave_speed(const Fields& f, const SimConfig& cfg,
                      int local_L, int local_M,
                      const MPIManager& mpi) {
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
// solution_change
// ============================================================

double solution_change(const Fields& f, int local_L, int local_M) {
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

void check_cfl(const Fields& f, const SimConfig& cfg,
               const MPIManager& mpi,
               int local_L, int local_M, int step_count) {
    const double speed  = max_wave_speed(f, cfg, local_L, local_M, mpi);
    const double dx     = std::min(cfg.dz, cfg.dy);
    const double dt_max = 0.5 * dx / (speed + 1e-10);   // CFL = 0.5

    if (cfg.dt > dt_max && mpi.rank == 0 && step_count % 1000 == 0) {
        std::printf("WARNING: dt=%.6e exceeds CFL limit dt_max=%.6e "
                    "(max_speed=%.3f)\n", cfg.dt, dt_max, speed);
    }
}

} // namespace Diagnostics
