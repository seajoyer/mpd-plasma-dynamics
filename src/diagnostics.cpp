#include "diagnostics.hpp"

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace Diagnostics {

// ============================================================
// max_wave_speed
// ============================================================
//
auto MaxWaveSpeed(const Fields& f, const SimConfig& cfg,
                      int local_L, int local_M,
                      const MPIManager& mpi) -> double {
    const double gamma_gamma_m1 = cfg.gamma * (cfg.gamma - 1.0);

    double local_max = 0.0;

    #pragma omp parallel for reduction(max : local_max)
    for (int l = 1; l <= local_L; ++l) {
        for (int m = 1; m <= local_M; ++m) {
            const double rho_inv = 1.0 / f.rho[l][m];

            const double cs2 = gamma_gamma_m1 * f.e[l][m];
            const double ca2 = (f.H_z  [l][m]*f.H_z  [l][m]
                              + f.H_r  [l][m]*f.H_r  [l][m]
                              + f.H_phi[l][m]*f.H_phi[l][m]) * rho_inv;
            const double v2  = f.v_z[l][m]*f.v_z[l][m]
                             + f.v_r[l][m]*f.v_r[l][m];

            const double speed = std::sqrt(v2) + std::sqrt(cs2) + std::sqrt(ca2);
            local_max = std::max(local_max, speed);
        }
    }

    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    return global_max;
}

// ============================================================
// min_physical_cell_size
// ============================================================
//
// Returns min(dz, min_l grid.dr[l]) reduced across every rank.
//
// The physical radial cell width is
//
//     grid.dr[l] = (r_outer(z_l) − r_inner(z_l)) / M_max,
//
auto MinPhysicalCellSize(const SimConfig& cfg, const Grid& grid,
                              int local_L,
                              const MPIManager& /*mpi*/) -> double {
    double local_min = std::numeric_limits<double>::infinity();

    // Scan owned interior cells only.  dr is 1-D in l, indexed 1..local_L
    // for the interior rows (index 0 is the left ghost row, and the right
    // ghost row sits at local_L + 1; both are excluded).
    for (int l = 1; l <= local_L; ++l) {
        local_min = std::min(local_min, grid.dr[l]);
    }

    double global_min = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);

    return std::min(cfg.dz, global_min);
}

// ============================================================
// compute_dt
// ============================================================
//
// Same c_s^2 = gamma*(gamma-1)*e identity as MaxWaveSpeed.
//
auto ComputeDt(const Fields& f, const SimConfig& cfg, const Grid& grid,
                  int local_L, int local_M,
                  const MPIManager& mpi,
                  double dt_current,
                  double& prev_max_speed,
                  double speed_rtol) -> double {

    // ── Step 1: local max wave speed ────────────────────────────────────
    const double gamma_gamma_m1 = cfg.gamma * (cfg.gamma - 1.0);

    double local_max = 0.0;

    #pragma omp parallel for reduction(max : local_max)
    for (int l = 1; l <= local_L; ++l) {
        for (int m = 1; m <= local_M; ++m) {
            const double rho_inv = 1.0 / f.rho[l][m];

            const double cs2 = gamma_gamma_m1 * f.e[l][m];
            const double ca2 = (f.H_z  [l][m]*f.H_z  [l][m]
                              + f.H_r  [l][m]*f.H_r  [l][m]
                              + f.H_phi[l][m]*f.H_phi[l][m]) * rho_inv;
            const double v2  = f.v_z[l][m]*f.v_z[l][m]
                             + f.v_r[l][m]*f.v_r[l][m];

            const double speed = std::sqrt(v2) + std::sqrt(cs2) + std::sqrt(ca2);
            local_max = std::max(local_max, speed);
        }
    }

    // ── Step 2: decide whether to run the global Allreduce ──────────────
    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    // Decide which speed to use for the dt formula.
    double speed_for_dt;
    if (prev_max_speed > 0.0 &&
        std::abs(global_max - prev_max_speed) <= speed_rtol * prev_max_speed) {
        // Flow is steady enough — reuse the cached value so dt is smooth.
        speed_for_dt = prev_max_speed;
    } else {
        // Speed has shifted noticeably; update the cache and use fresh value.
        prev_max_speed = global_max;
        speed_for_dt   = global_max;
    }

    // ── Step 3: CFL-limited dt ──────────────────────────────────────────
    //
    // dx is the global minimum of (dz, dr[l]) across every rank — i.e. the
    // smallest physical cell anywhere in the domain.
    const double dx     = MinPhysicalCellSize(cfg, grid, local_L, mpi);
    const double dt_cfl = cfg.cfl_number * dx / (speed_for_dt + 1.0e-10);

    // Limit growth to prevent sudden jumps when wave speeds drop sharply.
    const double dt_grown = dt_current * cfg.dt_growth_factor;

    return std::clamp(std::min(dt_cfl, dt_grown), cfg.dt_min, cfg.dt_max);
}

// ============================================================
// solution_change
// ============================================================
//
// L2 relative change of the seven primary fields between the current
// state and the previous snapshot.  The two partial sums (numerator and
// denominator) are coalesced into a single MPI_Allreduce on a 2-element
// buffer.
//
auto SolutionChange(const Fields& f, int local_L, int local_M) -> double {
    double sum_diff = 0.0;
    double sum_curr = 0.0;

    #pragma omp parallel for reduction(+ : sum_diff, sum_curr)
    for (int l = 1; l <= local_L; ++l) {
        for (int m = 1; m <= local_M; ++m) {
            auto sq = [](double x) -> double { return x * x; };

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

    double local_sums [2] = { sum_diff, sum_curr };
    double global_sums[2] = { 0.0,      0.0      };
    MPI_Allreduce(local_sums, global_sums, 2, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    const double norm_diff = std::sqrt(global_sums[0]);
    const double norm_curr = std::sqrt(global_sums[1]);
    return (norm_curr > 1e-15) ? (norm_diff / norm_curr) : norm_diff;
}

// ============================================================
// check_cfl
// ============================================================

void CheckCfl(const Fields& f, const SimConfig& cfg, const Grid& grid,
               const MPIManager& mpi,
               int local_L, int local_M,
               double dt, int step_count) {
    const double speed  = MaxWaveSpeed(f, cfg, local_L, local_M, mpi);
    const double dx     = MinPhysicalCellSize(cfg, grid, local_L, mpi);
    const double dt_max = cfg.cfl_number * dx / (speed + 1.0e-10);

    if (dt > dt_max && mpi.rank == 0 && step_count % 1000 == 0) {
        std::printf("WARNING [step %d]: dt=%.6e exceeds CFL limit dt_max=%.6e "
                    "(max_speed=%.3f, CFL=%.2f)\n",
                    step_count, dt, dt_max, speed, dt / dt_max * cfg.cfl_number);
    }
}

} // namespace Diagnostics
