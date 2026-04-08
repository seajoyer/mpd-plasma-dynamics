#pragma once

#include "config.hpp"
#include "fields.hpp"
#include "mpi_manager.hpp"

/// Stateless diagnostic utilities operating on the field arrays.
/// All functions that compute global quantities perform MPI reductions
/// internally and must therefore be called collectively by every rank.
namespace Diagnostics {

/// Compute the global maximum wave speed across all MPI ranks.
/// Includes sound speed, Alfvén speed, and bulk flow speed.
/// @return  max( |v| + c_s + c_a )  over all interior cells on all processes.
double max_wave_speed(const Fields& f, const SimConfig& cfg,
                      int local_L, int local_M,
                      const MPIManager& mpi);

/// Compute the next adaptive time step that satisfies the CFL condition.
///
/// The returned dt is:
///   1. Derived from the current global max wave speed:
///        dt_cfl = cfg.cfl_number * min(dz, dy) / (max_speed + ε)
///   2. Limited to at most cfg.dt_growth_factor × dt_current to prevent
///      sudden large increases when wave speeds drop.
///   3. Clamped to [cfg.dt_min, cfg.dt_max].
///
/// This function performs one MPI_Allreduce (for max_wave_speed) and
/// must be called collectively by every rank.
///
/// @param dt_current  The dt used in the step that just completed.
///                    Used to enforce the growth-rate limit.
/// @return            Recommended dt for the *next* time step.
double compute_dt(const Fields& f, const SimConfig& cfg,
                  int local_L, int local_M,
                  const MPIManager& mpi,
                  double dt_current);

/// Compute the relative change in the solution between the current
/// fields and the fields stored in the prev arrays.
/// Returns ||curr - prev|| / ||curr||  (L2 norm summed over all 7 fields).
double solution_change(const Fields& f, int local_L, int local_M);

/// Emit a CFL warning if the current time step exceeds the CFL limit.
/// Only prints on rank 0 and only every 1000 steps.
/// Useful as a sanity check even when adaptive_dt is enabled.
void check_cfl(const Fields& f, const SimConfig& cfg,
               const MPIManager& mpi,
               int local_L, int local_M,
               double dt, int step_count);

} // namespace Diagnostics
