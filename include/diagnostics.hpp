#pragma once

#include "config.hpp"
#include "fields.hpp"
#include "grid.hpp"
#include "mpi_manager.hpp"

/// Stateless diagnostic utilities operating on the field arrays.
/// All functions that compute global quantities perform MPI reductions
/// internally and must therefore be called collectively by every rank.
namespace Diagnostics {

/// Compute the global maximum wave speed across all MPI ranks.
/// Includes sound speed, Alfvén speed, and bulk flow speed.
/// @return  max( |v| + c_s + c_a )  over all interior cells on all processes.
auto MaxWaveSpeed(const Fields& f, const SimConfig& cfg,
                      int local_L, int local_M,
                      const MPIManager& mpi) -> double;

/// Compute the global minimum physical cell size across all MPI ranks.
///
/// Returns  min( dz, min_l grid.dr[l] )  reduced over every rank.
auto MinPhysicalCellSize(const SimConfig& cfg, const Grid& grid,
                              int local_L,
                              const MPIManager& mpi) -> double;

/// Compute the next adaptive time step that satisfies the CFL condition.
///
/// The returned dt is:
///   1. Derived from the current global max wave speed:
///        dt_cfl = cfg.cfl_number * min(dz, min_l dr[l]) / (max_speed + ε)
///      where the radial minimum is reduced globally across all ranks.
///   2. Limited to at most cfg.dt_growth_factor × dt_current to prevent
///      sudden large increases when wave speeds drop.
///   3. Clamped to [cfg.dt_min, cfg.dt_max].
///
/// @param dt_current     The dt used in the step that just completed.
///                       Used to enforce the growth-rate limit.
/// @param prev_max_speed In/out: cached max wave speed from the last full
///                       Allreduce.  Negative → force a fresh value to be
///                       used (the Allreduce itself always runs because
///                       MPI collectives must be called by every rank).
///                       Updated on every call.
/// @param speed_rtol     Relative tolerance for the cache-reuse decision.
///                       Default 0.02 (2 %).
/// @return               Recommended dt for the *next* time step.
auto ComputeDt(const Fields& f, const SimConfig& cfg, const Grid& grid,
                  int local_L, int local_M,
                  const MPIManager& mpi,
                  double dt_current,
                  double& prev_max_speed,
                  double speed_rtol = 0.02) -> double;

/// Compute the relative change in the solution between the current
/// fields and the fields stored in the prev arrays.
/// Returns ||curr - prev|| / ||curr||  (L2 norm summed over all 7 fields).
auto SolutionChange(const Fields& f, int local_L, int local_M) -> double;

/// Emit a CFL warning if the current time step exceeds the CFL limit.
/// Only prints on rank 0 and only every 1000 steps.
void CheckCfl(const Fields& f, const SimConfig& cfg, const Grid& grid,
               const MPIManager& mpi,
               int local_L, int local_M,
               double dt, int step_count);

} // namespace Diagnostics
