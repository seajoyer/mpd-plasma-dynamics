#pragma once

#include "config.hpp"
#include "fields.hpp"
#include "mpi_manager.hpp"

/// Stateless diagnostic utilities operating on the field arrays.
/// All functions perform MPI reductions internally so they must be called
/// collectively by every rank.
namespace Diagnostics {

/// Compute the global maximum wave speed across all MPI ranks.
/// Includes sound speed, Alfvén speed, and bulk flow speed.
/// @return  max( |v| + c_s + c_a )  over all cells and all processes.
double max_wave_speed(const Fields& f, const SimConfig& cfg,
                      int local_L, int M_max);

/// Compute the relative change in the solution between the current
/// fields and the fields stored in the prev arrays.
/// Returns ||curr - prev|| / ||curr||  (L2 norm summed over all 7 fields).
double solution_change(const Fields& f, int local_L, int M_max);

/// Emit a CFL warning if the current time step exceeds the CFL limit.
/// Only prints on rank 0 and only every 1000 steps.
void check_cfl(const Fields& f, const SimConfig& cfg,
               const MPIManager& mpi,
               int local_L, int step_count);

} // namespace Diagnostics
