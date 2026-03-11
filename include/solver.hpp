#pragma once

#include <vector>
#include "config.hpp"
#include "fields.hpp"
#include "grid.hpp"
#include "mpi_manager.hpp"

/// Owns the numerical time-stepping algorithm:
///   1. Ghost-cell exchange (MPI)
///   2. Lax–Friedrichs central update
///   3. Boundary conditions
///   4. Physical-variable reconstruction
///   5. Advance u0 ← u
///
/// The public interface is a single advance() call.
class Solver {
public:
    Solver(const SimConfig& cfg, const MPIManager& mpi,
           const Grid& grid, Fields& f);

    /// Execute one complete time step.
    void advance();

private:
    const SimConfig&  cfg_;
    const MPIManager& mpi_;
    const Grid&       grid_;
    Fields&           f_;

    // MPI scratch buffers (size M_max + 1)
    std::vector<double> sl_, sr_, rl_, rr_;

    // ---- sub-steps ----

    /// Exchange ghost cells for all 18 arrays (8 conservative + 10 physical).
    void exchange_all_ghosts();

    /// Lax–Friedrichs update for interior cells (l=1..local_L, m=1..M_max-1).
    void compute_central_update();

    /// Reconstruct physical vars from u for the central interior only.
    /// Called after compute_central_update so that the BC routines below
    /// can use up-to-date neighbour values.
    void update_central_physical();

    // ---- boundary conditions ----
    void apply_bc_left();             ///< inflow BC at z = 0  (rank 0 only)
    void apply_bc_right();            ///< outflow BC at z = L (last rank only)
    void apply_bc_upper();            ///< outer-wall BC (m = M_max)
    void apply_bc_lower_inner();      ///< axis BC for l ≤ L_end (m = 0)
    void apply_bc_lower_outer();      ///< axis BC for l > L_end (m = 0, free)

    /// Rebuild u from the physical values set by the BC routines
    /// at the cell (l, m).
    inline void rebuild_u_from_physical(int l, int m);
};
